"""Hierarchical (Manager-Worker) variant of safe RL from preferences.

Manager (high level, acts every k steps)
    input : normalized observation
    output: a subgoal = a 2D displacement in the agent's egocentric frame,
            scaled to at most `subgoal_scale` metres and fixed in the world for k steps
    reward: sum over the k primitive steps of the task reward (env / oracle / learned RM)
    cost  : sum of hazard costs over the k steps -> manager is PPO-Lagrangian at the
            semi-MDP level (discount gamma^k), so it learns not to propose unsafe subgoals

Worker (low level, acts every step)
    input : [normalized observation, subgoal expressed in the current egocentric frame]
    reward: intrinsic progress towards the subgoal, (d_t - d_{t+1}) / subgoal_scale
    cost  : per-step hazard cost -> worker is also PPO-Lagrangian, so it must reach
            subgoals without cutting through hazards

Both levels are trained on-policy every epoch, which sidesteps HIRO's off-policy
relabeling problem (the manager's data always comes from the current worker).
"""

import csv
import json
import os
import time

import numpy as np
import torch

from srl.envs import PreferenceOracleEnv
from srl.ppo_lag import PPOLag, RunningMeanStd, gae
from srl.reward_model import PreferenceLearner, label_schedule, make_segments, rm_features
from train import build_parser, summarize


def parse_args(argv=None):
    p = build_parser()
    p.add_argument("--k", type=int, default=10, help="manager decision interval (steps)")
    p.add_argument("--subgoal-scale", type=float, default=0.5, help="max subgoal distance (m)")
    args = p.parse_args(argv)
    if args.name is None:
        args.name = f"hrl_{'ppolag' if args.use_constraint else 'ppo'}_{args.reward}_s{args.seed}"
    return args


class Hierarchy:
    def __init__(self, env, obs_dim, act_dim, args):
        self.env, self.args = env, args
        self.manager = PPOLag(obs_dim, 2, args)
        self.worker = PPOLag(obs_dim + 2, act_dim, args)
        self.goal = None

    def _agent(self):
        a = self.env.env.unwrapped.task.agent
        return a.pos[:2].copy(), a.mat[:2, :2].copy()

    def set_subgoal(self, m):
        pos, R = self._agent()
        self.goal = pos + R @ (np.clip(m, -1, 1) * self.args.subgoal_scale)

    def goal_ego(self):
        pos, R = self._agent()
        return np.clip(R.T @ (self.goal - pos) / self.args.subgoal_scale, -3, 3), np.linalg.norm(self.goal - pos)


def rollout(env, H, obs_rms, obs, n_steps, args):
    k = args.k
    W = {key: [] for key in ["raw", "next_raw", "obs", "act", "logp", "v_r", "v_c", "r_int",
                             "r_env", "r_oracle", "cost", "done", "boot_r", "boot_c", "mgr_idx"]}
    M = {key: [] for key in ["obs", "act", "logp", "v_r", "v_c", "done", "boot_r", "boot_c"]}
    eps, ep = [], dict(ret=0.0, cost=0.0, oracle=0.0, near=0.0, goals=0.0, len=0)
    t_ep = 0
    for _ in range(n_steps):
        obs_rms.update(obs)
        no = obs_rms.normalize(obs)
        if t_ep % k == 0:
            m, lp, vr, vc = H.manager.ac.act(torch.as_tensor(no, dtype=torch.float32))
            H.set_subgoal(m.numpy())
            for key, v in [("obs", no), ("act", m.numpy()), ("logp", lp.item()), ("v_r", vr.item()), ("v_c", vc.item())]:
                M[key].append(v)
        g, d_prev = H.goal_ego()
        wo = np.concatenate([no, g])
        a, lp, vr, vc = H.worker.ac.act(torch.as_tensor(wo, dtype=torch.float32))
        a = a.numpy()
        nxt, r, c, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
        _, d_now = H.goal_ego()
        done = term or trunc
        t_ep += 1
        mgr_done = done or t_ep % k == 0

        boot_r = boot_c = 0.0
        nn_ = obs_rms.normalize(nxt)
        if trunc and not term:
            g2, _ = H.goal_ego()
            with torch.no_grad():
                x = torch.as_tensor(np.concatenate([nn_, g2]), dtype=torch.float32)
                boot_r, boot_c = H.worker.ac.v_r(x).item(), H.worker.ac.v_c(x).item()
        # worker sees the subgoal as an option: its own episode continues across subgoal switches
        for key, v in [("raw", obs), ("next_raw", nxt), ("obs", wo), ("act", a), ("logp", lp.item()),
                       ("v_r", vr.item()), ("v_c", vc.item()), ("r_int", (d_prev - d_now) / args.subgoal_scale),
                       ("r_env", r), ("r_oracle", info["oracle_reward"]), ("cost", c), ("done", float(done)),
                       ("boot_r", boot_r), ("boot_c", boot_c), ("mgr_idx", len(M["obs"]) - 1)]:
            W[key].append(v)
        if mgr_done:
            mb_r = mb_c = 0.0
            if trunc and not term:
                with torch.no_grad():
                    x = torch.as_tensor(nn_, dtype=torch.float32)
                    mb_r, mb_c = H.manager.ac.v_r(x).item(), H.manager.ac.v_c(x).item()
            M["done"].append(float(done)); M["boot_r"].append(mb_r); M["boot_c"].append(mb_c)

        ep["ret"] += r; ep["cost"] += c; ep["oracle"] += info["oracle_reward"]
        ep["near"] += info["near_miss"]; ep["goals"] += info["goal_reached"]; ep["len"] += 1
        if done:
            eps.append(ep)
            ep = dict(ret=0.0, cost=0.0, oracle=0.0, near=0.0, goals=0.0, len=0)
            obs, _ = env.reset()
            t_ep = 0
        else:
            obs = nxt
    assert W["done"][-1] == 1.0, "steps_per_epoch must be a multiple of the episode length"
    W = {key: np.asarray(v, dtype=np.float32) for key, v in W.items()}
    M = {key: np.asarray(v, dtype=np.float32) for key, v in M.items()}
    return W, M, eps, obs


@torch.no_grad()
def evaluate(env, H, obs_rms, args):
    stats = []
    for i in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + 10_000 + i)
        ep = dict(ret=0.0, cost=0.0, oracle=0.0, near=0.0, goals=0.0, len=0)
        done, t = False, 0
        while not done:
            no = obs_rms.normalize(obs)
            if t % args.k == 0:
                m, *_ = H.manager.ac.act(torch.as_tensor(no, dtype=torch.float32), deterministic=True)
                H.set_subgoal(m.numpy())
            g, _ = H.goal_ego()
            a, *_ = H.worker.ac.act(torch.as_tensor(np.concatenate([no, g]), dtype=torch.float32), deterministic=True)
            obs, r, c, term, trunc, info = env.step(np.clip(a.numpy(), -1.0, 1.0))
            ep["ret"] += r; ep["cost"] += c; ep["oracle"] += info["oracle_reward"]
            ep["near"] += info["near_miss"]; ep["goals"] += info["goal_reached"]; ep["len"] += 1
            done, t = term or trunc, t + 1
        stats.append(ep)
    return summarize(stats)


def main(argv=None):
    args = parse_args(argv)
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = PreferenceOracleEnv(args.env_id, margin=args.margin, alpha=args.alpha)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    H = Hierarchy(env, obs_dim, act_dim, args)
    obs_rms = RunningMeanStd(obs_dim)
    feat_dim = rm_features(np.zeros((1, obs_dim)), np.zeros((1, act_dim)), np.zeros((1, obs_dim)), args.rm_features).shape[1]
    learner = PreferenceLearner(feat_dim, args, rng) if args.reward == "rm" else None
    gamma_m = args.gamma ** args.k

    obs, _ = env.reset(seed=args.seed)
    writer, total_steps, t0 = None, 0, time.time()
    with open(os.path.join(run_dir, "progress.csv"), "w", newline="") as f:
        for epoch in range(args.epochs):
            W, M, eps, obs = rollout(env, H, obs_rms, obs, args.steps_per_epoch, args)
            total_steps += args.steps_per_epoch
            row = {"epoch": epoch, "steps": total_steps, **summarize(eps)}

            if args.reward == "env":
                rew = W["r_env"]
            elif args.reward == "oracle":
                rew = W["r_oracle"]
            else:
                feats = rm_features(W["raw"], W["act"], W["next_raw"], args.rm_features)
                n_new = label_schedule(epoch, args.epochs, args.label_budget)
                if n_new > 0:
                    seg_feats, seg_rets = make_segments(feats, W["r_oracle"], W["done"], args.seg_len)
                    learner.add_queries(seg_feats, seg_rets, n_new)
                    row["rm_loss"] = learner.train(args.rm_steps_first if epoch == 0 else args.rm_steps)
                rew = learner.predict(feats)
                row["rm_corr"] = float(np.corrcoef(rew, W["r_oracle"])[0, 1])
                row["rm_val_acc"] = learner.accuracy(learner.val_pairs)
                row["n_labels"] = learner.n_labels
                rew = (rew - rew.mean()) / (rew.std() + 1e-8)

            # ---- manager: semi-MDP transitions, reward/cost summed over k steps ----
            idx = W["mgr_idx"].astype(int)
            m_rew = np.bincount(idx, weights=rew, minlength=len(M["obs"])) + gamma_m * M["boot_r"]
            m_cost = np.bincount(idx, weights=W["cost"], minlength=len(M["obs"])) + gamma_m * M["boot_c"]
            adv_r, ret_r = gae(m_rew, M["v_r"], 0.0, M["done"], gamma_m, args.gae_lambda)
            adv_c, ret_c = gae(m_cost, M["v_c"], 0.0, M["done"], gamma_m, args.gae_lambda)
            m_stats = H.manager.update(dict(obs=M["obs"], act=M["act"], logp=M["logp"], adv_r=adv_r,
                                            adv_c=adv_c, ret_r=ret_r, ret_c=ret_c), row["ep_cost"])

            # ---- worker: intrinsic subgoal-reaching reward + hazard cost ----
            w_rew = W["r_int"] + args.gamma * W["boot_r"]
            w_cost = W["cost"] + args.gamma * W["boot_c"]
            adv_r, ret_r = gae(w_rew, W["v_r"], 0.0, W["done"], args.gamma, args.gae_lambda)
            adv_c, ret_c = gae(w_cost, W["v_c"], 0.0, W["done"], args.gamma, args.gae_lambda)
            w_stats = H.worker.update(dict(obs=W["obs"], act=W["act"], logp=W["logp"], adv_r=adv_r,
                                           adv_c=adv_c, ret_r=ret_r, ret_c=ret_c), row["ep_cost"])

            row.update({"lambda": m_stats["lambda"], "lambda_worker": w_stats["lambda"],
                        "kl": m_stats["kl"], "kl_worker": w_stats["kl"],
                        "intrinsic_per_step": float(W["r_int"].mean()), "time": time.time() - t0})
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
            f.flush()
            print(f"[{args.name}] ep {epoch:3d} | ret {row['ep_ret']:6.2f} | cost {row['ep_cost']:6.1f} | "
                  f"oracle {row['ep_oracle_ret']:6.2f} | near {row['near_miss_rate']:.3f} | lam {row['lambda']:.3f}/"
                  f"{row['lambda_worker']:.3f} | int {row['intrinsic_per_step']:.4f}"
                  + (f" | rm_acc {row['rm_val_acc']:.3f} corr {row['rm_corr']:.3f}" if args.reward == "rm" else "")
                  + f" | {row['time']:.0f}s", flush=True)

    final = {"eval": evaluate(env, H, obs_rms, args)}
    torch.save({"manager": H.manager.ac.state_dict(), "worker": H.worker.ac.state_dict(),
                "obs_mean": obs_rms.mean, "obs_var": obs_rms.var,
                "rm": learner.model.state_dict() if learner else None}, os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"[{args.name}] eval: {final['eval']}")


if __name__ == "__main__":
    main()
