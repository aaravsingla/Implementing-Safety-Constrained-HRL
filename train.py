"""Train a PPO(-Lagrangian) agent on SafetyPointGoal1 with one of three reward sources:

  --reward env     : the environment's hand-coded task reward
  --reward oracle  : the hidden supervisor reward (upper bound, not available in practice)
  --reward rm      : a reward model learned online from pairwise preferences (RLHF)

and with (--constraint) or without (--no-constraint) the CMDP cost constraint J_C <= d.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from srl.envs import PreferenceOracleEnv
from srl.ppo_lag import PPOLag, RunningMeanStd, gae
from srl.reward_model import PreferenceLearner, label_schedule, make_segments, rm_features


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-id", default="SafetyPointGoal1-v0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=20000)
    # reward / constraint
    p.add_argument("--reward", choices=["env", "oracle", "rm"], default="rm")
    p.add_argument("--constraint", dest="use_constraint", action="store_true", default=True)
    p.add_argument("--no-constraint", dest="use_constraint", action="store_false")
    p.add_argument("--cost-limit", type=float, default=25.0)
    p.add_argument("--lambda-lr", type=float, default=0.01)
    p.add_argument("--lagrange-mode", choices=["sgd", "pid"], default="sgd")
    p.add_argument("--cost-adv-norm", choices=["standardize", "center"], default="standardize")
    # oracle (hidden human preference)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.03)
    # PPO
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--vf-lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--update-iters", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # reward model
    p.add_argument("--label-budget", type=int, default=2000)
    p.add_argument("--label-noise", type=float, default=0.1)
    p.add_argument("--seg-len", type=int, default=50)
    p.add_argument("--active-queries", dest="active_queries", action="store_true", default=True)
    p.add_argument("--random-queries", dest="active_queries", action="store_false")
    p.add_argument("--candidate-mult", type=int, default=10)
    p.add_argument("--rm-members", type=int, default=3)
    p.add_argument("--rm-hidden", type=int, default=128)
    p.add_argument("--rm-features", choices=["raw", "summary"], default="raw")
    p.add_argument("--rm-lr", type=float, default=3e-4)
    p.add_argument("--rm-weight-decay", type=float, default=1e-4)
    p.add_argument("--rm-batch", type=int, default=64)
    p.add_argument("--rm-steps", type=int, default=300)
    p.add_argument("--rm-steps-first", type=int, default=1500)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-episodes", type=int, default=10)
    return p


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.name is None:
        tag = f"{'ppolag' if args.use_constraint else 'ppo'}_{args.reward}"
        args.name = f"{tag}_s{args.seed}"
    return args


def rollout(env, agent, obs_rms, obs, n_steps, gamma, update_obs_norm=True):
    """Collect n_steps transitions. Returns flat buffers, per-episode stats and the next obs.
    update_obs_norm=False keeps the observation normalizer frozen (needed when the policy and its
    reference were pretrained under fixed normalization statistics)."""
    B = {k: [] for k in ["raw", "next_raw", "obs", "act", "logp", "v_r", "v_c",
                         "r_env", "r_oracle", "cost", "done", "boot_r", "boot_c"]}
    eps, ep = [], dict(ret=0.0, cost=0.0, oracle=0.0, near=0.0, goals=0.0, len=0)
    for _ in range(n_steps):
        if update_obs_norm:
            obs_rms.update(obs)
        no = obs_rms.normalize(obs)
        a, logp, v_r, v_c = agent.ac.act(torch.as_tensor(no, dtype=torch.float32))
        a = a.numpy()
        nxt, r, c, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
        done = term or trunc
        boot_r = boot_c = 0.0
        if trunc and not term:  # time-limit: bootstrap from V(s_T)
            nn_ = torch.as_tensor(obs_rms.normalize(nxt), dtype=torch.float32)
            with torch.no_grad():
                boot_r, boot_c = agent.ac.v_r(nn_).item(), agent.ac.v_c(nn_).item()
        for k, v in [("raw", obs), ("next_raw", nxt), ("obs", no), ("act", a), ("logp", logp.item()),
                     ("v_r", v_r.item()), ("v_c", v_c.item()), ("r_env", r), ("r_oracle", info["oracle_reward"]),
                     ("cost", c), ("done", float(done)), ("boot_r", boot_r), ("boot_c", boot_c)]:
            B[k].append(v)
        ep["ret"] += r; ep["cost"] += c; ep["oracle"] += info["oracle_reward"]
        ep["near"] += info["near_miss"]; ep["goals"] += info["goal_reached"]; ep["len"] += 1
        if done:
            eps.append(ep)
            ep = dict(ret=0.0, cost=0.0, oracle=0.0, near=0.0, goals=0.0, len=0)
            obs, _ = env.reset()
        else:
            obs = nxt
    B = {k: np.asarray(v, dtype=np.float32) for k, v in B.items()}
    with torch.no_grad():
        no = torch.as_tensor(obs_rms.normalize(obs), dtype=torch.float32)
        last_r, last_c = agent.ac.v_r(no).item(), agent.ac.v_c(no).item()
    if B["done"][-1]:
        last_r = last_c = 0.0
    return B, eps, obs, (last_r, last_c)


@torch.no_grad()
def evaluate(env, agent, obs_rms, n_episodes, seed):
    stats = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        ep = dict(ret=0.0, cost=0.0, oracle=0.0, near=0.0, goals=0.0, len=0)
        done = False
        while not done:
            a, *_ = agent.ac.act(torch.as_tensor(obs_rms.normalize(obs), dtype=torch.float32), deterministic=True)
            obs, r, c, term, trunc, info = env.step(np.clip(a.numpy(), -1.0, 1.0))
            ep["ret"] += r; ep["cost"] += c; ep["oracle"] += info["oracle_reward"]
            ep["near"] += info["near_miss"]; ep["goals"] += info["goal_reached"]; ep["len"] += 1
            done = term or trunc
        stats.append(ep)
    return summarize(stats)


def summarize(eps):
    return {
        "ep_ret": float(np.mean([e["ret"] for e in eps])),
        "ep_cost": float(np.mean([e["cost"] for e in eps])),
        "ep_oracle_ret": float(np.mean([e["oracle"] for e in eps])),
        "near_miss_rate": float(np.mean([e["near"] / e["len"] for e in eps])),
        "goals_per_ep": float(np.mean([e["goals"] for e in eps])),
    }


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
    agent = PPOLag(obs_dim, act_dim, args)
    obs_rms = RunningMeanStd(obs_dim)
    feat_dim = rm_features(np.zeros((1, obs_dim)), np.zeros((1, act_dim)), np.zeros((1, obs_dim)), args.rm_features).shape[1]
    learner = PreferenceLearner(feat_dim, args, rng) if args.reward == "rm" else None

    obs, _ = env.reset(seed=args.seed)
    log_path = os.path.join(run_dir, "progress.csv")
    writer, total_steps, t0 = None, 0, time.time()
    with open(log_path, "w", newline="") as f:
        for epoch in range(args.epochs):
            B, eps, obs, (last_r, last_c) = rollout(env, agent, obs_rms, obs, args.steps_per_epoch, args.gamma)
            total_steps += args.steps_per_epoch
            row = {"epoch": epoch, "steps": total_steps, **summarize(eps)}

            # ---- choose the reward the policy is optimized for ----
            if args.reward == "env":
                rew = B["r_env"]
            elif args.reward == "oracle":
                rew = B["r_oracle"]
            else:
                feats = rm_features(B["raw"], B["act"], B["next_raw"], args.rm_features)
                n_new = label_schedule(epoch, args.epochs, args.label_budget)
                if n_new > 0:
                    seg_feats, seg_rets = make_segments(feats, B["r_oracle"], B["done"], args.seg_len)
                    learner.add_queries(seg_feats, seg_rets, n_new)
                    row["rm_loss"] = learner.train(args.rm_steps_first if epoch == 0 else args.rm_steps)
                rew = learner.predict(feats)
                row["rm_corr"] = float(np.corrcoef(rew, B["r_oracle"])[0, 1])
                row["rm_val_acc"] = learner.accuracy(learner.val_pairs)
                row["rm_train_acc"] = learner.accuracy(learner.train_pairs[-500:])
                row["n_labels"] = learner.n_labels
                rew = (rew - rew.mean()) / (rew.std() + 1e-8)  # episodes have fixed length, so a shift is harmless

            rew = rew + args.gamma * B["boot_r"]
            cost = B["cost"] + args.gamma * B["boot_c"]
            adv_r, ret_r = gae(rew, B["v_r"], last_r, B["done"], args.gamma, args.gae_lambda)
            adv_c, ret_c = gae(cost, B["v_c"], last_c, B["done"], args.gamma, args.gae_lambda)
            batch = dict(obs=B["obs"], act=B["act"], logp=B["logp"], adv_r=adv_r, adv_c=adv_c, ret_r=ret_r, ret_c=ret_c)
            stats = agent.update(batch, row["ep_cost"])
            row.update(stats)
            row["time"] = time.time() - t0

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
            f.flush()
            print(f"[{args.name}] ep {epoch:3d} | ret {row['ep_ret']:6.2f} | cost {row['ep_cost']:6.1f} | "
                  f"oracle {row['ep_oracle_ret']:6.2f} | near {row['near_miss_rate']:.3f} | "
                  f"lam {row['lambda']:.3f}" + (f" | rm_acc {row['rm_val_acc']:.3f} corr {row['rm_corr']:.3f}"
                                                 if args.reward == "rm" else "") + f" | {row['time']:.0f}s", flush=True)

    final = {"eval": evaluate(env, agent, obs_rms, args.eval_episodes, args.seed)}
    torch.save({"ac": agent.ac.state_dict(), "obs_mean": obs_rms.mean, "obs_var": obs_rms.var,
                "rm": learner.model.state_dict() if learner else None}, os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"[{args.name}] eval: {final['eval']}")


if __name__ == "__main__":
    main()
