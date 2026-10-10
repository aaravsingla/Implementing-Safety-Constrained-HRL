"""v3 step 2 -- RLHF fine-tuning in the LLM recipe, with Safe-RLHF constraints.

    pi_ref (SFT / behavior cloning, make_reference.py)
        -> preference data on the current policy's segments (helpfulness + harmlessness + safety label)
        -> reward model R (pessimistic ensemble) and cost model C (Safe-RLHF loss)
        -> PPO-Lagrangian fine-tuning of pi, initialized from pi_ref, with a per-step KL penalty

    per-step reward :  r_t = standardize( mean_k R_k - kappa * std_k R_k )  -  beta * [log pi(a_t|s_t) - log pi_ref(a_t|s_t)]
    per-step cost   :  c_t = mean_k C_k + kappa * std_k C_k          (--cost-source rm, Safe-RLHF)
                       c_t = true hazard indicator                     (--cost-source env, ablation)
    constraint      :  #segments per episode the cost model judges unsafe (C(sigma) > 0) <= --unsafe-seg-limit
                       (or true episode cost <= --cost-limit for --cost-source env)

The KL term is the InstructGPT-style per-token penalty: its expectation under pi is KL(pi || pi_ref).
The observation normalizer is frozen at pi_ref's statistics so pi and pi_ref see identical inputs.
"""

import csv
import json
import os
import time

import numpy as np
import torch

from srl.envs import PreferenceOracleEnv
from srl.ppo_lag import ActorCritic, PPOLag, RunningMeanStd, gae
from srl.reward_model import label_schedule, make_segments, rm_features
from srl.safe_rlhf import SafePreferenceLearner
from train import build_parser, evaluate, rollout, summarize


def parse_args(argv=None):
    p = build_parser()
    p.set_defaults(epochs=25, out_dir="runs/v3", reward="rm")
    p.add_argument("--reference-dir", default=None, help="default: runs/v3/reference_s<seed>")
    p.add_argument("--kl-coef", type=float, default=0.1, help="beta; 0 disables the KL anchor")
    p.add_argument("--pessimism", type=float, default=1.0, help="kappa; 0 = ensemble mean")
    p.add_argument("--cost-source", choices=["rm", "env"], default="rm")
    p.add_argument("--unsafe-seg-limit", type=float, default=2.0, help="budget in predicted-unsafe 50-step segments per episode")
    p.add_argument("--safety-label-noise", type=float, default=0.1)
    p.add_argument("--critic-warmup", type=int, default=1, help="epochs of critic-only updates before moving the policy")
    p.add_argument("--init-log-std", type=float, default=None,
                   help="reset the fine-tuned policy's action log-std (pi_ref keeps its own); restores exploration")
    p.add_argument("--lambda-init", type=float, default=0.0)
    args = p.parse_args(argv)
    if args.name is None:
        args.name = f"v3_kl{args.kl_coef}_k{args.pessimism}_cost{args.cost_source}_s{args.seed}"
    args.reference_dir = args.reference_dir or os.path.join("runs/v3", f"reference_s{args.seed}")
    return args


def segment_sums(x, dones, seg_len):
    _, sums = make_segments(np.zeros((len(x), 1), np.float32), x, dones, seg_len)
    return sums


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

    # ---- reference (SFT) policy: initializes pi and anchors it via KL ----
    ref_ck = torch.load(os.path.join(args.reference_dir, "policy.pt"), weights_only=False)
    ref = ActorCritic(obs_dim, act_dim)
    ref.load_state_dict(ref_ck["ac"])
    ref.requires_grad_(False)
    agent = PPOLag(obs_dim, act_dim, args)
    agent.ac.pi.load_state_dict(ref.pi.state_dict())
    with torch.no_grad():
        agent.ac.log_std.copy_(ref.log_std if args.init_log_std is None else torch.full_like(ref.log_std, args.init_log_std))
    obs_rms = RunningMeanStd(obs_dim)
    obs_rms.mean, obs_rms.var = ref_ck["obs_mean"], ref_ck["obs_var"]

    feat_dim = rm_features(np.zeros((1, obs_dim)), np.zeros((1, act_dim)), np.zeros((1, obs_dim)), args.rm_features).shape[1]
    learner = SafePreferenceLearner(feat_dim, args, rng, learn_cost=args.cost_source == "rm")
    if args.cost_source == "rm":
        agent.lagrange.cost_limit = args.unsafe_seg_limit
    segs_per_ep = 1000 // args.seg_len

    obs, _ = env.reset(seed=args.seed)
    writer, total_steps, t0 = None, 0, time.time()
    with open(os.path.join(run_dir, "progress.csv"), "w", newline="") as f:
        for epoch in range(args.epochs):
            B, eps, obs, (last_r, last_c) = rollout(env, agent, obs_rms, obs, args.steps_per_epoch, args.gamma,
                                                   update_obs_norm=False)
            total_steps += args.steps_per_epoch
            row = {"epoch": epoch, "steps": total_steps, **summarize(eps)}
            row["unsafe_segs_true"] = float((segment_sums(B["cost"], B["done"], args.seg_len) > 0).mean() * segs_per_ep)

            # ---- preference collection + reward/cost model fitting ----
            feats = rm_features(B["raw"], B["act"], B["next_raw"], args.rm_features)
            n_new = label_schedule(epoch, args.epochs, args.label_budget)
            if n_new > 0:
                seg_feats, seg_rets = make_segments(feats, B["r_oracle"], B["done"], args.seg_len)
                seg_costs = segment_sums(B["cost"], B["done"], args.seg_len)
                learner.add_queries(seg_feats, seg_rets, seg_costs, n_new)
                row["rm_loss"] = learner.train(args.rm_steps_first if epoch == 0 else args.rm_steps)
            row.update(learner.metrics())
            row["n_labels"] = learner.n_labels

            # ---- reward: pessimistic RM + KL anchor to the reference ----
            r_hat, r_std = learner.predict_reward(feats, args.pessimism)
            row["rm_corr"] = float(np.corrcoef(r_hat, B["r_oracle"])[0, 1])
            row["rm_ens_std"] = float(r_std.mean())
            rew = (r_hat - r_hat.mean()) / (r_hat.std() + 1e-8)
            with torch.no_grad():
                ref_logp = ref.dist(torch.as_tensor(B["obs"])).log_prob(torch.as_tensor(B["act"])).sum(-1).numpy()
            log_ratio = B["logp"] - ref_logp
            row["kl_ref"] = float(log_ratio.mean())
            rew = rew - args.kl_coef * log_ratio

            # ---- cost: learned (Safe-RLHF) or true ----
            if args.cost_source == "rm":
                cost, c_std = learner.predict_cost(feats, args.pessimism)
                row["cm_corr"] = float(np.corrcoef(cost, B["cost"])[0, 1]) if B["cost"].std() > 0 else float("nan")
                row["unsafe_segs_pred"] = float((segment_sums(cost, B["done"], args.seg_len) > 0).mean() * segs_per_ep)
                constraint_value = row["unsafe_segs_pred"]
            else:
                cost = B["cost"]
                constraint_value = row["ep_cost"]
            row["constraint_value"] = constraint_value

            rew = rew + args.gamma * B["boot_r"]
            cost = cost + args.gamma * B["boot_c"]
            adv_r, ret_r = gae(rew, B["v_r"], last_r, B["done"], args.gamma, args.gae_lambda)
            adv_c, ret_c = gae(cost, B["v_c"], last_c, B["done"], args.gamma, args.gae_lambda)
            batch = dict(obs=B["obs"], act=B["act"], logp=B["logp"], adv_r=adv_r, adv_c=adv_c, ret_r=ret_r, ret_c=ret_c)
            row.update(agent.update(batch, constraint_value, train_pi=epoch >= args.critic_warmup))
            row["time"] = time.time() - t0

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()) + [k for k in ("rm_loss",) if k not in row])
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
            f.flush()
            print(f"[{args.name}] ep {epoch:3d} | ret {row['ep_ret']:6.2f} | cost {row['ep_cost']:6.1f} | "
                  f"oracle {row['ep_oracle_ret']:6.2f} | goals {row['goals_per_ep']:.1f} | lam {row['lambda']:.3f} | "
                  f"kl_ref {row['kl_ref']:.3f} | rm_corr {row['rm_corr']:.3f}"
                  + (f" | cm_corr {row['cm_corr']:.3f} unsafe_pred {row['unsafe_segs_pred']:.1f} true {row['unsafe_segs_true']:.1f}"
                     if args.cost_source == "rm" else "") + f" | {row['time']:.0f}s", flush=True)

    final = {"eval": evaluate(env, agent, obs_rms, args.eval_episodes, args.seed)}
    torch.save({"ac": agent.ac.state_dict(), "obs_mean": obs_rms.mean, "obs_var": obs_rms.var,
                "rm": learner.reward.state_dict(), "cm": learner.cost.state_dict() if learner.cost else None},
               os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"[{args.name}] eval: {final['eval']}")


if __name__ == "__main__":
    main()
