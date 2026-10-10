"""v3 step 1 -- build the "SFT" reference policy by behavior cloning.

LLM analogy:  pretrained/SFT model = capable but not aligned.
Here: demonstrations come from the v2 *unconstrained* PPO agent (task return ~25, but ~50 hazard steps
per episode). We clone it with maximum likelihood (supervised fine-tuning on demonstrations) to get
pi_ref. pi_ref initializes the RLHF policy and anchors it through the KL penalty.

Output: runs/v3/reference_s<seed>/policy.pt (actor + frozen observation-normalization stats) and final.json.
"""

import argparse
import json
import os

import numpy as np
import torch

from srl.envs import PreferenceOracleEnv
from srl.ppo_lag import ActorCritic
from train import summarize


def run_policy(env, ac, mean, var, n_episodes, seed, deterministic):
    norm = lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10)
    obs_buf, act_buf, eps = [], [], []
    for e in range(n_episodes):
        obs, _ = env.reset(seed=seed + e)
        ep = dict(ret=0.0, cost=0.0, oracle=0.0, near=0.0, goals=0.0, len=0)
        done = False
        while not done:
            no = norm(obs)
            a, *_ = ac.act(torch.as_tensor(no, dtype=torch.float32), deterministic=deterministic)
            a = a.numpy()
            obs_buf.append(no)
            act_buf.append(a)
            obs, r, c, term, trunc, info = env.step(np.clip(a, -1, 1))
            ep["ret"] += r; ep["cost"] += c; ep["oracle"] += info["oracle_reward"]
            ep["near"] += info["near_miss"]; ep["goals"] += info["goal_reached"]; ep["len"] += 1
            done = term or trunc
        eps.append(ep)
    return np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32), summarize(eps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expert-run", default=None, help="default: runs/v2/ppo_env_s<seed>")
    p.add_argument("--demo-episodes", type=int, default=30)
    p.add_argument("--bc-epochs", type=int, default=40)
    p.add_argument("--out-dir", default="runs/v3")
    args = p.parse_args()
    expert_run = args.expert_run or f"runs/v2/ppo_env_s{args.seed}"
    run_dir = os.path.join(args.out_dir, f"reference_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    env = PreferenceOracleEnv()
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    ck = torch.load(os.path.join(expert_run, "model.pt"), weights_only=False)
    expert = ActorCritic(obs_dim, act_dim)
    expert.load_state_dict(ck["ac"])
    mean, var = ck["obs_mean"], ck["obs_var"]

    # demonstrations: stochastic expert actions (keeps the action distribution, not just its mean)
    X, A, demo_stats = run_policy(env, expert, mean, var, args.demo_episodes, seed=1000 * (args.seed + 1), deterministic=False)

    # behavior cloning = maximum likelihood of demonstrated actions under a Gaussian policy
    bc = ActorCritic(obs_dim, act_dim)
    opt = torch.optim.Adam(list(bc.pi.parameters()) + [bc.log_std], lr=1e-3)
    Xt, At = torch.as_tensor(X), torch.as_tensor(A)
    n_val = len(X) // 10
    perm = torch.randperm(len(X))
    va, tr = perm[:n_val], perm[n_val:]
    log = []
    for ep in range(args.bc_epochs):
        for b in torch.randperm(len(tr)).split(256):
            idx = tr[b]
            loss = -bc.dist(Xt[idx]).log_prob(At[idx]).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_nll = -bc.dist(Xt[va]).log_prob(At[va]).sum(-1).mean().item()
        log.append({"epoch": ep, "train_nll": loss.item(), "val_nll": val_nll})

    _, _, bc_stats = run_policy(env, bc, mean, var, 10, seed=args.seed + 10_000, deterministic=True)
    torch.save({"ac": bc.state_dict(), "obs_mean": mean, "obs_var": var}, os.path.join(run_dir, "policy.pt"))
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump({"eval": bc_stats, "expert_demo_stats": demo_stats, "bc_log": log,
                   "n_demo_steps": int(len(X)), "expert_run": expert_run}, f, indent=2)
    print(f"[reference_s{args.seed}] demos {demo_stats}\n  BC val NLL {log[-1]['val_nll']:.3f}\n  BC eval {bc_stats}")


if __name__ == "__main__":
    main()
