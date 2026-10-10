"""PPO-Lagrangian for a Constrained MDP.

    max_pi  J_R(pi)   s.t.  J_C(pi) <= d

Lagrangian relaxation gives the saddle-point problem

    min_{lambda >= 0} max_pi  J_R(pi) - lambda * (J_C(pi) - d)

solved by alternating
  * dual step   : lambda <- max(0, lambda + eta * (J_C - d))      (projected gradient ascent)
  * primal step : PPO-clip on the combined advantage (A_R - lambda * A_C) / (1 + lambda)

Reward and cost each get their own critic and GAE estimate, so the penalty is
assigned to the timesteps that actually incurred cost (per-step credit assignment).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


def mlp(in_dim, out_dim, hidden=(64, 64), act=nn.Tanh):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        self.v_r = mlp(obs_dim, 1, hidden)  # reward critic
        self.v_c = mlp(obs_dim, 1, hidden)  # cost critic

    def dist(self, obs):
        return Normal(self.pi(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.v_r(obs).squeeze(-1), self.v_c(obs).squeeze(-1)


def gae(rewards, values, last_value, dones, gamma, lam):
    """Generalized Advantage Estimation over a flat buffer of concatenated episodes.

    dones[t] = 1 cuts the recursion at an episode boundary. For time-limit
    truncation the caller folds gamma * V(s_T) into rewards[t], so the cut does
    not bias the estimate. `last_value` bootstraps a buffer that ends mid-episode.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


class Lagrange:
    """Dual variable. 'sgd' = projected gradient ascent; 'pid' = PID Lagrangian
    (Stooke et al., 2020), which damps the oscillation of plain dual ascent."""

    def __init__(self, cost_limit, lr=0.01, init=0.0, mode="sgd", kp=0.05, ki=0.01, kd=0.05):
        self.cost_limit, self.lr, self.mode = cost_limit, lr, mode
        self.value = init
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral, self.prev_cost = 0.0, None

    def update(self, ep_cost):
        err = ep_cost - self.cost_limit
        if self.mode == "sgd":
            self.value = max(0.0, self.value + self.lr * err)
        else:
            self.integral = max(0.0, self.integral + err)
            deriv = 0.0 if self.prev_cost is None else max(0.0, ep_cost - self.prev_cost)
            self.prev_cost = ep_cost
            self.value = max(0.0, self.kp * err + self.ki * self.integral + self.kd * deriv)
        return self.value


class PPOLag:
    def __init__(self, obs_dim, act_dim, cfg):
        self.cfg = cfg
        self.ac = ActorCritic(obs_dim, act_dim)
        self.pi_opt = torch.optim.Adam(list(self.ac.pi.parameters()) + [self.ac.log_std], lr=cfg.pi_lr)
        self.v_opt = torch.optim.Adam(
            list(self.ac.v_r.parameters()) + list(self.ac.v_c.parameters()), lr=cfg.vf_lr
        )
        self.lagrange = Lagrange(cfg.cost_limit, lr=cfg.lambda_lr, init=getattr(cfg, "lambda_init", 0.0), mode=cfg.lagrange_mode)

    def update(self, batch, ep_cost, train_pi=True):
        """batch: dict of numpy arrays (obs, act, logp, adv_r, adv_c, ret_r, ret_c).
        train_pi=False only fits the critics (critic warm-up when starting from a pretrained policy)."""
        cfg = self.cfg
        lam = self.lagrange.update(ep_cost) if cfg.use_constraint else 0.0

        # Both advantages are standardized so that lambda alone sets the reward/cost trade-off.
        # (Only centering A_C lets its raw scale -- O(10) for discounted costs -- swamp A_R.)
        adv_r = (batch["adv_r"] - batch["adv_r"].mean()) / (batch["adv_r"].std() + 1e-8)
        adv_c = batch["adv_c"] - batch["adv_c"].mean()
        if cfg.cost_adv_norm == "standardize":
            adv_c = adv_c / (batch["adv_c"].std() + 1e-8)
        adv = (adv_r - lam * adv_c) / (1.0 + lam)

        t = lambda x: torch.as_tensor(x, dtype=torch.float32)
        obs, act, logp_old = t(batch["obs"]), t(batch["act"]), t(batch["logp"])
        adv, ret_r, ret_c = t(adv), t(batch["ret_r"]), t(batch["ret_c"])

        n = len(obs)
        stats = {"kl": 0.0, "pi_iters": 0}
        for it in range(cfg.update_iters):
            perm = torch.randperm(n)
            for start in range(0, n, cfg.minibatch):
                idx = perm[start : start + cfg.minibatch]
                d = self.ac.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = torch.exp(logp - logp_old[idx])
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]
                loss_pi = -torch.min(ratio * adv[idx], clipped).mean() - cfg.ent_coef * d.entropy().sum(-1).mean()
                if train_pi:
                    self.pi_opt.zero_grad()
                    loss_pi.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                    self.pi_opt.step()

                loss_v = ((self.ac.v_r(obs[idx]).squeeze(-1) - ret_r[idx]) ** 2).mean() + (
                    (self.ac.v_c(obs[idx]).squeeze(-1) - ret_c[idx]) ** 2
                ).mean()
                self.v_opt.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.v_opt.step()
            with torch.no_grad():
                kl = (logp_old - self.ac.dist(obs).log_prob(act).sum(-1)).mean().item()
            stats["kl"], stats["pi_iters"] = kl, it + 1
            if train_pi and kl > cfg.target_kl:  # early stop: trust-region surrogate
                break
        stats["lambda"] = lam
        return stats
