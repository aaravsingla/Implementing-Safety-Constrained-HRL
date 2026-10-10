"""Reward learning from pairwise trajectory preferences (Christiano et al., 2017).

Pipeline (run online, interleaved with policy optimization):
  1. Cut the latest rollouts into fixed-length segments sigma = (s_t, a_t, s_{t+1})_{t=1..L}.
  2. Propose candidate segment pairs and keep the ones the reward-model ensemble
     disagrees on most (active query selection).
  3. Ask the preference oracle which segment is better (simulated human with
     a fixed probability of answering wrongly).
  4. Fit an ensemble of reward networks with the Bradley-Terry likelihood
        P(sigma_i > sigma_j) = sigmoid( sum_t r(sigma_i,t) - sum_t r(sigma_j,t) )
  5. The policy is trained on the ensemble-mean predicted reward.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.ppo_lag import mlp


def _lidar_summary(obs):
    """Permutation-invariant lidar summaries: nearest goal reading and the two nearest hazard
    readings (SafetyPointGoal1 layout: goal lidar = obs[12:28], hazard lidar = obs[28:44]).
    Lidar is linear in distance (reading = (3 - dist) / 3), so these are distances in disguise."""
    hz = np.sort(obs[:, 28:44], axis=1)
    return np.stack([obs[:, 12:28].max(1), hz[:, -1], hz[:, -2]], axis=1)


def rm_features(obs, act, next_obs, kind="raw"):
    """Reward-model input for each transition.
    raw     : (s, a, s')
    summary : (s', s' - s, a, lidar summaries of s' and their change) -- exposes progress directly"""
    if kind == "raw":
        return np.concatenate([obs, act, next_obs], axis=1)
    s, s2 = _lidar_summary(obs), _lidar_summary(next_obs)
    return np.concatenate([next_obs, next_obs - obs, act, s2, s2 - s], axis=1).astype(np.float32)


class RewardEnsemble(nn.Module):
    def __init__(self, in_dim, n_members=3, hidden=(128, 128)):
        super().__init__()
        self.members = nn.ModuleList([mlp(in_dim, 1, hidden, act=nn.LeakyReLU) for _ in range(n_members)])
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))

    def forward(self, x):
        """x: (..., in_dim) -> per-member rewards (K, ...)."""
        x = (x - self.mu) / self.sigma
        return torch.stack([m(x).squeeze(-1) for m in self.members])


def bradley_terry_loss(ret_i, ret_j, label):
    """Cross-entropy between the BT preference probability and the (possibly soft) label.
    ret_*: predicted segment returns, label: P(i preferred over j) in {0, 0.5, 1}."""
    return F.binary_cross_entropy_with_logits(ret_i - ret_j, label)


class PreferenceOracle:
    """Simulated human: prefers the segment with higher hidden oracle return,
    but answers at random with probability `error_rate` (cf. Christiano et al.'s 10% error)."""

    def __init__(self, error_rate, rng):
        self.error_rate, self.rng = error_rate, rng

    def __call__(self, ret_i, ret_j):
        true = np.where(np.isclose(ret_i, ret_j), 0.5, (ret_i > ret_j).astype(np.float64))
        flip = self.rng.random(len(true)) < self.error_rate
        noisy = np.where(flip, self.rng.integers(0, 2, len(true)).astype(np.float64), true)
        return noisy, true


def make_segments(features, oracle_rewards, dones, seg_len):
    """Cut a flat buffer of concatenated episodes into non-overlapping segments that
    never cross an episode boundary. Returns (N, L, F) features and (N,) oracle returns."""
    ends = np.flatnonzero(dones) + 1
    starts = np.concatenate([[0], ends[:-1]])
    feats, rets = [], []
    for s, e in zip(starts, ends):
        for k in range(s, e - seg_len + 1, seg_len):
            feats.append(features[k : k + seg_len])
            rets.append(oracle_rewards[k : k + seg_len].sum())
    return np.stack(feats).astype(np.float32), np.asarray(rets)


class PreferenceLearner:
    def __init__(self, feat_dim, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.model = RewardEnsemble(feat_dim, cfg.rm_members, hidden=(cfg.rm_hidden, cfg.rm_hidden))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.rm_lr, weight_decay=cfg.rm_weight_decay)
        self.oracle = PreferenceOracle(cfg.label_noise, rng)
        cap = 2 * cfg.label_budget
        self.seg_store = np.zeros((cap, cfg.seg_len, feat_dim), dtype=np.float32)
        self.n_segs = 0
        self.train_pairs, self.val_pairs = [], []  # (i, j, noisy_label, true_label)
        self.n_labels = 0

    # ---------------- querying ----------------
    def _disagreement(self, seg_feats, i, j):
        with torch.no_grad():
            r = self.model(torch.as_tensor(seg_feats))  # (K, N, L)
            ret = r.sum(-1).numpy()  # (K, N)
        p = 1.0 / (1.0 + np.exp(-(ret[:, i] - ret[:, j])))
        return p.std(0)

    def add_queries(self, seg_feats, seg_rets, n_queries):
        n = len(seg_feats)
        n_cand = n_queries * (self.cfg.candidate_mult if self.cfg.active_queries else 1)
        i = self.rng.integers(0, n, n_cand)
        j = (i + self.rng.integers(1, n, n_cand)) % n  # j != i
        if self.cfg.active_queries and self.n_labels > 0:
            keep = np.argsort(-self._disagreement(seg_feats, i, j))[:n_queries]
        else:
            keep = np.arange(n_queries)
        i, j = i[keep], j[keep]
        noisy, true = self.oracle(seg_rets[i], seg_rets[j])

        # store segments (pairs are kept in a fixed-size store)
        idx_i = np.arange(self.n_segs, self.n_segs + len(i))
        idx_j = idx_i + len(i)
        self.seg_store[idx_i] = seg_feats[i]
        self.seg_store[idx_j] = seg_feats[j]
        self.n_segs += 2 * len(i)
        for a, b, y, yt in zip(idx_i, idx_j, noisy, true):
            # a val_frac share of labels is held out to measure generalization
            (self.val_pairs if self.rng.random() < self.cfg.val_frac else self.train_pairs).append((a, b, y, yt))
        self.n_labels += len(i)

    # ---------------- fitting ----------------
    def train(self, n_steps):
        segs = self.seg_store[: self.n_segs]
        flat = segs.reshape(-1, segs.shape[-1])
        self.model.mu.copy_(torch.as_tensor(flat.mean(0)))
        self.model.sigma.copy_(torch.as_tensor(flat.std(0) + 1e-6))

        pairs = np.asarray(self.train_pairs)
        store = torch.as_tensor(segs)
        losses = []
        for _ in range(n_steps):
            loss = 0.0
            for k, member in enumerate(self.model.members):
                # each member sees an independent minibatch -> diversity for disagreement
                b = pairs[self.rng.integers(0, len(pairs), self.cfg.rm_batch)]
                si, sj = store[b[:, 0].astype(int)], store[b[:, 1].astype(int)]
                y = torch.as_tensor(b[:, 2], dtype=torch.float32)
                ri = member((si - self.model.mu) / self.model.sigma).squeeze(-1).sum(-1)
                rj = member((sj - self.model.mu) / self.model.sigma).squeeze(-1).sum(-1)
                loss = loss + bradley_terry_loss(ri, rj, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item() / len(self.model.members))
        return float(np.mean(losses[-50:]))

    def accuracy(self, pairs, use_true=True):
        """Agreement of the ensemble-mean preference with the noiseless oracle label."""
        if not pairs:
            return float("nan")
        p = np.asarray(pairs)
        decided = p[:, 3] != 0.5
        p = p[decided]
        with torch.no_grad():
            r = self.model(torch.as_tensor(self.seg_store[: self.n_segs])).mean(0).sum(-1).numpy()
        pred = r[p[:, 0].astype(int)] > r[p[:, 1].astype(int)]
        target = p[:, 3 if use_true else 2] > 0.5
        return float((pred == target).mean())

    def predict(self, features):
        with torch.no_grad():
            return self.model(torch.as_tensor(features, dtype=torch.float32)).mean(0).numpy()


def label_schedule(epoch, total_epochs, budget, frac_first=0.25, active_frac=0.5):
    """How many new preference labels to request at a given epoch: a quarter of the
    budget up front, the rest spread evenly over the first half of training."""
    n_first = int(frac_first * budget)
    last = int(active_frac * total_epochs)
    if last <= 1:
        return budget if epoch == 0 else 0
    if epoch == 0:
        return n_first
    if epoch < last:
        rest = budget - n_first
        per = rest // (last - 1)
        return per + (rest - per * (last - 1) if epoch == last - 1 else 0)
    return 0
