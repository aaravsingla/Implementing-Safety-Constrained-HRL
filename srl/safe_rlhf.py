"""v3: Safe-RLHF-style preference learning with a pessimistic reward ensemble and a learned cost.

Every queried segment pair is answered twice by the simulated annotator, as in Safe-RLHF (Dai et al., 2024):
  * helpfulness  : which segment better matches the supervisor's task preference  -> reward model R
  * harmlessness : which segment is *less safe*                                   -> cost model C
and every segment additionally gets a binary safety label (did it enter a hazard?).

Reward model  : Bradley-Terry on helpfulness preferences (as in v2).
Cost model    : Bradley-Terry on harmlessness preferences + a classification term that anchors the
                sign of the segment cost, so that C(sigma) > 0  <=>  "sigma is unsafe":
                    L_C = -log s(C_w - C_l) - log s(z_w * C_w) - log s(z_l * C_l),   z = +1 unsafe, -1 safe
                (the cost-model loss of Safe-RLHF, with "w" the less safe segment).
Pessimism     : the policy is trained on  r = mean_k r_k - kappa * std_k r_k   and
                                           c = mean_k c_k + kappa * std_k c_k
                i.e. it assumes the worst plausible reward and cost where the ensemble disagrees,
                which discourages exploiting reward-model errors in unfamiliar states.
"""

import numpy as np
import torch
import torch.nn.functional as F

from srl.reward_model import PreferenceOracle, RewardEnsemble, bradley_terry_loss


class SafePreferenceLearner:
    def __init__(self, feat_dim, cfg, rng, learn_cost=True):
        self.cfg, self.rng, self.learn_cost = cfg, rng, learn_cost
        hidden = (cfg.rm_hidden, cfg.rm_hidden)
        self.reward = RewardEnsemble(feat_dim, cfg.rm_members, hidden)
        self.cost = RewardEnsemble(feat_dim, cfg.rm_members, hidden) if learn_cost else None
        params = list(self.reward.parameters()) + (list(self.cost.parameters()) if learn_cost else [])
        self.opt = torch.optim.Adam(params, lr=cfg.rm_lr, weight_decay=cfg.rm_weight_decay)
        self.oracle = PreferenceOracle(cfg.label_noise, rng)
        cap = 2 * cfg.label_budget
        self.seg_store = np.zeros((cap, cfg.seg_len, feat_dim), dtype=np.float32)
        self.unsafe = np.zeros(cap)       # noisy binary safety label per stored segment
        self.unsafe_true = np.zeros(cap)
        self.n_segs = 0
        # (i, j, reward_label, reward_label_true, cost_label, cost_label_true); labels = P(i preferred / i less safe)
        self.train_pairs, self.val_pairs = [], []
        self.n_labels = 0

    # ---------------- querying ----------------
    @staticmethod
    def _disagreement(model, seg_feats, i, j):
        with torch.no_grad():
            ret = model(torch.as_tensor(seg_feats)).sum(-1).numpy()  # (K, N)
        p = 1.0 / (1.0 + np.exp(-(ret[:, i] - ret[:, j])))
        return p.std(0)

    def add_queries(self, seg_feats, seg_rets, seg_costs, n_queries):
        n = len(seg_feats)
        n_cand = n_queries * self.cfg.candidate_mult
        i = self.rng.integers(0, n, n_cand)
        j = (i + self.rng.integers(1, n, n_cand)) % n
        if self.n_labels > 0:
            score = self._disagreement(self.reward, seg_feats, i, j)
            if self.learn_cost:
                score = score + self._disagreement(self.cost, seg_feats, i, j)
            keep = np.argsort(-score)[:n_queries]
        else:
            keep = np.arange(n_queries)
        i, j = i[keep], j[keep]
        y_r, y_r_true = self.oracle(seg_rets[i], seg_rets[j])
        y_c, y_c_true = self.oracle(seg_costs[i], seg_costs[j])  # 1 = segment i has MORE cost (less safe)

        idx_i = np.arange(self.n_segs, self.n_segs + len(i))
        idx_j = idx_i + len(i)
        for idx, src in ((idx_i, i), (idx_j, j)):
            self.seg_store[idx] = seg_feats[src]
            true = (seg_costs[src] > 0).astype(np.float64)
            flip = self.rng.random(len(src)) < self.cfg.safety_label_noise
            self.unsafe_true[idx] = true
            self.unsafe[idx] = np.where(flip, 1.0 - true, true)
        self.n_segs += 2 * len(i)
        for row in zip(idx_i, idx_j, y_r, y_r_true, y_c, y_c_true):
            (self.val_pairs if self.rng.random() < self.cfg.val_frac else self.train_pairs).append(row)
        self.n_labels += len(i)

    # ---------------- fitting ----------------
    def train(self, n_steps):
        segs = self.seg_store[: self.n_segs]
        flat = segs.reshape(-1, segs.shape[-1])
        mu, sigma = torch.as_tensor(flat.mean(0)), torch.as_tensor(flat.std(0) + 1e-6)
        for m in [self.reward] + ([self.cost] if self.learn_cost else []):
            m.mu.copy_(mu)
            m.sigma.copy_(sigma)
        store = torch.as_tensor(segs)
        unsafe = torch.as_tensor(self.unsafe[: self.n_segs], dtype=torch.float32)
        pairs = np.asarray(self.train_pairs)
        losses = []
        for _ in range(n_steps):
            loss = 0.0
            for k in range(self.cfg.rm_members):
                b = pairs[self.rng.integers(0, len(pairs), self.cfg.rm_batch)]
                ii, jj = b[:, 0].astype(int), b[:, 1].astype(int)
                si, sj = (store[ii] - mu) / sigma, (store[jj] - mu) / sigma
                rm = self.reward.members[k]
                ri, rj = rm(si).squeeze(-1).sum(-1), rm(sj).squeeze(-1).sum(-1)
                loss = loss + bradley_terry_loss(ri, rj, torch.as_tensor(b[:, 2], dtype=torch.float32))
                if self.learn_cost:
                    cm = self.cost.members[k]
                    ci, cj = cm(si).squeeze(-1).sum(-1), cm(sj).squeeze(-1).sum(-1)
                    loss = loss + bradley_terry_loss(ci, cj, torch.as_tensor(b[:, 4], dtype=torch.float32))
                    # sign anchor: C > 0 for unsafe segments, C < 0 for safe ones
                    loss = loss + 0.5 * (F.binary_cross_entropy_with_logits(ci, unsafe[ii])
                                         + F.binary_cross_entropy_with_logits(cj, unsafe[jj]))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item() / self.cfg.rm_members)
        return float(np.mean(losses[-50:]))

    # ---------------- inference ----------------
    def predict_reward(self, features, kappa):
        """Pessimistic reward. Bradley-Terry fixes each member only up to an additive constant (and its
        scale is arbitrary), so members are standardized before their disagreement is measured;
        otherwise std_k would mostly reflect offsets, not uncertainty."""
        with torch.no_grad():
            r = self.reward(torch.as_tensor(features, dtype=torch.float32)).numpy()  # (K, N)
        r = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-8)
        return r.mean(0) - kappa * r.std(0), r.std(0)

    def predict_cost(self, features, kappa):
        """Pessimistic cost. No per-member standardization here: the classification term pins the
        zero point (C > 0 = unsafe), so members are already on a common, meaningful scale."""
        with torch.no_grad():
            c = self.cost(torch.as_tensor(features, dtype=torch.float32)).numpy()
        return c.mean(0) + kappa * c.std(0), c.std(0)

    # ---------------- diagnostics (against the noiseless annotator) ----------------
    def _seg_scores(self, model):
        with torch.no_grad():
            return model(torch.as_tensor(self.seg_store[: self.n_segs])).mean(0).sum(-1).numpy()

    def metrics(self):
        out = {}
        if not self.val_pairs:
            return out
        p = np.asarray(self.val_pairs)
        i, j = p[:, 0].astype(int), p[:, 1].astype(int)
        r = self._seg_scores(self.reward)
        m = p[:, 3] != 0.5
        out["rm_val_acc"] = float(((r[i] > r[j]) == (p[:, 3] > 0.5))[m].mean())
        if self.learn_cost:
            c = self._seg_scores(self.cost)
            m = p[:, 5] != 0.5
            out["cm_val_acc"] = float(((c[i] > c[j]) == (p[:, 5] > 0.5))[m].mean()) if m.any() else float("nan")
            segs = np.unique(np.concatenate([i, j]))
            out["cm_safety_acc"] = float(((c[segs] > 0) == (self.unsafe_true[segs] > 0.5)).mean())
        return out
