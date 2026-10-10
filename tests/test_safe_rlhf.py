from types import SimpleNamespace

import numpy as np
import torch

from srl.safe_rlhf import SafePreferenceLearner


def _cfg(**kw):
    base = dict(rm_members=3, rm_hidden=64, rm_lr=1e-3, rm_weight_decay=0.0, label_noise=0.0, label_budget=600,
                seg_len=10, candidate_mult=5, rm_batch=64, val_frac=0.2, safety_label_noise=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _train(learner, rng, w_r, w_c, rounds=3):
    for _ in range(rounds):
        segs = rng.normal(size=(200, 10, 4)).astype(np.float32)
        rets = (segs @ w_r).sum(-1)
        hazard = np.maximum(segs @ w_c - 1.0, 0)            # sparse, non-negative "true cost"
        learner.add_queries(segs, rets, hazard.sum(-1), 200)
        learner.train(200)


def test_learned_reward_and_cost_recover_their_targets_and_cost_sign():
    rng = np.random.default_rng(0); torch.manual_seed(0)
    L = SafePreferenceLearner(4, _cfg(), rng, learn_cost=True)
    _train(L, rng, rng.normal(size=4), rng.normal(size=4))
    m = L.metrics()
    assert m["rm_val_acc"] > 0.8
    assert m["cm_val_acc"] > 0.75
    assert m["cm_safety_acc"] > 0.75     # sign of the segment cost separates safe from unsafe


def test_pessimism_lowers_reward_and_raises_cost():
    rng = np.random.default_rng(1); torch.manual_seed(1)
    L = SafePreferenceLearner(4, _cfg(), rng, learn_cost=True)
    _train(L, rng, rng.normal(size=4), rng.normal(size=4), rounds=1)
    x = rng.normal(size=(500, 4)).astype(np.float32)
    r0, _ = L.predict_reward(x, 0.0); r1, s = L.predict_reward(x, 1.0)
    c0, _ = L.predict_cost(x, 0.0); c1, _ = L.predict_cost(x, 1.0)
    np.testing.assert_allclose(r0 - r1, s, rtol=1e-4, atol=1e-5)
    assert (r1 <= r0 + 1e-6).all() and (c1 >= c0 - 1e-6).all()
