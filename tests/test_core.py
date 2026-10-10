from types import SimpleNamespace

import numpy as np
import torch

from srl.ppo_lag import Lagrange, gae
from srl.reward_model import (PreferenceLearner, PreferenceOracle, bradley_terry_loss,
                              label_schedule, make_segments)


def test_gae_lambda1_equals_discounted_return_minus_value():
    r = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    adv, ret = gae(r, v, 0.0, np.array([0, 0, 1.0]), gamma=0.9, lam=1.0)
    expected = np.array([1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3.0])
    np.testing.assert_allclose(ret, expected, rtol=1e-5)
    np.testing.assert_allclose(adv, expected - 0.5, rtol=1e-5)


def test_gae_does_not_leak_across_episodes():
    r = np.array([0.0, 10.0], dtype=np.float32)
    adv, _ = gae(r, np.zeros(2, dtype=np.float32), 0.0, np.array([1.0, 1.0]), 0.99, 0.95)
    assert adv[0] == 0.0


def test_lagrange_rises_on_violation_and_stays_nonnegative():
    lag = Lagrange(cost_limit=25, lr=0.01)
    assert lag.update(50) > 0
    for _ in range(100):
        lag.update(0)
    assert lag.value == 0.0
    pid = Lagrange(cost_limit=25, mode="pid")
    assert pid.update(50) > 0


def test_bradley_terry_prefers_correct_ordering():
    y = torch.ones(1)
    good = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([0.0]), y)
    bad = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([2.0]), y)
    assert good < bad


def test_oracle_noise_free_and_ties():
    o = PreferenceOracle(0.0, np.random.default_rng(0))
    noisy, true = o(np.array([1.0, 0.0, 3.0]), np.array([0.0, 1.0, 3.0]))
    np.testing.assert_array_equal(true, [1.0, 0.0, 0.5])
    np.testing.assert_array_equal(noisy, true)


def test_segments_respect_episode_boundaries():
    feats = np.arange(10, dtype=np.float32)[:, None]
    dones = np.zeros(10); dones[4] = 1; dones[9] = 1  # two episodes of length 5
    segs, rets = make_segments(feats, np.ones(10), dones, seg_len=2)
    assert segs.shape == (4, 2, 1)
    for s in segs[:, :, 0]:
        assert (s[0] < 5) == (s[1] < 5)
    np.testing.assert_array_equal(rets, 2.0)


def test_label_schedule_spends_budget():
    assert sum(label_schedule(e, 50, 2000) for e in range(50)) == 2000
    assert sum(label_schedule(e, 3, 200) for e in range(3)) == 200


def test_reward_model_recovers_linear_reward():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    cfg = SimpleNamespace(rm_members=3, rm_hidden=128, rm_lr=1e-3, rm_weight_decay=0.0, label_noise=0.0, label_budget=600,
                          seg_len=10, active_queries=True, candidate_mult=5, rm_batch=64, val_frac=0.2)
    w = rng.normal(size=4)
    learner = PreferenceLearner(4, cfg, rng)
    for _ in range(3):
        segs = rng.normal(size=(200, 10, 4)).astype(np.float32)
        learner.add_queries(segs, (segs @ w).sum(-1), 200)
        learner.train(200)
    assert learner.accuracy(learner.val_pairs) > 0.85
