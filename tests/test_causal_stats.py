"""Causal statistics against known answers."""
import math

import numpy as np
import pytest

from skills import causal_stats as cs

RNG = np.random.default_rng(7)


def near(est, truth):
    """Within 4 standard errors — a 95% interval misses 1 time in 20, so tests can't use it."""
    return abs(est["effect"] - truth) < 4 * est["se"]


def test_srm_detects_a_skewed_split_and_passes_a_fair_one():
    assert cs.srm(10_000, 10_000)["p_value"] == 1.0
    bad = cs.srm(10_500, 9_500)       # chi2 = 50
    assert bad["chi2"] == pytest.approx(50.0) and bad["p_value"] < 1e-10
    assert cs.srm(5_200, 4_800)["p_value"] == pytest.approx(math.erfc(math.sqrt(16 / 2)), rel=1e-3)


def test_diff_in_means_recovers_a_known_effect():
    n = 40_000
    t = (RNG.random(n) < 0.5).astype(int)
    y = 10 + 2.0 * t + RNG.normal(0, 5, n)
    est = cs.diff_in_means(y, t)
    assert near(est, 2.0)
    assert est["se"] == pytest.approx(5 * math.sqrt(4 / n), rel=0.05)


def test_regression_adjustment_removes_confounding():
    n = 20_000
    x = RNG.normal(0, 1, n)
    t = (RNG.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)   # x drives treatment...
    y = 1.0 * t + 3.0 * x + RNG.normal(0, 1, n)                     # ...and the outcome
    naive = cs.diff_in_means(y, t)
    adjusted = cs.ols_effect(y, t, x[:, None])
    assert naive["effect"] > 2.5                                      # badly biased
    assert near(adjusted, 1.0)


def test_cuped_keeps_the_effect_and_shrinks_the_interval():
    n = 20_000
    pre = RNG.normal(50, 10, n)
    t = (RNG.random(n) < 0.5).astype(int)
    y = 0.8 * pre + 1.5 * t + RNG.normal(0, 4, n)
    plain, adj = cs.diff_in_means(y, t), cs.cuped(y, t, pre)
    assert near(adj, 1.5)
    assert adj["se"] < plain["se"] / 2


def test_ipw_recovers_the_effect_under_confounding():
    n = 20_000
    x = RNG.normal(0, 1, n)
    t = (RNG.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    y = 2.0 * t + 2.0 * x + RNG.normal(0, 1, n)
    ps = cs.propensity_scores(t, x[:, None])
    est = cs.ipw(y, t, ps, n_boot=100)
    assert near(est, 2.0)
    assert cs.overlap(ps, t)["share_outside_0.05_0.95"] < 0.1


def test_smd_and_mde():
    t = np.array([1] * 500 + [0] * 500)
    x = np.concatenate([RNG.normal(0.5, 1, 500), RNG.normal(0, 1, 500)])
    assert cs.smd(x, t) == pytest.approx(0.5, abs=0.15)
    assert cs.mde(1.0) == pytest.approx(2.8016, abs=1e-3)
