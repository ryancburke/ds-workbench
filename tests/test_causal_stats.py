"""Causal statistics against known answers."""
import math

import numpy as np
import pytest

from ds_workbench.core import causal_stats as cs

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


def _panel(effect=3.0, pre_slope_gap=0.0, n_units=60, n_periods=12, start=7, seed=1):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(1, n_periods + 1), n_units)
    treated = (unit < n_units // 2).astype(int)
    unit_fe = rng.normal(0, 5, n_units)[unit]
    y = (20 + unit_fe + 0.5 * period + pre_slope_gap * treated * period
         + effect * treated * (period >= start) + rng.normal(0, 1, len(unit)))
    return y, treated, period, (period >= start).astype(int), unit


def test_did_recovers_the_effect_with_cluster_robust_se():
    y, tr, period, post, unit = _panel()
    est = cs.did(y, tr, post, unit)
    assert near(est, 3.0)


def test_event_study_pre_gaps_are_flat_under_parallel_trends():
    y, tr, period, post, unit = _panel()
    gaps = cs.event_study(y, tr, period, ref_period=6, clusters=unit)
    pre = [g for g in gaps if g["period"] < 6]
    assert all(abs(g["gap_vs_ref"]) < 4 * g["se"] + 0.5 for g in pre)
    assert np.mean([g["gap_vs_ref"] for g in gaps if g["period"] >= 7]) == pytest.approx(3.0, abs=0.6)


def test_pre_trend_test_detects_diverging_groups():
    y, tr, period, post, unit = _panel(effect=0.0, pre_slope_gap=0.6)
    flat = cs.pre_trend(*_panel()[:2], _panel()[2], 1 - _panel()[3], _panel()[4])
    diverging = cs.pre_trend(y, tr, period, 1 - post, unit)
    assert diverging["p_value"] < 0.001 and abs(diverging["differential_slope_per_period"] - 0.6) < 0.2
    assert flat["p_value"] > 0.001
    assert cs.did(y, tr, post, unit)["effect"] > 2      # naive DiD mistakes the trend for an effect


def test_e_value():
    ev = cs.e_value(effect=5.0, ci_low=4.5, ci_high=5.5, outcome_sd=50.0)   # d = 0.1
    rr = math.exp(0.091)
    assert ev["e_value"] == pytest.approx(rr + math.sqrt(rr * (rr - 1)), abs=1e-3)
    assert cs.e_value(1.0, -0.5, 2.5, 10.0)["e_value_ci"] == 1.0           # CI includes zero
