"""
Statistics for the causal inference skill — computed in code so every number is exact and
reproducible; the model only chooses among them and judges the design.

Pure numpy (no scipy/statsmodels). Treatment is a 0/1 array throughout.
"""
from __future__ import annotations

import math

import numpy as np

Z_95 = 1.959963984540054   # two-sided 95%
Z_80_POWER = 0.8416212335729143


def two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def srm(n_treated: int, n_control: int, expected_treated_share: float = 0.5) -> dict:
    """Sample-ratio mismatch: chi-square test (1 df) of the observed split vs the design."""
    n = n_treated + n_control
    exp_t, exp_c = n * expected_treated_share, n * (1 - expected_treated_share)
    chi2 = (n_treated - exp_t) ** 2 / exp_t + (n_control - exp_c) ** 2 / exp_c
    return {"n_treated": int(n_treated), "n_control": int(n_control),
            "observed_treated_share": round(n_treated / n, 4),
            "expected_treated_share": expected_treated_share,
            "chi2": round(chi2, 3), "p_value": float(f"{math.erfc(math.sqrt(chi2 / 2)):.3g}")}


def _estimate(effect: float, se: float, method: str, **extra) -> dict:
    z = effect / se if se > 0 else float("inf")
    return {"method": method, "effect": effect, "se": se,
            "ci_low": effect - Z_95 * se, "ci_high": effect + Z_95 * se,
            "p_value": two_sided_p(z), **extra}


def diff_in_means(y: np.ndarray, t: np.ndarray) -> dict:
    """Difference in means with a Welch (unequal-variance) standard error."""
    yt, yc = y[t == 1], y[t == 0]
    se = math.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return _estimate(float(yt.mean() - yc.mean()), se, "difference_in_means",
                     mean_treated=float(yt.mean()), mean_control=float(yc.mean()))


def ols_effect(y: np.ndarray, t: np.ndarray, X: np.ndarray | None, method: str = "regression_adjustment") -> dict:
    """Coefficient on t from OLS of y on [1, t, X] with HC1 robust standard errors."""
    cols = [np.ones_like(y), t.astype(float)]
    if X is not None and X.size:
        Xc = X - X.mean(axis=0)
        cols.extend(Xc.T)
    D = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    n, k = D.shape
    bread = np.linalg.pinv(D.T @ D)
    meat = (D * resid[:, None] ** 2).T @ D
    cov = bread @ meat @ bread * n / max(n - k, 1)
    return _estimate(float(beta[1]), float(math.sqrt(max(cov[1, 1], 0.0))), method)


def cuped(y: np.ndarray, t: np.ndarray, pre: np.ndarray) -> dict:
    """CUPED: subtract theta * (pre - mean(pre)) to cut variance, then difference in means."""
    theta = float(np.cov(y, pre, ddof=1)[0, 1] / pre.var(ddof=1)) if pre.var() > 0 else 0.0
    est = diff_in_means(y - theta * (pre - pre.mean()), t)
    est.update(method="cuped", theta=theta)
    return est


def propensity_scores(t: np.ndarray, X: np.ndarray, iters: int = 50) -> np.ndarray:
    """Logistic regression of t on X (IRLS, lightly ridge-regularised for stability)."""
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    D = np.column_stack([np.ones(len(t)), Xs])
    beta = np.zeros(D.shape[1])
    ridge = 1e-4 * np.eye(D.shape[1]); ridge[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(D @ beta, -30, 30)))
        W = p * (1 - p)
        H = (D * W[:, None]).T @ D + ridge
        step = np.linalg.solve(H, D.T @ (t - p) - ridge @ beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return 1 / (1 + np.exp(-np.clip(D @ beta, -30, 30)))


def ipw(y: np.ndarray, t: np.ndarray, ps: np.ndarray, n_boot: int = 200, seed: int = 0) -> dict:
    """Normalised (Hajek) inverse-propensity-weighted ATE, propensities clipped to
    [0.01, 0.99]; bootstrap standard error over rows with fixed propensities."""
    def ate(idx):
        yy, tt, pp = y[idx], t[idx], np.clip(ps[idx], 0.01, 0.99)
        w1, w0 = tt / pp, (1 - tt) / (1 - pp)
        return float((w1 * yy).sum() / w1.sum() - (w0 * yy).sum() / w0.sum())
    n = len(y)
    effect = ate(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [ate(rng.integers(0, n, n)) for _ in range(n_boot)]
    return _estimate(effect, float(np.std(boots, ddof=1)), "inverse_propensity_weighting")


def overlap(ps: np.ndarray, t: np.ndarray) -> dict:
    """How much of each group sits where the other group is nearly absent."""
    extreme = (ps < 0.05) | (ps > 0.95)
    return {"share_outside_0.05_0.95": round(float(extreme.mean()), 4),
            "treated_ps_range": [round(float(ps[t == 1].min()), 3), round(float(ps[t == 1].max()), 3)],
            "control_ps_range": [round(float(ps[t == 0].min()), 3), round(float(ps[t == 0].max()), 3)]}


def smd(x: np.ndarray, t: np.ndarray) -> float:
    """Standardised mean difference of a covariate between groups (|d| > 0.1 is imbalance)."""
    xt, xc = x[t == 1], x[t == 0]
    pooled = math.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
    return float((xt.mean() - xc.mean()) / pooled) if pooled > 0 else 0.0


def mde(se: float) -> float:
    """Minimum detectable effect at alpha 0.05 (two-sided) and 80% power."""
    return (Z_95 + Z_80_POWER) * se


# ── Sensitivity to unmeasured confounding ─────────────────────────────────────

def e_value(effect: float, ci_low: float, ci_high: float, outcome_sd: float) -> dict:
    """E-value (VanderWeele & Ding) via the standardised-difference approximation RR ≈ exp(0.91·d):
    how strongly an unmeasured confounder would have to be associated with both treatment and
    outcome (as a risk ratio) to fully explain the estimate away. For the CI, the bound nearer
    the null; 1.0 when the interval already includes zero."""
    def ev(d: float) -> float:
        rr = math.exp(0.91 * abs(d))
        return rr + math.sqrt(rr * (rr - 1))
    if outcome_sd <= 0:
        return {"e_value": None, "e_value_ci": None}
    near_null = 0.0 if ci_low <= 0 <= ci_high else min(abs(ci_low), abs(ci_high))
    return {"e_value": round(ev(effect / outcome_sd), 3),
            "e_value_ci": round(ev(near_null / outcome_sd), 3) if near_null else 1.0}


# ── Difference-in-differences ─────────────────────────────────────────────────

def cluster_ols(y: np.ndarray, D: np.ndarray, clusters: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and covariance; cluster-robust (CR1) when clusters are given, else HC1."""
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    n, k = D.shape
    bread = np.linalg.pinv(D.T @ D)
    if clusters is None:
        meat = (D * resid[:, None] ** 2).T @ D
        scale = n / max(n - k, 1)
    else:
        _, idx = np.unique(clusters, return_inverse=True)
        g = idx.max() + 1
        scores = np.zeros((g, k))
        np.add.at(scores, idx, D * resid[:, None])
        meat = scores.T @ scores
        scale = g / max(g - 1, 1) * (n - 1) / max(n - k, 1)
    return beta, bread @ meat @ bread * scale


def did(y: np.ndarray, treated: np.ndarray, post: np.ndarray, clusters: np.ndarray | None) -> dict:
    """2x2 difference-in-differences: coefficient on treated×post in y ~ 1 + treated + post + treated×post."""
    D = np.column_stack([np.ones_like(y), treated, post, treated * post]).astype(float)
    beta, cov = cluster_ols(y, D, clusters)
    return _estimate(float(beta[3]), float(math.sqrt(max(cov[3, 3], 0.0))), "difference_in_differences")


def event_study(y: np.ndarray, treated: np.ndarray, period: np.ndarray, ref_period,
                clusters: np.ndarray | None) -> list[dict]:
    """Treated-vs-control gap in each period relative to ref_period (the last pre-period), with
    period fixed effects. Pre-period gaps near zero support parallel trends."""
    periods = [p for p in sorted(np.unique(period)) if p != ref_period]
    period_fe = [(period == p).astype(float) for p in periods]
    inter = [((period == p) & (treated == 1)).astype(float) for p in periods]
    D = np.column_stack([np.ones_like(y), treated.astype(float), *period_fe, *inter])
    beta, cov = cluster_ols(y, D, clusters)
    first = 2 + len(periods)
    out = []
    for j, p in enumerate(periods):
        b, se = float(beta[first + j]), float(math.sqrt(max(cov[first + j, first + j], 0.0)))
        out.append({"period": p, "gap_vs_ref": b, "se": se, "p_value": two_sided_p(b / se) if se > 0 else 1.0})
    return out


def pre_trend(y: np.ndarray, treated: np.ndarray, time_index: np.ndarray, pre: np.ndarray,
              clusters: np.ndarray | None) -> dict:
    """Differential linear trend before treatment: slope of treated − control per period,
    from y ~ 1 + treated + t + treated×t on pre-period rows."""
    m = pre.astype(bool)
    t = time_index[m].astype(float)
    D = np.column_stack([np.ones(m.sum()), treated[m], t, treated[m] * t]).astype(float)
    beta, cov = cluster_ols(y[m], D, clusters[m] if clusters is not None else None)
    se = float(math.sqrt(max(cov[3, 3], 0.0)))
    return {"differential_slope_per_period": float(beta[3]), "se": se,
            "p_value": two_sided_p(beta[3] / se) if se > 0 else 1.0}
