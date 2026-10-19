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
