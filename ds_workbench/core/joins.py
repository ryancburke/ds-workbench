"""Exact join statistics from per-key counts, without materialising the join.

SQL/Spark semantics: null keys never match (pandas' merge would match NaN to NaN).
Memory is O(distinct keys), so a many-to-many join that would produce billions of rows is
measured, not built.
"""
from __future__ import annotations

import pandas as pd

# Verdict thresholds
FANOUT_REVIEW, FANOUT_DANGER = 1.1, 2.0     # rows produced per matched left row
DROP_REVIEW, DROP_DANGER = 0.05, 0.20       # share of left rows an inner/right join drops

_JOIN_TYPES = {"inner", "left", "right", "outer"}


def _key_counts(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    return df.dropna(subset=keys).groupby(keys, sort=False).size()


def join_stats(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], how: str = "inner",
               left_name: str = "left", right_name: str = "right") -> dict:
    """Row counts, fan-out, data loss and key diagnostics for `left JOIN right ON keys`."""
    if how not in _JOIN_TYPES:
        raise ValueError(f"how must be one of {sorted(_JOIN_TYPES)}, got {how!r}")
    missing = [k for k in keys if k not in left.columns or k not in right.columns]
    if missing:
        raise ValueError(f"Join keys missing from a table: {missing}")

    lc, rc = _key_counts(left, keys), _key_counts(right, keys)
    common = lc.index.intersection(rc.index)
    l_common, r_common = lc.loc[common], rc.loc[common]

    inner_rows = int((l_common * r_common).sum())
    matched_left, matched_right = int(l_common.sum()), int(r_common.sum())
    n_left, n_right = len(left), len(right)
    unmatched_left, unmatched_right = n_left - matched_left, n_right - matched_right
    result_rows = inner_rows + (unmatched_left if how in ("left", "outer") else 0) \
                             + (unmatched_right if how in ("right", "outer") else 0)
    left_dropped = unmatched_left if how in ("inner", "right") else 0

    stats = {
        "left_name": left_name, "right_name": right_name, "join_type": how, "join_keys": list(keys),
        "left_rows": n_left, "right_rows": n_right, "result_rows": result_rows,
        "fan_out_factor": round(inner_rows / matched_left, 4) if matched_left else 1.0,
        "row_ratio": round(result_rows / n_left, 4) if n_left else 0.0,
        "left_rows_dropped": left_dropped,
        "left_rows_dropped_pct": round(left_dropped / n_left * 100, 2) if n_left else 0.0,
        "left_key_null_rows": int(left[keys].isnull().any(axis=1).sum()),
        "right_key_null_rows": int(right[keys].isnull().any(axis=1).sum()),
        "left_key_duplicates": int((lc - 1).clip(lower=0).sum()),
        "right_key_duplicates": int((rc - 1).clip(lower=0).sum()),
        "unmatched_keys_in_left": int(len(lc) - len(common)),
        "unmatched_keys_in_right": int(len(rc) - len(common)),
        "many_to_many_keys": int(((l_common > 1) & (r_common > 1)).sum()),
        "max_rows_from_one_key": int((l_common * r_common).max()) if len(common) else 0,
    }
    stats["verdict"], stats["verdict_reasons"] = join_verdict(stats)
    return stats


def join_verdict(stats: dict) -> tuple[str, list[str]]:
    """safe / review_needed / dangerous from fan-out, data loss and many-to-many keys."""
    reasons: list[tuple[int, str]] = []
    fo, drop = stats["fan_out_factor"], stats["left_rows_dropped_pct"] / 100
    if fo > FANOUT_DANGER:
        reasons.append((2, f"fan-out ×{fo:.2f} > {FANOUT_DANGER}"))
    elif fo > FANOUT_REVIEW:
        reasons.append((1, f"fan-out ×{fo:.2f} > {FANOUT_REVIEW}"))
    if drop >= DROP_DANGER:
        reasons.append((2, f"drops {drop:.0%} of {stats['left_name']} rows"))
    elif drop >= DROP_REVIEW:
        reasons.append((1, f"drops {drop:.0%} of {stats['left_name']} rows"))
    if stats["many_to_many_keys"]:
        reasons.append((1, f"{stats['many_to_many_keys']} many-to-many keys"))
    level = max((lvl for lvl, _ in reasons), default=0)
    return ["safe", "review_needed", "dangerous"][level], [text for _, text in reasons]
