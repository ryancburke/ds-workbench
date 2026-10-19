"""Statistical signals of target leakage, computed per feature against a target column.

Pure pandas/numpy; the leakage_detector skill sends this profile to a model to interpret.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _single_feature_auc(values: pd.Series, y: pd.Series) -> float | None:
    """Rank-based AUC of one feature against a 0/1 target, folded to >= 0.5.
    Unlike Pearson r it sees any monotone separation, linear or not."""
    valid = values.notna() & y.notna()
    v, t = values[valid].astype(float), y[valid].astype(float)
    n1 = float(t.sum()); n0 = float(len(t) - n1)
    if n1 < 5 or n0 < 5:
        return None
    ranks = v.rank()
    auc = (ranks[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return round(float(max(auc, 1 - auc)), 4)


def leakage_profile(df: pd.DataFrame, target_col: str) -> dict:
    """Compute statistical signals that reveal target leakage."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    target = df[target_col]
    is_binary = target.dtype == bool or (
        target.nunique() == 2 and pd.api.types.is_numeric_dtype(target)
    )

    if target.dtype == bool:
        target_numeric = target.astype(float)
    elif is_binary:
        uvals = sorted(target.dropna().unique())
        target_numeric = target.map({uvals[0]: 0.0, uvals[1]: 1.0})
    elif pd.api.types.is_numeric_dtype(target) and target.dtype != bool:
        target_numeric = target.astype(float)
    else:
        target_numeric = None

    profile: dict = {
        "target_col": target_col,
        "target_dtype": str(target.dtype),
        "target_value_counts": {str(k): int(v) for k, v in target.value_counts().head(5).items()},
        "n_rows": len(df),
        "features": {},
    }

    for col in df.columns:
        if col == target_col:
            continue
        feat = df[col]
        info: dict = {
            "dtype": str(feat.dtype),
            "null_pct": round(float(feat.isnull().mean() * 100), 2),
            "unique_count": int(feat.nunique()),
        }

        # Point-biserial / Pearson correlation with target
        if (
            target_numeric is not None
            and pd.api.types.is_numeric_dtype(feat)
            and feat.dtype != bool
        ):
            valid = feat.notna() & target_numeric.notna()
            if valid.sum() > 10:
                try:
                    corr = float(
                        pd.Series(feat[valid].values, dtype=float).corr(
                            pd.Series(target_numeric[valid].values, dtype=float)
                        )
                    )
                    if not np.isnan(corr):
                        info["correlation_with_target"] = round(abs(corr), 4)
                except Exception:
                    pass

        # Single-feature AUC — catches non-linear separation that Pearson r misses
        if is_binary and target_numeric is not None and pd.api.types.is_numeric_dtype(feat) \
                and feat.dtype != bool:
            auc = _single_feature_auc(feat, target_numeric)
            if auc is not None:
                info["single_feature_auc"] = auc
            # Values that are zero for one class only (e.g. a refund credit after the event)
            zeros = feat.fillna(0) == 0
            if 0.1 <= zeros.mean() <= 0.95:
                info["zero_pct_by_class"] = {
                    str(val): round(float(zeros[target == val].mean() * 100), 2)
                    for val in target.dropna().unique() if (target == val).sum() >= 5}

        # Categorical: target rate per value, and values seen in only one class
        if is_binary and target_numeric is not None and not pd.api.types.is_numeric_dtype(feat) \
                and 1 < feat.nunique() <= 50:
            rates = target_numeric.groupby(feat.astype(str)).agg(["mean", "size"])
            rates = rates[rates["size"] >= max(5, int(0.01 * len(df)))]
            if len(rates):
                info["target_rate_by_value"] = {
                    str(k): round(float(r["mean"]), 3)
                    for k, r in rates.sort_values("size", ascending=False).head(8).iterrows()}
                pure = rates[(rates["mean"] <= 0.0) | (rates["mean"] >= 1.0)]
                if len(pure):
                    info["single_class_values"] = {
                        str(k): {"rows": int(r["size"]), "target_rate": round(float(r["mean"]), 3)}
                        for k, r in pure.iterrows()}
                encoded = feat.astype(str).map(rates["mean"]).astype(float)
                auc = _single_feature_auc(encoded, target_numeric)
                if auc is not None:
                    info["single_feature_auc"] = auc

        # Conditional null rate — exposes post-event proxy features
        if feat.isnull().any() and is_binary:
            cond: dict = {}
            for val in target.dropna().unique():
                mask = target == val
                if mask.sum() >= 5:
                    cond[str(val)] = round(float(feat[mask].isnull().mean() * 100), 2)
            if cond:
                info["conditional_null_pct"] = cond

        if pd.api.types.is_numeric_dtype(feat) and feat.dtype != bool:
            non_null = feat.dropna().astype(float)
            if len(non_null):
                info["stats"] = {
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "mean": round(float(non_null.mean()), 4),
                }
        else:
            vc = feat.value_counts(dropna=False).head(5)
            info["sample_values"] = {str(k): int(v) for k, v in vc.items()}

        profile["features"][col] = info

    return profile
