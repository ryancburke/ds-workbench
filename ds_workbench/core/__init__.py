"""
Deterministic analysis functions with no model calls: no API key, network or config needed.

    from ds_workbench.core import leakage_profile, join_stats
    profile = leakage_profile(df, target_col="churned")   # single-feature AUC, null/zero patterns...
    stats = join_stats(orders, customers, ["customer_id"])  # exact rows, fan-out, data loss, verdict

The skills build on these: they compute the evidence here, then ask a model to interpret it.
This package must never import anthropic or config (a test enforces it).
"""
from ds_workbench.core.joins import join_stats, join_verdict
from ds_workbench.core.leakage import leakage_profile

__all__ = ["join_stats", "join_verdict", "leakage_profile"]
