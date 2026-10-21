"""
Deterministic analysis functions with no model calls: no API key, network or config needed.

    from ds_workbench.core import leakage_profile, join_stats
    profile = leakage_profile(df, target_col="churned")   # single-feature AUC, null/zero patterns...
    stats = join_stats(orders, customers, ["customer_id"])  # exact rows, fan-out, data loss, verdict

The skills build on these: they compute the evidence here, then ask a model to interpret it.
This package must never import anthropic or config (a test enforces it).

Names resolve on first use (PEP 562) rather than being imported here: a package __init__ that
imports its own submodules makes concurrent first imports of the package and a submodule
deadlock on Python's per-module import locks (seen when Dash served callbacks on several
threads).
"""
import importlib

_EXPORTS = {
    "join_stats": "ds_workbench.core.joins",
    "join_verdict": "ds_workbench.core.joins",
    "leakage_profile": "ds_workbench.core.leakage",
    "assess_readiness": "ds_workbench.core.readiness",
}
_SUBMODULES = {"joins", "leakage", "causal_stats", "readiness", "problem_spec"}

__all__ = [*_EXPORTS, *_SUBMODULES]


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"ds_workbench.core.{name}")
    else:
        raise AttributeError(f"module 'ds_workbench.core' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
