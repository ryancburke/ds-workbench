"""
Data verification: check the tables you selected from the Data Scout's shortlist against the data.

    report = data_scout.scout(session=session)
    session.select_tables(["demo.marketing.campaign_exposure", "demo.crm.customers",
                           "demo.sales.customer_orders"], note="exposure, unit and outcome")
    check = data_verification.verify(session=session)          # free: aggregates only
    check = data_verification.verify(session=session, dq=True) # + the DQ skill per table (paid)

The checks are deterministic (ds_workbench.core.verify) and run as aggregates where the data
lives: row counts, null rates, date ranges, key match rates and cardinality — no table is
copied out. They mark joins verified (or not), check that the outcome is observed after the
exposure and that covariates exist from before it, and flag stale or deprecated tables. With
dq=True, the data quality skill also runs on a capped sample of each table (one model call per
table — this is the step to approve when tables are large or many).

The result is a readiness check ("data_verification"): critical findings block, warnings caution.
"""
from __future__ import annotations

from datetime import date

from ds_workbench.core.catalog import SAMPLE_CAP
from ds_workbench.core.verify import verify_tables
from skills.api_retry import skill_entry
from skills.result import SkillResult

DQ_MAX_TABLES = 6


@skill_entry("data_verification")
def verify(session=None, tables: list[str] | None = None, spec: dict | None = None, scout: dict | None = None,
           catalog=None, dq: bool = False, sample_rows: int = SAMPLE_CAP, today: date | None = None) -> dict:
    """Verify selected tables against the data.

    Args:
        session: WorkflowSession with a spec, a scout report and a selection (select_tables).
        tables: Tables to verify (default: the session's selection).
        spec, scout: ProblemSpec and Data Scout report (default: from the session).
        catalog: Catalog to query (default: from CATALOG_BACKEND).
        dq: Also run the data quality skill on a capped sample of each table (model calls).
        sample_rows: Rows per DQ sample (at most SAMPLE_CAP).

    Returns:
        Dict: verdict (verified | verified_with_issues | failed), tables, joins (with status set
        here), temporal checks, findings, and dq per table when dq=True.
    """
    try:
        return _verify(session, tables, spec, scout, catalog, dq, sample_rows, today)
    except Exception as exc:
        return SkillResult({"error": str(exc)}, skill="data_verification")


def _verify(session, tables, spec, scout, catalog, dq, sample_rows, today):
    from skills.data_scout import get_catalog
    if session is not None:
        spec = spec or session.spec
        scout = scout or session.scout
        tables = tables or (session.selection or {}).get("tables")
    if not tables:
        raise ValueError("Select the tables to verify first: session.select_tables([...], note=...) "
                         "or verify(tables=[...])")
    catalog = catalog or get_catalog()
    tables = list(dict.fromkeys(tables))
    missing = []
    for t in tables:
        try:
            catalog.describe(t)
        except KeyError:
            missing.append(t)
    if missing:
        raise ValueError(f"Not in the {catalog.name} catalog: {missing}")

    out = verify_tables(catalog, tables, scout=scout, spec=spec, today=today)
    shortlisted = {c["table"] for c in (scout or {}).get("candidates", []) if c["recommendation"] != "reject"}
    out["not_from_shortlist"] = [t for t in tables if scout and t not in shortlisted]
    out["catalog"] = catalog.name

    if dq:
        from skills.data_quality import assess
        out["dq"] = {}
        for t in tables[:DQ_MAX_TABLES]:
            r = assess(catalog.sample(t, n=sample_rows), table_name=t)
            if r.get("error"):
                out["findings"].insert(0, {"severity": "critical", "check": "data_quality", "table": t, "column": "—",
                                           "description": f"data quality check failed: {r['error'][:150]}",
                                           "recommendation": "re-run it"})
                out["dq"][t] = {"error": r["error"]}
                continue
            crit = [f for f in r.get("findings", []) if f.get("severity") == "critical"]
            out["dq"][t] = {"score": r.get("overall_score"), "critical": len(crit),
                            "warnings": sum(f.get("severity") == "warning" for f in r.get("findings", [])),
                            "top": [f"{f.get('column')}: {f.get('description', '')}"[:140] for f in crit[:3]]}
            out["findings"] += [{"severity": "critical", "check": "data_quality", "table": t,
                                 "column": f.get("column", "—"), "description": f.get("description", ""),
                                 "recommendation": f.get("recommendation", "")} for f in crit]
        if len(tables) > DQ_MAX_TABLES:
            out["dq_skipped"] = tables[DQ_MAX_TABLES:]
        if any(f["severity"] == "critical" for f in out["findings"]):
            out["verdict"] = "failed"

    result = SkillResult(out, skill="data_verification")
    if session is not None:
        session.register("data_verification", result)
    return result
