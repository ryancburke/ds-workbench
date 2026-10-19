"""
Autonomous DS agent simulation.

Three personas with distinct voices receive a business scenario,
independently decide which skills to run, interpret the results,
and deliver a final recommendation.
"""

import logging
import time
from datetime import datetime, timezone

from skills.api_retry import (call_context, check_response, get_client as _client,
                              response_text, strict_tool, tool_input)
from skills.models import role

log = logging.getLogger(__name__)


# ── Personas ──────────────────────────────────────────────────────────────────

PERSONAS = [
    {
        "name": "Alex Chen",
        "role": "Lead Data Scientist",
        "icon": "bi-person-workspace",
        "color": "#6478ff",
        "approach": (
            "methodical and direct. You always ground analysis in data quality before drawing "
            "conclusions — 'garbage in, garbage out' is your north star. You speak technically "
            "but always close the loop on business impact. You're willing to be the voice that "
            "says 'we can't trust this number yet.' You tend to start with what you *don't* know "
            "before stating what you do."
        ),
        "focus": "data reliability, statistical validity, preventing analytical mistakes upstream",
    },
    {
        "name": "Priya Sharma",
        "role": "ML Engineer",
        "icon": "bi-cpu-fill",
        "color": "#ff7eb3",
        "approach": (
            "systematic and production-minded. You bridge the gap between DS notebooks and "
            "real systems. You ask hard questions about reproducibility, hardcoded values, and "
            "what happens when this code touches new data. You're efficient and direct — you "
            "call out problems clearly but always suggest the fix. You trust Alex on data quality "
            "but you own the 'will this hold up in prod?' question."
        ),
        "focus": "code quality, reproducibility, pipeline integrity, production handoff readiness",
    },
    {
        "name": "Jordan Kim",
        "role": "Data Analyst",
        "icon": "bi-bar-chart-line-fill",
        "color": "#64c8ff",
        "approach": (
            "business-first and pragmatic. You translate what Alex and Priya find into language "
            "the CFO or product team can act on. You think in terms of revenue impact, percentage "
            "changes, and time-to-decision. You're the one who asks 'so what does this mean for "
            "the question they actually asked?' You know when to stop digging and when to escalate. "
            "You keep the team anchored to the original ask."
        ),
        "focus": "KPI impact, executive communication, actionable recommendations, scope management",
    },
]

# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "q3_revenue",
        "title": "Q3 Revenue Shortfall",
        "icon": "bi-graph-down-arrow",
        "color": "#dc3545",
        "brief": (
            "The CFO flagged that Q3 revenue came in 12% below forecast. Leadership wants to "
            "understand whether this is a data quality issue, a genuine business problem, or both "
            "— before the board meeting next week. The data team has access to the orders table "
            "and the sales records."
        ),
        "context": "Focus on order integrity, revenue calculations, and whether the numbers can be trusted before any business conclusions are drawn.",
        "focus_skills": ["data_quality", "join_auditor", "nl_to_spark", "mlops_review"],
        "available_datasets": ["orders_dirty", "sales_clean"],
        "join_scenario": "orders_customers_fanout",
        "mlops_code": """\
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

# Load Q3 orders
df = spark.read.parquet("/mnt/data/orders/2024-Q3/")
df = df.filter(df.status == "COMPLETED")
df = df.withColumn("revenue", df.amount * 1.082)  # tax hardcoded

daily = df.groupBy("order_date").agg({"revenue": "sum"})
result = daily.orderBy("order_date").toPandas()

result.to_csv("/tmp/q3_revenue.csv")
print(f"Q3 total: {result['sum(revenue)'].sum()}")
""",
    },
    {
        "id": "churn_risk",
        "title": "Customer Churn Risk",
        "icon": "bi-person-dash",
        "color": "#ffc107",
        "brief": (
            "Product flagged a 15% drop in 30-day retention over the last quarter. The DS team "
            "needs to characterize which customer segments are at highest risk and evaluate "
            "whether the available data is sufficient to build a churn prediction model."
        ),
        "context": "Focus on behavioral patterns in session data, data completeness for modeling, and whether the signal is real or artifactual.",
        "focus_skills": ["eda", "model_selector", "leakage_detector", "data_quality"],
        "available_datasets": ["ecommerce", "hr_attrition"],
        "join_scenario": "sales_products_clean",
        "mlops_code": """\
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/mnt/data/sessions/")
df = df.filter(df.last_30_days > 0)

features = ["sessions_count", "avg_session_duration", "pages_per_session"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

lr = LogisticRegression(maxIter=100)
model = lr.fit(df)
model.save("/tmp/churn_model_v1")
""",
    },
    {
        "id": "model_readiness",
        "title": "Churn Model Readiness",
        "icon": "bi-shield-check",
        "color": "#6610f2",
        "brief": (
            "The e-commerce team wants to build a conversion prediction model. A junior DS prepared "
            "an initial feature table and a quick-and-dirty training script. Before the team invests "
            "time training and tuning, leadership wants a pre-flight check: data quality, feature "
            "leakage, an appropriate modeling approach, and a production-ready preprocessing pipeline."
        ),
        "context": "Prioritise correctness over speed. A model trained on leaky features will look great in dev and fail in production. Catch it now.",
        "focus_skills": ["leakage_detector", "feature_engineering", "model_selector", "data_quality"],
        "available_datasets": ["ecommerce_leakage", "hr_attrition"],
        "join_scenario": "sales_products_clean",
        "mlops_code": """\
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline

# Feature columns — needs review
feature_cols = [
    "page_views", "time_on_site_s", "cart_value",
    "conversion_score",  # suspicious — very correlated with target
    "order_id",          # suspicious — only exists after conversion
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
gbt = GBTClassifier(labelCol="converted", maxIter=50)

pipeline = Pipeline(stages=[assembler, gbt])
model = pipeline.fit(df)   # fit on full data — no train/test split
model.save("/tmp/churn_model_v1")
""",
    },
    {
        "id": "launch_impact",
        "title": "Did the Launch Work?",
        "icon": "bi-diagram-2",
        "color": "#20c997",
        "brief": (
            "Leadership wants to know whether three recent changes moved revenue: a new checkout "
            "flow tested as an A/B experiment (run twice through different logging pipelines), a "
            "new onboarding flow tested as an A/B experiment, and a coupon campaign marketing ran "
            "without a control group. The team must say which effects are real, how big they are, "
            "and which numbers shouldn't be trusted."
        ),
        "context": (
            "Decisions are worth millions: a wrong 'it worked' ships a bad change, a wrong 'no "
            "effect' kills a good one. Randomisation, logging, what was measured after treatment, "
            "and how coupon recipients were chosen all matter."
        ),
        "focus_skills": ["causal_inference", "data_quality"],
        "available_datasets": ["ab_test_clean", "ab_test_srm", "ab_test_bad_control", "observational_coupon"],
        "join_scenario": "sales_products_clean",
        "mlops_code": "",
    },
    {
        "id": "fraud_detection",
        "title": "Fraud Model Pre-flight",
        "icon": "bi-shield-exclamation",
        "color": "#fd7e14",
        "brief": (
            "The risk team flagged a spike in chargebacks. The DS team needs to build a fraud "
            "detection model on transaction data, but the data has known quality issues and the "
            "feature set hasn't been reviewed for leakage. Before training, they need data quality "
            "validation, a leakage check, an appropriate model for a 32:1 class imbalance, and a "
            "clean preprocessing pipeline."
        ),
        "context": (
            "Class imbalance is extreme (3% fraud rate, 32:1 ratio) — precision vs. recall tradeoff "
            "is a critical decision. Any feature computed after the fraud label is assigned is "
            "potential leakage. ID columns (transaction_id, merchant_id) should not be in features."
        ),
        "focus_skills": ["leakage_detector", "model_selector", "feature_engineering", "data_quality"],
        "available_datasets": ["transactions_fraud"],
        "join_scenario": "sales_products_clean",
        "mlops_code": """\
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Using raw transaction features — leakage not reviewed
feature_cols = [
    "amount",
    "merchant_id",   # ID column — should be excluded or target-encoded carefully
    "user_id",       # 8.4% nulls + null pattern may correlate with fraud label
]

assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features", handleInvalid="skip"
)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
rf     = RandomForestClassifier(
    labelCol="is_fraud", featuresCol="scaled_features", numTrees=100
)

pipeline = Pipeline(stages=[assembler, scaler, rf])
model    = pipeline.fit(df)   # no train/test split; class imbalance unaddressed
model.save("/tmp/fraud_model_v1")
""",
    },
    {
        "id": "pipeline_handoff",
        "title": "New Pipeline Handoff",
        "icon": "bi-arrow-left-right",
        "color": "#198754",
        "brief": (
            "Engineering just handed over a new orders-to-customers data pipeline. Before the "
            "DS team builds any features on top of it, they need to validate data quality, "
            "check that the join logic is sound, and review the pipeline notebook for "
            "production readiness."
        ),
        "context": "Treat this as a formal acceptance review. Flag anything that would cause problems downstream in modeling or reporting.",
        "focus_skills": ["data_quality", "join_auditor", "mlops_review", "nl_to_spark"],
        "available_datasets": ["orders_dirty", "sales_clean"],
        "join_scenario": "orders_customers_fanout",
        "mlops_code": """\
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

orders = spark.read.parquet("/mnt/data/orders_new/2024-01-15/")
customers = spark.read.parquet("/mnt/data/customers/")

joined = orders.join(customers, "customer_id", "left")
result = joined.toPandas()

for i, row in result.iterrows():
    result.at[i, "segment"] = "high" if row["lifetime_value"] > 1000 else "low"

result.to_csv("/tmp/pipeline_output.csv")
""",
    },
]

# ── Skill metadata ─────────────────────────────────────────────────────────────

AVAILABLE_SKILLS = {
    "data_quality":        "Audit a dataset for null rates, duplicates, type inconsistencies, and format issues. Returns a quality score and detailed findings per column.",
    "eda":                 "Exploratory analysis — generates statistical insights, chart recommendations, and potential ML target identification.",
    "join_auditor":        "Checks join integrity between two tables: fan-out detection, unmatched keys, unexpected row-count changes.",
    "nl_to_spark":         "Converts a natural-language business question into Spark SQL and PySpark DataFrame API code.",
    "mlops_review":        "Reviews PySpark/Python code for production readiness: hardcoded configs, missing logging, inefficient patterns.",
    "model_selector":      "Given a target column, recommends appropriate model families with starter code. Flags class imbalance and data readiness gaps.",
    "leakage_detector":    "Scans features for target leakage before model training: high correlation, post-event proxy columns, identifier columns that encode the target.",
    "feature_engineering": "Produces a complete PySpark MLlib preprocessing pipeline: per-column imputation/scaling/encoding decisions and runnable Pipeline code.",
    "causal_inference":    "Estimates the effect of a treatment (A/B variant, campaign, feature) on an outcome and says whether it can be trusted: sample-ratio mismatch, balance, confounding, bad controls, power.",
}

AVAILABLE_DATASETS = {
    "orders_dirty":       "Transactional orders table with known quality issues (null amounts, mixed-case statuses, duplicate IDs).",
    "sales_clean":        "Clean sales records with product categories, pricing tiers, and discount data.",
    "hr_attrition":       "HR dataset with salary, tenure, performance scores, and attrition labels (18% positive, 4.7:1 imbalance).",
    "ecommerce":          "E-commerce sessions with channel, device, conversion flag, and cart value (8% conversion rate, 11.7:1 imbalance).",
    "ecommerce_leakage":  "Same e-commerce sessions but with two planted leakage features: conversion_score (r=0.89 with target) and order_id (only exists post-conversion).",
    "transactions_fraud": "Financial transactions with 3% fraud rate (32:1 imbalance), extreme outlier amounts as fraud signal, mixed-case country values.",
    "ab_test_clean":        "Checkout A/B test: one row per user with variant (A/B), revenue_30d, pre-period revenue, country, device, tenure.",
    "ab_test_srm":          "A second run of the checkout A/B test, same columns, from a different logging pipeline.",
    "ab_test_bad_control":  "Onboarding A/B test: new_onboarding flag, orders_30d, sessions_first_week, country, device, tenure.",
    "observational_coupon": "Customers who did or didn't receive a marketing coupon (not randomised), with prior spend, loyalty status, tenure and spend over the next 90 days.",
}

# Causal datasets → (treatment, outcome, pre-period outcome, randomised?)
_CAUSAL_SETUP = {
    "ab_test_clean":        ("variant", "revenue_30d", "revenue_prev_30d", True),
    "ab_test_srm":          ("variant", "revenue_30d", "revenue_prev_30d", True),
    "ab_test_bad_control":  ("new_onboarding", "orders_30d", None, True),
    "observational_coupon": ("got_coupon", "spend_next_90d", None, False),
}

# Maps dataset name → natural target column for skills that need one
_DATASET_TARGETS = {
    "ecommerce":          "converted",
    "ecommerce_leakage":  "converted",
    "hr_attrition":       "attrition",
    "transactions_fraud": "is_fraud",
    "sales_forecasting":  "revenue",
}


MAX_SKILL_STEPS = 2


# ── Claude helpers ─────────────────────────────────────────────────────────────

def think_step(persona: dict, scenario: dict, transcript: list) -> dict:
    """Persona decides what to investigate next. Returns a decide_action tool result."""
    prior = ""
    for s in transcript:
        who = f"{s['persona_name']} ({s['persona_role']})"
        if s.get("skill"):
            interp = s.get("interpretation", "")[:220]
            prior += f"\n• {who} ran {s['skill']} on {s['dataset']}: {interp}"
        elif s.get("debrief"):
            prior += f"\n• {who} gave their final recommendation."

    if not prior:
        prior = "Nothing yet — you're the first to look at this."

    available_datasets = [d for d in scenario["available_datasets"]]
    skills_block = "\n".join(f"  • {k}: {v}" for k, v in AVAILABLE_SKILLS.items())
    datasets_block = "\n".join(
        f"  • {k}: {v}" for k, v in AVAILABLE_DATASETS.items()
        if k in available_datasets
    )

    focus_skills = scenario.get("focus_skills", [])
    focus_block = (
        "\nSKILLS MOST RELEVANT TO THIS SCENARIO\n"
        + "\n".join(f"  • {k}: {AVAILABLE_SKILLS[k]}" for k in focus_skills if k in AVAILABLE_SKILLS)
        + "\nThese are the highest-value skills for this specific question — strongly prefer them unless the team has already covered them.\n"
    ) if focus_skills else ""

    system = f"""You are {persona['name']}, {persona['role']}.
Your working style: {persona['approach']}
Your focus: {persona['focus']}

BUSINESS SCENARIO
{scenario['brief']}

ANALYTICAL CONTEXT
{scenario.get('context', '')}

WHAT THE TEAM HAS FOUND SO FAR
{prior}
{focus_block}
ALL AVAILABLE SKILLS
{skills_block}

AVAILABLE DATASETS
{datasets_block}

Think about the business question and what the team has already surfaced. What's the most valuable next investigation you could run? Reason through it in your own voice — consider what's still unknown, what could change the picture, and what the business actually needs to decide. Then choose your action.

If you have enough information to give a solid recommendation, choose give_debrief instead."""

    # 5.x models reject forced tool_choice, so the tool is strict + auto and the
    # prompt names it. Thinking counts toward max_tokens, hence the headroom.
    resp = _client().messages.create(
        **role("sim"),
        max_tokens=4096,
        system=system,
        messages=[{
            "role": "user",
            "content": "Choose your next action by calling the decide_action tool.",
        }],
        tools=[{
            "name": "decide_action",
            "strict": True,
            "description": "Choose your next step",
            "input_schema": {
                "type": "object",
                "required": ["reasoning", "action", "skill", "dataset"],
                "additionalProperties": False,
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "2-4 sentences in your own voice: why is this the right next step given the business question and what the team has already found?",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["run_skill", "give_debrief"],
                    },
                    "skill": {
                        "type": "string",
                        "enum": list(AVAILABLE_SKILLS.keys()) + ["none"],
                        "description": "The skill to run, or \"none\" when action=give_debrief",
                    },
                    "dataset": {
                        "type": "string",
                        "enum": list(AVAILABLE_DATASETS.keys()) + ["none"],
                        "description": "The dataset to run it on, or \"none\" when action=give_debrief",
                    },
                    "nl_question": {
                        "type": "string",
                        "description": "The specific business question to translate to SQL (required when skill=nl_to_spark)",
                    },
                },
            },
        }],
        tool_choice={"type": "auto"},
    )
    return tool_input(check_response(resp), "decide_action")


def interpret_step(persona: dict, skill: str, dataset: str, result: dict) -> str:
    """Persona reacts to skill results in their own voice."""
    summary = _result_summary(skill, result)

    resp = _client().messages.create(
        **role("sim"),
        max_tokens=2048,  # length is set by the prompt; headroom is for thinking
        system=(
            f"You are {persona['name']}, {persona['role']}. "
            f"Your style: {persona['approach']}"
        ),
        messages=[{
            "role": "user",
            "content": (
                f"You just ran {skill.replace('_', ' ')} on the {dataset} dataset.\n\n"
                f"Key findings:\n{summary}\n\n"
                "In your own voice: what does this tell you? "
                "What stands out most, and does it change anything about how you see the problem? "
                "Answer in at most 3 sentences (under 70 words) of plain prose — no headings or lists."
            ),
        }],
    )
    return response_text(check_response(resp))


def debrief_step(persona: dict, scenario: dict, my_steps: list) -> str:
    """Persona delivers their final business recommendation."""
    history = "\n".join(
        f"• Ran {s['skill']} on {s['dataset']}: {s.get('interpretation', '')}"
        for s in my_steps if s.get("skill")
    ) or "Worked from my teammates' findings."

    resp = _client().messages.create(
        **role("sim_judge"),
        max_tokens=4096,  # length is set by the prompt; headroom is for thinking
        system=(
            f"You are {persona['name']}, {persona['role']}. "
            f"Your style: {persona['approach']}"
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Business question: {scenario['brief']}\n\n"
                f"What you investigated:\n{history}\n\n"
                "Give your final recommendation in your own voice. Be specific and direct. "
                "What's the bottom line? What should the business actually do? "
                "What are the key risks or caveats they need to know? "
                "Keep it under 200 words: a bottom-line sentence, then at most 3 short bullets "
                "for actions and risks. No headings."
            ),
        }],
    )
    return response_text(check_response(resp))


# ── Skill execution ────────────────────────────────────────────────────────────

def _execute_skill(skill: str, dataset: str, decision: dict, scenario: dict) -> tuple[dict, dict]:
    """Run a skill and return (result_dict, run_record_for_store)."""
    # Tags any refusal-log entry with the dataset and scenario behind it
    with call_context(dataset=dataset, scenario=scenario["id"]):
        return _run_skill(skill, dataset, decision, scenario)


def _run_skill(skill: str, dataset: str, decision: dict, scenario: dict) -> tuple[dict, dict]:
    run_id = f"{skill}_{int(time.time() * 1000)}"
    ts = datetime.now().strftime("%H:%M")

    try:
        if skill == "data_quality":
            from databricks_client import load_sample_dataset
            from skills.data_quality import assess, build_profile
            df = load_sample_dataset(dataset)
            result = assess(df, table_name=dataset)
            profile = build_profile(df)
            findings = result.get("findings", [])
            run_record = {
                "id": run_id, "label": f"[Agent] {dataset}", "timestamp": ts,
                "status": "complete", "result": result, "profile": profile,
                "summary": {
                    "score":    result.get("overall_score", 0),
                    "critical": sum(1 for f in findings if f["severity"] == "critical"),
                    "warning":  sum(1 for f in findings if f["severity"] == "warning"),
                    "info":     sum(1 for f in findings if f["severity"] == "info"),
                },
            }

        elif skill == "eda":
            from databricks_client import load_sample_dataset
            from skills.eda import analyze
            df = load_sample_dataset(dataset)
            result = analyze(df, table_name=dataset)
            run_record = {
                "id": run_id, "label": f"[Agent] {dataset}", "timestamp": ts,
                "status": "complete", "result": result,
                "df_json": df.to_json(date_format="iso", orient="split"),
                "summary": {
                    "charts":   len(result.get("recommended_charts", [])),
                    "insights": len(result.get("key_insights", [])),
                    "targets":  len(result.get("potential_targets", [])),
                },
            }

        elif skill == "join_auditor":
            from databricks_client import load_join_scenario
            from skills.join_auditor import audit
            js_key = scenario.get("join_scenario", "orders_customers_fanout")
            left_df, right_df, key_str = load_join_scenario(js_key)
            join_keys = [k.strip() for k in key_str.split(",") if k.strip()]
            result = audit(left_df, right_df, join_keys, "inner")
            rc = result.get("row_count_change", {})
            run_record = {
                "id": run_id, "label": f"[Agent] {js_key}", "timestamp": ts,
                "status": "complete", "result": result,
                "summary": {
                    "verdict":        result.get("overall_verdict", "review_needed"),
                    "fan_out_factor": rc.get("fan_out_factor", 1.0),
                },
            }

        elif skill == "nl_to_spark":
            import pandas as pd
            from databricks_client import load_sample_dataset
            from skills.nl_to_spark import generate
            df = load_sample_dataset(dataset)
            question = decision.get("nl_question") or (
                f"Show revenue by {df.columns[1] if len(df.columns) > 1 else 'category'} "
                f"for top 10 records in {dataset}"
            )
            result = generate(question, {dataset: df})
            sql = result.get("sql_query", "")
            run_record = {
                "id": run_id, "label": f"[Agent] {question[:40]}…", "timestamp": ts,
                "status": "complete", "result": result,
                "summary": {"sql_preview": sql.split("\n")[0] if sql else ""},
            }

        elif skill == "mlops_review":
            from skills.mlops_reviewer import review
            code = scenario.get("mlops_code", "# no code provided")
            result = review(code)
            findings = result.get("findings", [])
            run_record = {
                "id": run_id, "label": "[Agent] pipeline notebook", "timestamp": ts,
                "status": "complete", "result": result,
                "summary": {
                    "score":    result.get("readiness_score", 0),
                    "critical": sum(1 for f in findings if f["severity"] == "critical"),
                    "warning":  sum(1 for f in findings if f["severity"] == "warning"),
                },
            }

        elif skill == "model_selector":
            from databricks_client import load_sample_dataset
            from skills.model_selector import select
            df = load_sample_dataset(dataset)
            target_col = _DATASET_TARGETS.get(dataset)
            if not target_col:
                raise ValueError(f"No target column defined for dataset '{dataset}'")
            result = select(df, target_col=target_col)
            recs = result.get("recommendations", [])
            run_record = {
                "id": run_id, "label": f"[Agent] {dataset} → {target_col}", "timestamp": ts,
                "status": "complete", "result": result,
                "summary": {
                    "task_type": result.get("task_type", "?"),
                    "top_model": recs[0].get("model_family", "?") if recs else "?",
                    "n_recommendations": len(recs),
                },
            }

        elif skill == "causal_inference":
            from databricks_client import load_sample_dataset
            from skills.causal_inference import analyze
            if dataset not in _CAUSAL_SETUP:
                raise ValueError(f"No treatment/outcome defined for dataset '{dataset}'")
            treatment, outcome, pre, randomized = _CAUSAL_SETUP[dataset]
            result = analyze(load_sample_dataset(dataset), treatment, outcome, randomized=randomized,
                             pre_outcome_col=pre, context=scenario.get("brief", ""))
            est = result.get("estimate") or {}
            run_record = {
                "id": run_id, "label": f"[Agent] {dataset}: {treatment} → {outcome}", "timestamp": ts,
                "status": "error" if result.get("error") else "complete", "result": result,
                "error": result.get("error"),
                "summary": {"verdict": result.get("verdict", "?"), "effect": est.get("effect")},
            }

        elif skill == "leakage_detector":
            from databricks_client import load_sample_dataset
            from skills.leakage_detector import check
            df = load_sample_dataset(dataset)
            target_col = _DATASET_TARGETS.get(dataset)
            if not target_col:
                raise ValueError(f"No target column defined for dataset '{dataset}'")
            result = check(df, target_col=target_col)
            findings = result.get("findings", [])
            run_record = {
                "id": run_id, "label": f"[Agent] {dataset}", "timestamp": ts,
                "status": "complete", "result": result,
                "summary": {
                    "verdict":    result.get("verdict", "?"),
                    "risk_level": result.get("risk_level", "?"),
                    "n_findings": len(findings),
                    "critical":   sum(1 for f in findings if f.get("severity") == "critical"),
                },
            }

        elif skill == "feature_engineering":
            from databricks_client import load_sample_dataset
            from skills.feature_engineering import suggest
            df = load_sample_dataset(dataset)
            target_col = _DATASET_TARGETS.get(dataset)
            if not target_col:
                raise ValueError(f"No target column defined for dataset '{dataset}'")
            result = suggest(df, target_col=target_col)
            features = result.get("features", [])
            active = [f for f in features if f.get("action") != "drop"]
            run_record = {
                "id": run_id, "label": f"[Agent] {dataset} pipeline", "timestamp": ts,
                "status": "complete", "result": result,
                "summary": {
                    "n_features":  len(active),
                    "n_dropped":   len(result.get("dropped_cols", [])),
                    "has_imputer": any(f.get("imputation", "none") != "none" for f in active),
                },
            }

        else:
            result = {}
            run_record = {"id": run_id, "label": dataset, "timestamp": ts, "status": "error",
                          "error": f"Unknown skill: {skill}", "summary": {}}

    except Exception as exc:
        result = {}
        run_record = {"id": run_id, "label": dataset, "timestamp": ts, "status": "error",
                      "error": str(exc), "summary": {}}

    return result, run_record


# Profiling data the skill computed for its prompt; the data scientist doesn't read it
_JUDGE_OMIT = {"profile"}
_JUDGE_MAX_FIELD = 6000


def _judge_view(result: dict) -> str:
    """The skill's full output as JSON, which is what the data scientist sees in the
    notebook. The rating must judge the skill, not _result_summary's truncation."""
    import json

    def clip(v):
        if isinstance(v, str) and len(v) > _JUDGE_MAX_FIELD:
            return v[:_JUDGE_MAX_FIELD] + " …[shortened by the simulation, not the skill]"
        if isinstance(v, list):
            return [clip(x) for x in v]
        if isinstance(v, dict):
            return {k: clip(x) for k, x in v.items()}
        return v

    return json.dumps({k: clip(v) for k, v in dict(result).items() if k not in _JUDGE_OMIT},
                      indent=1, default=str)


def _result_summary(skill: str, result: dict) -> str:
    """Build a brief text summary of skill results for the interpret prompt."""
    if skill == "data_quality":
        findings = result.get("findings", [])
        crits = sum(1 for f in findings if f["severity"] == "critical")
        warns = sum(1 for f in findings if f["severity"] == "warning")
        lines = [
            f"Overall quality score: {result.get('overall_score', '?')}/100",
            f"{crits} critical issues, {warns} warnings, {len(findings)} total findings",
            result.get("summary", ""),
        ]
        top = [f"  - {f['column']}: {f['description']}" for f in findings[:3]]
        return "\n".join(lines + top)

    elif skill == "eda":
        insights = result.get("key_insights", [])
        targets = result.get("potential_targets", [])
        lines = [
            result.get("narrative", ""),
            f"{len(insights)} key insights, {len(result.get('recommended_charts', []))} charts suggested",
            f"{len(targets)} potential ML targets identified",
        ]
        lines += [f"  - {i}" for i in insights[:3]]
        return "\n".join(lines)

    elif skill == "join_auditor":
        rc = result.get("row_count_change", {})
        findings = result.get("findings", [])
        return "\n".join([
            f"Verdict: {result.get('overall_verdict', '?')}",
            f"{rc.get('left_rows','?')} left × {rc.get('right_rows','?')} right → {rc.get('result_rows','?')} result rows",
            f"Fan-out factor: {rc.get('fan_out_factor', 1.0):.2f}x",
            result.get("summary", ""),
            *[f"  - {f['description']}" for f in findings[:2]],
        ])

    elif skill == "nl_to_spark":
        return "\n".join([
            result.get("explanation", ""),
            f"SQL: {result.get('sql_query','')[:200]}",
        ])

    elif skill == "mlops_review":
        findings = result.get("findings", [])
        crits = sum(1 for f in findings if f["severity"] == "critical")
        return "\n".join([
            f"Readiness score: {result.get('readiness_score', '?')}/100",
            f"{crits} critical issues, {len(findings)} total findings",
            result.get("summary", ""),
            *[f"  - {f['description']}" for f in findings[:3]],
        ])

    elif skill == "model_selector":
        recs = result.get("recommendations", [])
        ta = result.get("target_analysis", {})
        lines = [
            f"Task type: {result.get('task_type', '?')}  |  Data readiness: {result.get('data_readiness', '?')}",
            f"Class balance issue: {ta.get('class_balance_issue', '?')}  |  Imbalance ratio: {ta.get('imbalance_ratio', '?')}",
            f"{len(recs)} model recommendations",
        ]
        if recs:
            top = recs[0]
            lines.append(f"  Top: {top.get('model_family','?')} — {top.get('rationale','')[:150]}")
        return "\n".join(lines)

    elif skill == "leakage_detector":
        findings = result.get("findings", [])
        crits = sum(1 for f in findings if f.get("severity") == "critical")
        warns = sum(1 for f in findings if f.get("severity") == "warning")
        lines = [
            f"Verdict: {result.get('verdict','?')}  (risk: {result.get('risk_level','?')})",
            f"{crits} critical, {warns} warnings, {len(findings)} total findings",
            result.get("summary", ""),
        ]
        lines += [
            f"  - {f.get('column','?')} [{f.get('leakage_type','?')}]: {f.get('evidence','')[:120]}"
            for f in findings[:3]
        ]
        return "\n".join(lines)

    elif skill == "causal_inference":
        est = result.get("estimate") or {}
        lines = [f"Verdict: {result.get('verdict', '?')} ({result.get('design', '')}, "
                 f"{result.get('identification_strategy', '')})", result.get("summary", "")]
        if est.get("effect") is not None:
            lines.append(f"Estimate ({est.get('method')}): {est['effect']:+.3g} "
                         f"[{est['ci_low']:.3g}, {est['ci_high']:.3g}]")
        lines += [f"  - [{f.get('severity')}] {f.get('check')}: {f.get('evidence', '')[:120]}"
                  for f in result.get("findings", [])[:4]]
        return "\n".join(lines)

    elif skill == "feature_engineering":
        features = result.get("features", [])
        active = [f for f in features if f.get("action") != "drop"]
        dropped = result.get("dropped_cols", [])
        lines = [
            result.get("summary", ""),
            f"{len(active)} features in pipeline, {len(dropped)} columns dropped",
            "Per-column plan (sample):",
        ]
        lines += [
            f"  - {f.get('column','?')}: {f.get('action','?')} (impute={f.get('imputation','none')})"
            for f in active[:5]
        ]
        return "\n".join(lines)

    return str(result)[:400]


# ── Skill output evaluator ─────────────────────────────────────────────────────

def evaluate_output_step(
    persona: dict,
    skill: str,
    dataset: str,
    result: dict,
    scenario: dict,
) -> dict:
    """
    Evaluate a skill's output.  Returns a benchmark record combining:
    - quantitative auto-score (recall against ground-truth answer key)
    - qualitative persona rating (usefulness + clarity, 1-5 each)
    """
    from skills.auto_score import auto_score
    join_scenario = scenario.get("join_scenario")
    quant = auto_score(skill, result, dataset, scenario["id"], join_scenario)

    # ── Qualitative (persona rates the output) ────────────────────────────────
    result_view = _judge_view(result)

    try:
        qual_resp = _client().messages.create(
            **role("sim_judge"),
            max_tokens=2048,
            system=(
                f"You are {persona['name']}, {persona['role']}. "
                f"You just used the {skill} skill while working on this question: "
                f"{scenario['brief']}"
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"This is the complete output the {skill} skill returned on dataset "
                    f"'{dataset}' (what you'd see in your notebook):\n\n"
                    f"{result_view}\n\n"
                    "Rate the output on two dimensions:\n"
                    "- usefulness (1-5): Did it help you answer the business question?\n"
                    "- clarity (1-5): How easy was it to extract the key insight at a glance?\n"
                    "- brief_note: One sentence on what you'd improve (or 'none').\n\n"
                    "Submit your rating by calling the rate_output tool."
                ),
            }],
            # strict mode doesn't support minimum/maximum, so the 1-5 range is an enum
            tools=[{
                "name": "rate_output",
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "required": ["usefulness", "clarity", "brief_note"],
                    "additionalProperties": False,
                    "properties": {
                        "usefulness":  {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "clarity":     {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "brief_note":  {"type": "string"},
                    },
                },
            }],
            tool_choice={"type": "auto"},
        )
        check_response(qual_resp)
        qual_block = next((b for b in qual_resp.content
                           if b.type == "tool_use" and b.name == "rate_output"), None)
        if qual_block is None:
            raise RuntimeError(f"persona did not call rate_output (stop_reason={qual_resp.stop_reason})")
        qual = qual_block.input
        error = None
    except Exception as exc:
        # Record the failure rather than a made-up rating; overall=None keeps it
        # out of every aggregate (all readers filter on overall is not None).
        qual, error = None, f"{type(exc).__name__}: {exc}"

    # ── Combine ───────────────────────────────────────────────────────────────
    if error:
        overall, score_type = None, "error"
    elif quant is not None:
        qual_avg = (qual["usefulness"] + qual["clarity"]) / 2.0
        overall = round(0.6 * quant["score"] + 0.4 * qual_avg, 2)
        score_type = "quantitative+qualitative"
    else:
        overall = round((qual["usefulness"] + qual["clarity"]) / 2.0, 2)
        score_type = "qualitative"

    return {
        "status":        "error" if error else "ok",
        "error":         error,
        "id":            f"bench_{int(time.time() * 1000)}",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "skill":         skill,
        "dataset":       dataset,
        "scenario_id":   scenario["id"],
        "scenario_title": scenario["title"],
        "persona_name":  persona["name"],
        "persona_role":  persona["role"],
        "quantitative":  quant,
        "qualitative":   qual,
        "overall":       overall,
        "score_type":    score_type,
    }


# ── Main step runner ───────────────────────────────────────────────────────────

def run_sim_step(sim_state: dict) -> tuple[dict, dict, list[dict]]:
    """
    Run one complete step for the current persona.

    Returns (updated_sim_state, new_run_records, new_benchmarks).
    new_run_records is {skill_key: [run_record]} to merge into runs-store.
    new_benchmarks is a list of benchmark dicts (empty if no skill was run).
    """
    scenario_id = sim_state["scenario_id"]
    scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), SCENARIOS[0])

    persona_states = [dict(ps) for ps in sim_state["persona_states"]]
    transcript = list(sim_state.get("transcript", []))

    # Find first non-done persona
    current_idx = next(
        (i for i, ps in enumerate(persona_states) if not ps["done"]),
        None,
    )

    if current_idx is None:
        updated = dict(sim_state)
        updated["status"] = "complete"
        return updated, {}, []

    persona = PERSONAS[current_idx]
    ps = persona_states[current_idx]
    my_steps = [s for s in transcript if s.get("persona_name") == persona["name"]]

    # Decide next action (force debrief if max skill steps reached)
    if ps["steps_taken"] >= MAX_SKILL_STEPS:
        decision = {"action": "give_debrief", "reasoning": "I've gathered enough data. Time to give my recommendation."}
    else:
        decision = think_step(persona, scenario, transcript)

    step_record = {
        "persona_name":  persona["name"],
        "persona_role":  persona["role"],
        "persona_color": persona["color"],
        "persona_icon":  persona["icon"],
        "timestamp":     datetime.now().strftime("%H:%M"),
        "thinking":      decision.get("reasoning", ""),
        "action":        decision.get("action", "give_debrief"),
    }

    new_run_records = {}
    new_benchmarks: list[dict] = []

    if decision.get("action") == "run_skill":
        skill   = decision.get("skill")
        dataset = decision.get("dataset")
        if skill not in AVAILABLE_SKILLS:
            raise RuntimeError(f"{persona['name']} chose run_skill without a skill: {decision}")
        if dataset not in scenario["available_datasets"]:
            log.warning("%s picked dataset %r outside scenario %s; using %s",
                        persona["name"], dataset, scenario["id"], scenario["available_datasets"][0])
            dataset = scenario["available_datasets"][0]
        # Fallback: join_auditor needs its own scenario, not a generic dataset
        if skill == "join_auditor":
            dataset = scenario["available_datasets"][0]  # cosmetic label only

        result, run_record = _execute_skill(skill, dataset, decision, scenario)
        interpretation = interpret_step(persona, skill, dataset, result)

        # Evaluate output and record benchmark
        try:
            bench = evaluate_output_step(persona, skill, dataset, result, scenario)
            new_benchmarks.append(bench)
        except Exception:
            # never let evaluation failure break the simulation, but leave a trace
            log.exception("Evaluating %s on %s failed", skill, dataset)

        step_record.update({
            "skill":          skill,
            "dataset":        dataset,
            "run_id":         run_record["id"],
            "interpretation": interpretation,
            "summary":        run_record.get("summary", {}),
        })
        new_run_records[skill] = [run_record]
        ps["steps_taken"] += 1

    else:
        debrief = debrief_step(persona, scenario, my_steps)
        step_record["debrief"] = debrief
        ps["done"] = True

    updated = dict(sim_state)
    updated["transcript"] = transcript + [step_record]
    updated["persona_states"] = persona_states

    if all(p["done"] for p in persona_states):
        updated["status"] = "complete"

    return updated, new_run_records, new_benchmarks


def init_sim_state(scenario_id: str) -> dict:
    return {
        "status": "running",
        "scenario_id": scenario_id,
        "transcript": [],
        "persona_states": [{"steps_taken": 0, "done": False} for _ in PERSONAS],
    }


# ── Workbench reflection ───────────────────────────────────────────────────────

def reflect_step(persona: dict, scenario: dict, transcript: list) -> list[dict]:
    """
    Persona critiques the workbench itself after completing their analysis.
    Returns 2-3 structured improvement suggestions covering both workflow alignment
    and visual/usability quality.
    """
    my_steps = [s for s in transcript if s.get("persona_name") == persona["name"]]
    skills_used = [s["skill"] for s in my_steps if s.get("skill")]

    system = f"""You are {persona['name']}, {persona['role']}.

You just used AI-powered skills to work on this business question:
{scenario['brief']}

You used: {', '.join(skills_used) if skills_used else "none directly — you worked from teammates' findings"}.

These skills exist on two surfaces:
1. NOTEBOOK (primary): Importable Python — `from ds_workbench import leakage_detector; result = leakage_detector.check(df, 'converted')`. This is where the daily DS work happens in Databricks.
2. TEAM DASHBOARD (secondary): A web UI showing session overviews, skill benchmark quality trends, agent simulation results, and the improvement feedback loop. Used by team leads and for skill quality tracking — not where the analysis itself happens.

Now critique the *skill outputs and both surfaces* across two equally important dimensions:

**1. Notebook experience**
Imagine calling each skill directly in a Databricks notebook cell. Did the output give you exactly what you needed to make your next decision — or did it over-explain, under-explain, or bury the critical finding? Was the timing right in the DS workflow (e.g., does leakage detection naturally arrive before feature engineering)? Would the output render clearly in a notebook cell? Is there something the skill should have flagged that it didn't, or flagged that it shouldn't have? A skill that is technically correct but interrupts your mental model or produces 20 findings when 3 matter is failing you.

**2. Team dashboard value**
When these skill results appear in the team dashboard — do the findings surface at the right level of detail for a team lead scanning across multiple sessions? Are the benchmark quality trends informative enough to act on? Is there information the dashboard should show that it doesn't? Is anything cluttered or ambiguous that would slow down a 30-second review? The dashboard's job is team visibility, not analysis — it should answer "are things healthy, what changed, what needs attention."

Be specific and honest. Reference what you actually ran and what you saw in the outputs."""

    resp = _client().messages.create(
        **role("sim_judge"),
        max_tokens=8192,  # 2-3 tool calls plus thinking
        system=system,
        messages=[{
            "role": "user",
            "content": (
                "Give 2-3 specific improvements that would have helped you most. "
                "Cover both workflow friction and usability/layout issues — treat them as equal priorities. "
                "Think about: moments where the tool slowed your thinking, outputs that were hard to read "
                "or interpret, navigation that wasn't obvious, information that was buried when it should "
                "have been front and center, or steps that felt out of order with how you actually work. "
                "Each suggestion must be concrete enough that a developer could implement it this week. "
                "Submit each one as a separate suggest_improvement tool call."
            ),
        }],
        tools=[strict_tool({
            "name": "suggest_improvement",
            "description": "Suggest a specific, implementable improvement to the workbench",
            "input_schema": {
                "type": "object",
                "required": ["title", "description", "skill_affected", "issue_type",
                             "priority", "implementation_hint", "user_story"],
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, specific title (< 60 chars)",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "2-3 sentences describing the specific friction or gap — could be a missing "
                            "feature, a layout/visual issue, a workflow mismatch, or a confusing output. "
                            "Be precise about when it happened and why it mattered to your work."
                        ),
                    },
                    "skill_affected": {
                        "type": "string",
                        "enum": ["data_quality", "eda", "join_auditor", "nl_to_spark",
                                 "mlops_review", "model_selector", "leakage_detector",
                                 "feature_engineering", "experiment_tracker",
                                 "notebook_generator", "causal_inference", "agent_sim", "general"],
                    },
                    "issue_type": {
                        "type": "string",
                        "enum": ["missing_feature", "ux_friction", "visualization_gap",
                                 "workflow_improvement", "notebook_output_quality",
                                 "data_insight"],
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                    "implementation_hint": {
                        "type": "string",
                        "description": (
                            "Concrete technical suggestion: which file or component, what to change, "
                            "and how — could be a layout tweak, a new UI element, reordering steps, "
                            "improving an output display, or adding new functionality."
                        ),
                    },
                    "user_story": {
                        "type": "string",
                        "description": "As a [role], I want [feature] so that [benefit]",
                    },
                },
            },
        })],
        tool_choice={"type": "auto"},
    )

    suggestions = []
    for block in resp.content:
        if block.type == "tool_use" and block.name == "suggest_improvement":
            suggestions.append(block.input)

    return suggestions


def run_all_personas(scenario_id: str) -> tuple[dict, list[dict], list[dict]]:
    """
    Run a complete simulation (all personas) + usability reflection + benchmarking.
    Returns (final_sim_state, all_recommendations, all_benchmarks).
    Called by the scheduler — no Dash dependency.
    """
    state = init_sim_state(scenario_id)
    all_benchmarks: list[dict] = []

    while state["status"] == "running":
        state, _, new_benches = run_sim_step(state)
        all_benchmarks.extend(new_benches)

    scenario = next(s for s in SCENARIOS if s["id"] == scenario_id)
    recommendations = []

    for persona in PERSONAS:
        try:
            recs = reflect_step(persona, scenario, state["transcript"])
            for rec in recs:
                rec.update({
                    "persona_name":    persona["name"],
                    "persona_role":    persona["role"],
                    "persona_color":   persona["color"],
                    "scenario_id":     scenario_id,
                    "scenario_title":  scenario["title"],
                })
            recommendations.extend(recs)
        except Exception:
            # don't let one reflection failure abort the whole run
            log.exception("Reflection by %s failed", persona["name"])

    return state, recommendations, all_benchmarks
