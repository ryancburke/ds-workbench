"""
Run all five skills against synthetic datasets and print results.

Usage:
    .venv/bin/python test_skills.py
"""

import sys
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config import config  # triggers load_dotenv; all skills read config.anthropic_api_key

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

SEV_COLOR = {"critical": RED, "warning": YELLOW, "info": CYAN,
             "suggestion": CYAN, "safe": GREEN}

def hr(char="─", width=72): print(DIM + char * width + RESET)
def header(title): hr("═"); print(f"{BOLD}{CYAN}  {title}{RESET}"); hr("═")
def section(title): print(f"\n{BOLD}{title}{RESET}"); hr()


# ── Synthetic datasets ────────────────────────────────────────────────────────

def make_messy_orders(n=800) -> pd.DataFrame:
    """Orders table with injected quality issues."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "order_id":    range(1001, n + 1001),
        "customer_id": [f"C{rng.integers(1, 200):04d}" if rng.random() > 0.04 else None
                        for _ in range(n)],
        "amount":      np.where(rng.random(n) > 0.09, rng.exponential(120, n).round(2), None),
        "status":      rng.choice(
            ["completed", "pending", "Completed", "CANCELLED", "refunded", None],
            n, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
        ),
        "region":      rng.choice(["NORTH", "South", "east", "West", None], n,
                                  p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        "score":       rng.normal(65, 20, n).clip(0, 100).round(1),
        "created_at":  pd.date_range("2023-01-01", periods=n, freq="2h"),
    })
    # inject 3% duplicate rows
    dupes = df.sample(int(n * 0.03), random_state=7)
    return pd.concat([df, dupes], ignore_index=True).head(n)


def make_clean_sales(n=600) -> pd.DataFrame:
    """Clean sales dataset — good for EDA and NL→PySpark."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "sale_id":     range(1, n + 1),
        "product":     rng.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], n),
        "category":    rng.choice(["Electronics", "Accessories", "Bundles"], n, p=[0.5, 0.3, 0.2]),
        "quantity":    rng.integers(1, 20, n),
        "unit_price":  rng.choice([9.99, 19.99, 49.99, 99.99, 199.99], n),
        "discount_pct":rng.choice([0, 5, 10, 15, 20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "region":      rng.choice(["North", "South", "East", "West"], n),
        "sale_date":   pd.date_range("2024-01-01", periods=n, freq="6h"),
    })


def make_join_pair(n=500):
    """Orders + customers — right table has duplicate customer_ids causing fan-out."""
    rng = np.random.default_rng(7)
    orders = pd.DataFrame({
        "order_id":    range(1, n + 1),
        "customer_id": rng.integers(1, 100, n),
        "amount":      rng.exponential(100, n).round(2),
    })
    # Customers table has ~10% duplicate customer_ids (simulates a bad ETL)
    base_customers = pd.DataFrame({
        "customer_id": range(1, 101),
        "name":        [f"Customer {i}" for i in range(1, 101)],
        "tier":        rng.choice(["gold", "silver", "bronze"], 100),
    })
    dupes = base_customers.sample(10, random_state=42)
    customers = pd.concat([base_customers, dupes], ignore_index=True)
    return orders, customers


NOTEBOOK_CODE = """\
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

# Load data — hardcoded path, no error handling
df = spark.read.parquet("/mnt/data/transactions/2024-01-15/")

# Filter to last 30 days — hardcoded date
from datetime import date
df = df.filter(df.date >= "2023-12-15")

# Compute revenue — magic number
df = df.withColumn("revenue", df.amount * 1.082)

# Collect to pandas for processing — will OOM on large data
result = df.toPandas()

# Row-wise loop instead of vectorised ops
for i, row in result.iterrows():
    result.at[i, "category"] = "high" if row["revenue"] > 500 else "low"

result.to_csv("/tmp/output.csv")
print("done")
"""


# ── Skill runners ─────────────────────────────────────────────────────────────

def run_data_quality():
    from skills.data_quality import assess
    df = make_messy_orders()
    t0 = time.time()
    result = assess(df, table_name="orders")
    elapsed = time.time() - t0

    section("1 / 5  DATA QUALITY AUDIT  →  orders (messy)")
    score = result["overall_score"]
    color = RED if score < 50 else YELLOW if score < 75 else GREEN
    print(f"  Score: {color}{BOLD}{score}/100{RESET}  ({elapsed:.1f}s)")
    print(f"  {result['summary']}\n")
    for f in result["findings"]:
        c = SEV_COLOR.get(f["severity"], "")
        bullet = f"  {c}[{f['severity'].upper():<8}]{RESET}  "
        bullet += f"{BOLD}{f.get('column','—'):<16}{RESET}  {f['issue_type'].replace('_',' ')}"
        print(bullet)
        print(f"    {DIM}{f['description']}{RESET}")
    return result


def run_eda():
    from skills.eda import analyze
    df = make_clean_sales()
    t0 = time.time()
    result = analyze(df, table_name="sales")
    elapsed = time.time() - t0

    section("2 / 5  EDA ACCELERATOR  →  sales (clean)")
    print(f"  {result['narrative']}  {DIM}({elapsed:.1f}s){RESET}\n")
    print(f"  {BOLD}Key Insights:{RESET}")
    for insight in result.get("key_insights", []):
        print(f"    • {insight}")
    charts = result.get("recommended_charts", [])
    print(f"\n  {BOLD}Recommended Charts ({len(charts)}):{RESET}")
    for ch in charts:
        x = ch.get("x_column", ""); y = ch.get("y_column", "")
        axes = f"{x} × {y}" if x and y else (x or y or "")
        print(f"    {CYAN}{ch['chart_type']:<12}{RESET}  {ch['title']}")
        if axes:
            print(f"    {DIM}           columns: {axes}{RESET}")
    targets = result.get("potential_targets", [])
    if targets:
        print(f"\n  {BOLD}ML Targets:{RESET}")
        for t in targets:
            print(f"    {GREEN}{t['column']:<18}{RESET}  {t['task_type']}  —  {t['rationale']}")
    return result


def run_join_auditor():
    from skills.join_auditor import audit
    orders, customers = make_join_pair()
    t0 = time.time()
    result = audit(orders, customers, ["customer_id"], "inner", "orders", "customers")
    elapsed = time.time() - t0

    section("3 / 5  JOIN AUDITOR  →  orders ⋈ customers on customer_id")
    verdict = result["overall_verdict"]
    vc = SEV_COLOR.get(verdict, "")
    rc = result.get("row_count_change", {})
    print(f"  Verdict: {vc}{BOLD}{verdict.replace('_',' ').upper()}{RESET}  ({elapsed:.1f}s)")
    print(f"  Rows:  {rc.get('left_rows','?'):,} left  ×  {rc.get('right_rows','?'):,} right"
          f"  →  {rc.get('result_rows','?'):,} result  (fan-out ×{rc.get('fan_out_factor',1):.2f})\n")
    print(f"  {result.get('summary','')}\n")
    for f in result.get("findings", []):
        c = SEV_COLOR.get(f["severity"], "")
        print(f"  {c}[{f['severity'].upper():<8}]{RESET}  {f['issue_type'].replace('_',' ')}")
        print(f"    {DIM}{f['description']}{RESET}")
    return result


def run_nl_to_spark():
    from skills.nl_to_spark import generate
    tables = {"sales": make_clean_sales()}
    question = "What are the top 5 products by total revenue, broken down by region?"
    t0 = time.time()
    result = generate(question, tables)
    elapsed = time.time() - t0

    section(f"4 / 5  NL → PYSPARK  →  \"{question}\"")
    print(f"  {DIM}({elapsed:.1f}s){RESET}\n")
    print(f"  {BOLD}Explanation:{RESET}  {result.get('explanation','')}\n")
    print(f"  {BOLD}Spark SQL:{RESET}")
    for line in result.get("sql_query", "").strip().splitlines():
        print(f"    {CYAN}{line}{RESET}")
    print(f"\n  {BOLD}PySpark:{RESET}")
    for line in result.get("pyspark_code", "").strip().splitlines():
        print(f"    {GREEN}{line}{RESET}")
    notes = result.get("performance_notes", [])
    if notes:
        print(f"\n  {BOLD}Performance notes:{RESET}")
        for n in notes:
            print(f"    • {n}")
    return result


def run_mlops_review():
    from skills.mlops_reviewer import review
    t0 = time.time()
    result = review(NOTEBOOK_CODE)
    elapsed = time.time() - t0

    section("5 / 5  MLOPS REVIEW  →  notebook with 5 known issues")
    score = result.get("readiness_score", 0)
    color = RED if score < 40 else YELLOW if score < 70 else GREEN
    print(f"  Readiness: {color}{BOLD}{score}/100{RESET}  ({elapsed:.1f}s)")
    print(f"  {result.get('summary','')}\n")
    for f in result.get("findings", []):
        c = SEV_COLOR.get(f["severity"], "")
        print(f"  {c}[{f['severity'].upper():<10}]{RESET}  {BOLD}{f['category'].replace('_',' '):<22}{RESET}")
        if f.get("line_reference"):
            print(f"    {DIM}code: {f['line_reference'][:80]}{RESET}")
        print(f"    {f['description']}")
        print(f"    {DIM}fix: {f['fix'][:120]}{RESET}")
    snippet = result.get("refactored_snippet")
    if snippet:
        print(f"\n  {BOLD}Refactored snippet:{RESET}")
        for line in snippet.strip().splitlines():
            print(f"    {CYAN}{line}{RESET}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

RUNNERS = [
    ("data_quality", run_data_quality),
    ("eda",          run_eda),
    ("join_auditor", run_join_auditor),
    ("nl_to_spark",  run_nl_to_spark),
    ("mlops_review", run_mlops_review),
]


def main():
    if not config.anthropic_api_key:
        print(f"{RED}ANTHROPIC_API_KEY not set in .env or environment.{RESET}")
        sys.exit(1)

    header(f"DS WORKBENCH  —  SKILL TEST SUITE  (5 skills, parallel)")

    results: dict = {}
    errors:  dict = {}

    print(f"\n  Running all 5 skills in parallel against synthetic data...\n")

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn): name for name, fn in RUNNERS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                errors[name] = exc

    # ── Summary ───────────────────────────────────────────────────────────────
    header("SUMMARY")
    for name, fn in RUNNERS:
        if name in errors:
            print(f"  {RED}✗  {name:<20}{RESET}  {errors[name]}")
        else:
            print(f"  {GREEN}✓  {name}{RESET}")

    if errors:
        print(f"\n{RED}  {len(errors)} skill(s) failed.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}  All 5 skills passed.{RESET}")


if __name__ == "__main__":
    main()
