# DS Workbench — Project Context for Claude

## What this is

A two-surface AI-powered data science platform:

1. **Skills library** (`skills/`) — pure Python, callable from Databricks notebooks. Data
   scientists import and run skills inline during their normal notebook workflow. Skills provide
   AI-backed analysis (DQ assessment, EDA, model selection, leakage detection, etc.) with rich
   HTML output that renders in notebook cells.

2. **Dash team dashboard** (`app.py` + `pages/`) — browser-based visibility layer for teams.
   Shows session overviews, skill benchmark quality trends, simulation results, and the feedback
   loop. Three autonomous agent personas continuously simulate DS workflows, critique skills after
   every run, and surface structured improvement recommendations — creating a self-improving loop.

The notebook is where data scientists do their work. The Dash app is where the team sees what's
happening and where skill quality is tracked over time.

## North star

**Technically sharp, and high usability.** Skills must fit naturally into how a data scientist
actually thinks — from problem framing → data discovery → quality assessment → exploration →
feature engineering → modeling → communication. A skill that is technically correct but interrupts
that mental flow, buries key findings, or makes the DS think about the tool instead of the
problem, is failing.

The primary DS interface is the **notebook**, not the Dash app. Skills should feel like a smart
colleague looking over your shoulder — they speak up at the right moments (bad data before
modeling, target leakage before training) and stay out of the way otherwise. The Dash app is the
team and quality layer, not the execution layer.

## Two-surface model (settled — don't revisit without reason)

```
Notebook (primary DS surface)          Dash app (team / quality layer)
─────────────────────────────          ───────────────────────────────
from ds_workbench import dq            Session overview across the team
dq.check(df)                           Skill benchmark quality trends
                                       Agent simulation + feedback loop
model_selector.select(df, "churn")     MLOps review, benchmarks
leakage.check(X_train, y_train)        Stakeholder-facing summaries
```

Skills are versioned. Before any skill change ships, it runs against the benchmark suite
(ground truth datasets with known expected outputs) and is scored. The Dash benchmarks page
shows quality trends per skill over time.

## Skill development pipeline

```
1. Write / modify skill in skills/
2. Run benchmark suite → precision/recall vs ground truth
3. Diff output vs. previous version
4. Run agent simulation → qualitative workflow fitness
5. Review in Dash (Benchmarks + Agent Sim pages)
6. Approve → version promoted to stable
```

## Stack

- **Dash 4.4.1** — `use_pages=True`, pattern-matching callbacks (`MATCH`/`ALL`), `dcc.Store`,
  `allow_duplicate=True` for multiple callbacks writing to the same output
- **Dash Bootstrap Components** — DARKLY theme throughout; never introduce a different theme
- **Anthropic SDK** — model IDs centralized in `skills/models.py`; Opus 5.5 for heavy
  reasoning (DQ), Fable 5.1 for leakage, Sonnet 5.5 for general analysis, Haiku 5.5 for
  structured extraction (Pass 2); all structured outputs use forced tool calls
  (`tool_choice={"type":"tool","name":"..."}`) so JSON is guaranteed; all 5.x models use
  `thinking={"type":"adaptive"}` (the 4.x `enabled`+`budget_tokens` format is retired)
- **APScheduler `BackgroundScheduler`** — runs simulations on a configurable interval
  (`SIM_INTERVAL_HOURS`, default 6h); guarded against Werkzeug double-start via `WERKZEUG_RUN_MAIN`
- **python-dotenv** — `load_dotenv()` is called at the top of `config.py` before `Config()`
  instantiation; this is the only place env vars are loaded
- Python 3.11+, `.venv` virtualenv at `.venv/`

## CRITICAL: API key handling

**Never ask for the `ANTHROPIC_API_KEY`. Never export it into the shell. Never hardcode it.**
The key lives exclusively in `.env` (gitignored), loaded by `config.py`'s `load_dotenv()` call.
All code accesses it via `from config import config` then `config.anthropic_api_key`.
Test scripts use `.venv/bin/python` and check `config.anthropic_api_key`, not `os.getenv`.

## Project structure

```
app.py                    # Dash app, global layout, global callbacks, scheduler start
config.py                 # load_dotenv() + Config dataclass — single source of truth for env
databricks_client.py      # Databricks connector; LOCAL_DEV_MODE=true uses synthetic data
requirements.txt

pages/                    # One file per Dash page (use_pages=True)
  home.py                 # Workbench canvas — single-page, all 6 skills as dbc.Accordion
                          # Route: /, sidebar link: "Workbench"
  data_quality.py         # Data Quality standalone page (/data-quality)
  eda.py                  # EDA standalone page (/eda)
  join_auditor.py         # Join Auditor standalone page (/join-auditor)
  nl_to_spark.py          # NL → PySpark standalone page (/nl-to-spark)
  mlops_review.py         # MLOps Review standalone page (/mlops-review)
  agent_sim.py            # Autonomous agent simulation UI (/agent-sim)
  feedback.py             # Feedback loop dashboard (/feedback)
  benchmarks.py           # Skill quality benchmark dashboard (/benchmarks)

skills/                   # Pure Python — no Dash dependency. Called by pages, agents,
                          # AND importable directly from Databricks notebooks.
  data_quality.py         # assess(df, table_name) → findings, score, summary
  eda.py                  # analyze(df, table_name) → insights, charts, targets
  join_auditor.py         # audit(left_df, right_df, key) → verdict, findings, stats
  nl_to_spark.py          # generate(question, tables) → sql, pyspark, explanation
  mlops_reviewer.py       # review(code) → feasibility, findings, score
  model_selector.py       # select(df, target_col, context) → task_type, recommendations

benchmarks/               # Skill quality testing infrastructure
  answer_keys/            # Ground truth expected outputs per dataset per skill (JSON)
  scoring.py              # Compare skill output vs answer key → precision/recall metrics
  runner.py               # Run full benchmark suite, write results to benchmark_history.json
  benchmark_history.json  # Time-series of benchmark scores (gitignored or tracked)

agents/
  simulation.py           # Personas, scenarios, think/interpret/debrief/reflect steps
  scheduler.py            # RecommendationStore, BackgroundScheduler, draft/apply code changes

components/
  sidebar.py              # Sidebar: 4 nav links (Workbench, Feedback, Benchmarks, Agent Sim)
                          # SKILLS list preserved for app.py SKILL_KEYS (run-count badges)
  run_card.py             # Reusable run history card component
  dq_gate.py              # find_dq_run(), dq_gate_banner() — DQ status in skill pages
  session_context.py      # session_banner() — active session strip in skill pages

assets/style.css          # Global dark-theme CSS overrides
feedback/                 # Persisted JSON files (gitignored)
  recommendations.json
  sim_history.json
```

## Workbench canvas (home.py)

`home.py` is a single-page workbench with all skills as `dbc.Accordion(always_open=True)`:
- **dq** — Data Quality (wb-dq-*)
- **eda** — EDA (wb-eda-*)
- **ms** — Model Selector (wb-ms-*)
- **ld** — Leakage Detector (wb-ld-*)
- **fe** — Feature Engineering (wb-fe-*)
- **ja** — Join Auditor (wb-ja-*)
- **nl** — NL → PySpark (wb-nl-*)
- **ml** — MLOps Review (wb-ml-*)
- **et** — Experiment Tracker (wb-et-*)

Session setup card at the top: problem statement + dataset checklist → "Start Session" button.
On start: runs DQ on all selected datasets automatically, writes session to runs-store, hides
setup card and shows active session card. `dcc.Loading` wraps the start button so spinner is
visible during the 20-60s DQ run.

Active session card shows: pipeline health summary, 7-step guided workflow checklist
(data_quality → eda → join_auditor → leakage_detector → feature_engineering → model_selector →
mlops_review), cross-skill conflict detection panel, and "Generate Starter Notebook" button.

Each skill render leads with a verdict hero (BLOCKING/SAFE/CLEAR/etc.) as the first element so
the go/no-go decision is always front and center before charts or tables.

## Skills — built and planned

### Built (10 skills)
| Key | Skill | Entry point | Benchmarked |
|-----|-------|-------------|-------------|
| `data_quality` | Data Quality Assessor | `skills/data_quality.assess()` | ✓ |
| `eda` | EDA Accelerator | `skills/eda.analyze()` | ✓ |
| `join_auditor` | Join Auditor | `skills/join_auditor.audit()` | ✓ |
| `nl_to_spark` | NL → PySpark | `skills/nl_to_spark.generate()` | ✓ |
| `mlops_review` | MLOps Reviewer | `skills/mlops_reviewer.review()` | ✓ |
| `model_selector` | Model Selector | `skills/model_selector.select()` | ✓ |
| `leakage_detector` | Leakage Detector | `skills/leakage_detector.check()` | ✓ |
| `feature_engineering` | Feature Engineering Assistant | `skills/feature_engineering.suggest()` | ✓ |
| `notebook_generator` | Starter Notebook Generator | `skills/notebook_generator.generate()` | — |
| `experiment_tracker` | Experiment Tracker | `skills/experiment_tracker.interpret()` | — |

The first 8 skills are importable via `from ds_workbench import <skill>` (see `ds_workbench/__init__.py`).
`notebook_generator` and `experiment_tracker` are workbench-only (no notebook import needed — they
consume session/training artifacts rather than raw DataFrames).

### Workflow the skills cover end-to-end
```
Start session → data_quality (auto-run on all datasets)
             → eda → join_auditor → leakage_detector → feature_engineering
             → model_selector → nl_to_spark (ad-hoc queries)
             → mlops_review → notebook_generator (export findings as .ipynb)
             → [train in Databricks]
             → experiment_tracker (paste metrics → next steps)
             → back to session
```

### Planned (next)
Nothing currently planned — the core workflow is complete.

## Benchmark suite architecture

Ground truth answer keys live in `benchmarks/answer_keys/<dataset>/<skill>.json`. Each key
specifies the expected findings, scores, and recommendations for a given (dataset, skill) pair.

Benchmark datasets and what they test:
- `sales_clean` — clean data; DQ should score high, no critical findings
- `orders_dirty` — planted issues: nulls, duplicates, mixed-case IDs, type errors; DQ must catch all
- `hr_attrition` — binary classification with known class imbalance; Model Selector must flag imbalance
- `ecommerce` — conversion prediction baseline; EDA, Model Selector
- `ecommerce_leakage` — same as ecommerce with two planted leakage features; Leakage Detector must flag both
- `transactions_fraud` — 3% fraud rate (32:1 imbalance), mixed-case country, user_id nulls, ID columns as leakage risk

Scoring (`benchmarks/scoring.py`):
- **Precision**: findings Claude reported that match ground truth / all findings Claude reported
- **Recall**: ground truth findings Claude caught / all ground truth findings
- **Correctness**: structured fields (verdict, task_type, etc.) match expected values exactly

Run the suite: `.venv/bin/python benchmarks/runner.py` — appends to `benchmark_history.json`.

## Dash architecture decisions (settled — don't revisit without reason)

**Global runs-store**: `dcc.Store(id="runs-store", storage_type="session")` lives in `app.layout`
so run history survives page navigation. Keyed by skill name plus `"session"` key. All skill
pages write to it with `allow_duplicate=True`.
`EMPTY_STORE = {key: [] for key in SKILL_KEYS} | {"session": None}`

**Run-card pattern**: Every skill page renders a list of past runs from the store. Each run has
a toggle/collapse, a dismiss button, and rendered output. Pattern-matching callbacks
(`{"type":"run-toggle","index":run_id}` with `MATCH`) are used so one callback handles all cards.

**Feedback loop**: Agent personas run `reflect_step` after each simulation, returning 2-3
structured `suggest_improvement` tool calls. Recommendations are deduplicated by title similarity
(>50% word overlap) and ranked by `priority_weight × votes`. The feedback dashboard shows
pending → drafted → implemented → dismissed tabs with one-click code drafting and one-click apply
(syntax-checked, `.bak` backup created first).

**Callback `prevent_initial_call`**: Any callback whose Input is a `dcc.Store` that lives inside
a page layout (not the global `app.layout`) must have `prevent_initial_call=True`. The store's
initial value fires the callback through a different internal path in Dash 4.x that causes
"Wrong length output_spec" errors.

**`dismiss_run` handles non-list store values**: The `session` key in runs-store is a dict (or
None), not a list. The dismiss_run callback uses `isinstance(val, list)` before iterating.

**Severity-ranked output**: DQ and Join Auditor findings are sorted critical → warning → info
with section header rows. Critical rows have a red left border, warning rows amber.

**Feasibility-first MLOps**: MLOps skill leads with a data/feature sufficiency verdict card
before any code-level findings.

## Simulation system

Three personas run autonomous simulations against five scenarios. Each persona gets up to
`MAX_SKILL_STEPS = 2` skill runs, then gives a final debrief. After all personas complete,
each runs `reflect_step` to produce 2-3 structured improvement suggestions.

**Personas**
- **Alex Chen** — Lead Data Scientist; methodical, grounds everything in data quality first
- **Priya Sharma** — ML Engineer; production-minded, owns reproducibility and pipeline integrity
- **Jordan Kim** — Data Analyst; business-first, translates findings to executive language

**Scenarios** (5 total, each with `focus_skills` that bias Claude's skill selection)
- `q3_revenue` — Revenue shortfall; focus: data_quality, join_auditor, nl_to_spark, mlops_review
- `churn_risk` — Retention drop; focus: eda, model_selector, leakage_detector, data_quality
- `model_readiness` — Pre-flight model check; focus: leakage_detector, feature_engineering, model_selector, data_quality
- `pipeline_handoff` — New pipeline acceptance; focus: data_quality, join_auditor, mlops_review, nl_to_spark
- `fraud_detection` — Fraud model pre-flight; focus: leakage_detector, model_selector, feature_engineering, data_quality

**Coverage-weighted scheduler** (`agents/scheduler.py`): instead of round-robin, `_coverage_weighted_scenario()`
counts benchmark runs per skill and picks whichever scenario's `focus_skills` have the fewest runs.
This prevents DQ/MLOps from dominating while model_selector/leakage_detector stay underrepresented.
Falls back to round-robin if benchmark data is unavailable. Interval: `SIM_INTERVAL_HOURS` (default 6h).

## Local dev

```bash
# Install deps
.venv/bin/pip install -r requirements.txt

# Run app
.venv/bin/python app.py          # debug mode, port 8050
# or
LOCAL_DEV_MODE=true .venv/bin/python app.py

# Run skill tests (reads .env automatically via config.py)
.venv/bin/python test_skills.py

# Run benchmark suite
.venv/bin/python benchmarks/runner.py
```

`LOCAL_DEV_MODE=true` bypasses Databricks and uses synthetic dataset generators in
`databricks_client.py`. This is the default for development.

## What good looks like

When evaluating a change, ask: does this make a data scientist's work faster and clearer, or does
it add friction? Key signals:
- Skills speak up at the right moment in the workflow — not too early, not too late
- Key findings are front and center, not buried in output
- Outputs are scannable — a practitioner can extract the insight in a few seconds
- Technical quality (reproducibility, correct stats, valid code) is never compromised for aesthetics
- The tool earns trust by being honest about data quality before drawing conclusions
- A new skill passes its benchmark suite before it ships — correctness is not optional
