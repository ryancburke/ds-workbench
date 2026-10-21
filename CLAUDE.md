# DS Workbench — Project Context for Claude

## Project rules (always apply)

1. **Use the venv explicitly.** The shell does not activate `.venv`. Run Python with
   `.venv/bin/python` and install packages with `.venv/bin/pip` — never bare `python`/`pip`.
2. **API keys live in `.env` only**, loaded by python-dotenv in `config.py`. Never ask for
   keys, print them, log them, or export them into the shell. See "API key handling" below.
3. **Model names are settings, never hardcoded.** Model IDs are read from `config.py`
   (overridable via `.env`). No `"claude-..."` string literals in skills, agents, pages, or
   benchmarks.
4. **Feature branches + pull requests.** Never commit or push directly to `main`. Branch,
   commit there, and open a PR.
5. **A failed or missing check never counts as passed.** A skill result with `{"error": ...}`
   is a failed check and a mandatory check that never ran is a missing one; both make the work
   *incomplete*. Never default an unknown or absent verdict to ready, clean or cleared: the
   policy fails closed — a missing or unrecognised skill verdict makes the check failed, and a
   finding with an unknown severity counts as a caution.
6. **Readiness is computed in one place:** `ds_workbench.core.readiness.assess()`. Preflight,
   the Sessions page, the Workbench strip, pipeline-health card and session digest all call it
   and only decide how to display the result. Don't add readiness rules anywhere else.
   Availability confirmations (`WorkflowSession.confirm_available`) are the user's judgment,
   never the model's: they're passed to `assess()` separately, need a note, and apply only to
   the exact (column, dataset, prediction point) they were made for.
   Statuses: `blocked` (critical findings) > `incomplete` (missing/failed mandatory check) >
   `caution` (warnings) > `ready_for_review` (every mandatory check ran clean). Mandatory checks:
   data quality + leakage for modelling work; data quality + causal for causal-only work.

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
- **Anthropic SDK** — model IDs and per-role effort levels are settings in `config.py`
  (overridable in `.env`; see `.env.example`). Calls use `**role("<role>")` from
  `skills/models.py` (model + effort, read at call time); `override_roles()` swaps a role
  for sweeps. Sonnet 5.5 for data quality, Opus 5.5 for leakage, Sonnet 5.5 for general analysis and simulation personas, Opus 5.5 for
  simulation judging, Haiku 5.5 for structured extraction (Pass 2). Every call uses the
  shared client `skills.api_retry.get_client()` (the SDK owns retries) and goes through
  `api_call` / `check_response`, which raise `ModelRefusal` on `stop_reason == "refusal"`.
  Read tool calls with `tool_input(resp, name)`, which raises a clear error when the model
  answered without the tool, and wrap tools used under `auto` in `strict_tool()`. Pass 2 uses forced tool calls
  (`tool_choice={"type":"tool","name":"..."}`) — this is valid **only on Haiku 5.5**; Opus 5.5,
  Sonnet 5.5, and Fable 5.1 return a 400 on forced `tool` / `any`, so on those use
  `tool_choice={"type":"auto"}` + `strict: true` or `output_config.format`. All 5.x models use
  adaptive thinking (`budget_tokens` returns a 400). Thinking is on by default on 5.x, so read
  response blocks by `type`, never `content[0].text`
- **APScheduler `BackgroundScheduler`** — runs simulations on a configurable interval
  (`SIM_INTERVAL_HOURS`; default 0 = manual, run from the Feedback page or `run_sim.py`); guarded against Werkzeug double-start via `WERKZEUG_RUN_MAIN`
- **python-dotenv** — `load_dotenv()` is called at the top of `config.py` before `Config()`
  instantiation; this is the only place env vars are loaded
- Python 3.11+, `.venv` virtualenv at `.venv/`

## CRITICAL: API key handling

**Never ask for the `ANTHROPIC_API_KEY`. Never print or log it. Never export it into the shell. Never hardcode it.**
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
  home.py                 # Workbench canvas — layout + session callbacks; every skill is a
                          # dbc.Accordion section. Route: /, sidebar link: "Workbench"
  data_quality.py         # Data Quality standalone page (/data-quality)
  eda.py                  # EDA standalone page (/eda)
  join_auditor.py         # Join Auditor standalone page (/join-auditor)
  nl_to_spark.py          # NL → PySpark standalone page (/nl-to-spark)
  mlops_review.py         # MLOps Review standalone page (/mlops-review)
  sessions.py             # Team view of every Workbench/notebook session (/sessions)
  agent_sim.py            # Autonomous agent simulation UI (/agent-sim)
  feedback.py             # Feedback loop dashboard (/feedback) — pending/drafted/exported/stale
  benchmarks.py           # Ground-truth suite + simulation scores (/benchmarks)
  costs.py                # Estimated spend by skill, user, model (/costs)

ds_workbench/              # Notebook import surface: `from ds_workbench import <skill>` (loaded lazily)
  core/                   # Deterministic functions, no API key/config/SDK (a test enforces it):
    leakage.py            #   leakage_profile(df, target) — AUC, null/zero patterns, single-class values
    joins.py              #   join_stats(left, right, keys, how) — exact rows, fan-out, data loss, verdict
    causal_stats.py       #   SRM, balance, IPW, CUPED, DiD, event study, pre-trend, E-value
    readiness.py          #   assess(results) — the one readiness policy (see project rule 6)

skills/                   # Pure Python — no Dash dependency. Called by pages, agents,
                          # AND importable directly from Databricks notebooks.
  data_quality.py         # assess(df, table_name) → findings, computed score, profile
  eda.py                  # analyze(df, table_name) → insights, charts, targets
  join_auditor.py         # audit(left_df, right_df, keys) → verdict (from core.joins), findings
  nl_to_spark.py          # generate(question, tables) → sql, pyspark, explanation
  mlops_reviewer.py       # review(code) → feasibility, findings, score
  model_selector.py       # select(df, target_col) → decision, task_type, recommendations
  leakage_detector.py     # check(df, target_col) → verdict, findings, drop_columns
  feature_engineering.py  # suggest(df, target_col) → plan, pyspark_code, model_family, imbalance
  experiment_tracker.py   # interpret(metrics) → verdict, findings, next steps
  causal_inference.py     # analyze(df, treatment, outcome) / did(...) → verdict, estimate, findings
  api_retry.py            # shared client, api_call/check_response, strict_tool, tool_input, call_context
  models.py               # role() → model + effort per role; override_roles() for sweeps
  usage.py                # usage ledger (every call: user, skill, run, tokens, cost); Delta export
  team_sessions.py        # persistent sessions + verdict/gates/blockers summary for /sessions
  session.py              # WorkflowSession (saves to team_sessions) and build_digest

benchmarks/               # Skill quality testing infrastructure
  answer_keys/            # Ground truth expected outputs per dataset per skill (JSON)
  scoring.py              # Compare skill output vs answer key → precision/recall metrics
  runner.py               # Run full benchmark suite, write results to benchmark_history.json
  model_compare.py        # Model/effort sweeps on the real skills (--arms, --efforts, --trials)
  history.py              # Suite history summary for the Benchmarks page
  benchmark_history.json  # Time-series of benchmark scores (gitignored)

agents/
  simulation.py           # Personas, scenarios, think/interpret/debrief/reflect steps
  scheduler.py            # Recommendation store (stale/revive), scheduler, edit drafts → patches

workbench/                # Workbench page internals (imported by pages/home.py)
  renderers/<skill>.py    # result dict → Dash components (dq, eda, ja, nl, ml, et, ms, ld, fe, ci);
                          # wrapped in safe_render so a stale stored run can't break a section
  callbacks/<skill>.py    # run/render callbacks per skill section; importing the package registers them
  session_panels.py       # readiness strip, pipeline health, guided workflow, synthesis, timeline
  constants.py            # dataset/join options, SKILL_META, colour maps

components/
  sidebar.py              # Sidebar: Workbench, Sessions, Feedback, Benchmarks, Agent Sim, Costs
                          # SKILLS list preserved for app.py SKILL_KEYS (run-count badges)
  run_card.py             # Reusable run history card component
  dq_gate.py              # find_dq_run(), dq_gate_banner() — DQ status in skill pages
  session_context.py      # session_banner() — active session strip in skill pages

assets/style.css          # Global dark-theme CSS overrides
feedback/                 # Persisted local data (gitignored)
  recommendations.json    # simulation suggestions (pending/drafted/implemented/stale/dismissed)
  sim_history.json
  sessions.json           # team sessions
  usage.jsonl             # usage ledger
  patches/                # exported feedback drafts
runs/                     # live-check outputs (gitignored)
```

## Workbench canvas (home.py)

`home.py` holds the layout and session-level callbacks; each section's renderer and callbacks
live in `workbench/renderers/<key>.py` and `workbench/callbacks/<key>.py`. All skills are
`dbc.Accordion(always_open=True)` sections:
- **dq** — Data Quality (wb-dq-*)
- **eda** — EDA (wb-eda-*)
- **ms** — Model Selector (wb-ms-*)
- **ld** — Leakage Detector (wb-ld-*)
- **fe** — Feature Engineering (wb-fe-*)
- **ja** — Join Auditor (wb-ja-*)
- **nl** — NL → PySpark (wb-nl-*)
- **ml** — MLOps Review (wb-ml-*)
- **et** — Experiment Tracker (wb-et-*)
- **ci** — Causal Inference (wb-ci-*): A/B, observational, or difference-in-differences

Session setup card at the top: problem statement + dataset checklist → "Start Session" button.
Each session gets an id and is saved to the team session store whenever a run is added
(`workbench/callbacks/session_sync.py`), so it appears on /sessions.
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
| `experiment_tracker` | Experiment Tracker | `skills/experiment_tracker.interpret()` | ✓ |
| `causal_inference` | Causal Inference (A/B + observational) | `skills/causal_inference.analyze()` | ✓ |

The first 8 skills plus `experiment_tracker` (training metrics come from the notebook) and
`preflight` are importable via `from ds_workbench import <skill>` (see `ds_workbench/__init__.py`).
`notebook_generator` is workbench-only (it consumes the session store). Every notebook skill
returns `{"error": ...}` instead of raising (tested).

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
See `ROADMAP.md`: causal inference has shipped (A/B, observational, DiD); next are
recommenders, then optimisation, plus a real-data pilot and benchmark cases for the untested
archetypes. Keep it updated as skills land.

## Benchmark suite architecture

Ground truth answer keys live in `benchmarks/answer_keys/<dataset>/<skill>.json`. Each key
specifies the expected findings, scores, and recommendations for a given (dataset, skill) pair.

Benchmark cases (30, all synthetic; `.venv/bin/python benchmarks/runner.py --list`):

| Dataset | Skills | What it tests |
|---|---|---|
| `orders_dirty` | DQ | nulls, mixed-case categories |
| `sales_clean` | DQ | precision: clean data, no false alarms |
| `hr_attrition` | DQ, FE, leakage, model | 4.7:1 imbalance; leakage precision case |
| `ecommerce` | DQ, FE, model | conversion baseline |
| `transactions_fraud` | DQ, FE, leakage, model | 32:1 imbalance, high-cardinality IDs, split-before-fit; leakage precision (amount) |
| `sales_forecasting` | DQ, model | regression / time series |
| `customer_accounts` | DQ | hidden issues: '' / 'N/A' placeholders, sentinels, future dates, padded codes, units |
| `ecommerce_leakage` | leakage | two planted leaks |
| `subscription_churn` | leakage | disguised leaks: neutral-named status code, non-linear index (r=0.11, AUC=1.0), zero-for-one-class credit |
| `renewal_timing` | leakage ×2 | same data and column notes, two prediction points: `support_tickets_mtd` leaks scoring on the 1st, legitimate on the 15th |
| `clinical_small_wide` | model | n/p ≈ 3.4: regularised model + CV, not boosting + holdout |
| `experiment_metrics` (×4) | experiment tracker | leakage, overfitting, imbalance, healthy (no false alarms) |
| `ab_test_clean`, `ab_test_srm`, `ab_test_bad_control`, `ab_test_underpowered` | causal | clean, sample-ratio mismatch, post-treatment control, underpowered — known true effects |
| `observational_coupon` | causal | confounded targeting; naive gap ~9× the truth |
| `did_store_rollout`, `did_pretrend` | causal (DiD) | clean rollout; pre-trend trap (true effect 0) |

Answer keys must never appear in skill prompts (`tests/test_benchmark_integrity.py`).

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
pending → drafted → patch exported → dismissed tabs. Drafting asks Claude for exact
search/replace edits to one target file (`agents.scheduler.target_file`: the skill module for
findings, `skills/result.py` for notebook output, `workbench/renderers/<skill>.py` for a
skill's Workbench UX, `pages/home.py` otherwise); each edit
must match once and the result must compile. "Export Patch" writes `feedback/patches/<id>.patch`
for a feature branch — the app never rewrites its own files.

**Button callbacks check the click**: Dash Pages can fire a button callback when its page
mounts even with `prevent_initial_call=True` (every Workbench load used to run DQ, EDA and a join
audit — paid API calls). Every callback with an `n_clicks` input starts with
`if not n_clicks: raise PreventUpdate` (lists for `ALL` patterns). `tests/test_static.py`
calls each one without a click and fails if it does anything.

**Import skills at startup**: callbacks import skills lazily and Dash serves callbacks on
several threads; concurrent first imports deadlocked on module locks. `app.py` preloads every
skill and `ds_workbench.core` module before the scheduler or server start, and
`ds_workbench/core/__init__` never imports its own submodules eagerly (it uses `__getattr__`).

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

Three personas run autonomous simulations against six scenarios. Each persona gets up to
`MAX_SKILL_STEPS = 2` skill runs, then gives a final debrief. After all personas complete,
each runs `reflect_step` to produce 2-3 structured improvement suggestions.

**Personas**
- **Alex Chen** — Lead Data Scientist; methodical, grounds everything in data quality first
- **Priya Sharma** — ML Engineer; production-minded, owns reproducibility and pipeline integrity
- **Jordan Kim** — Data Analyst; business-first, translates findings to executive language

**Scenarios** (6 total, each with `focus_skills` that bias Claude's skill selection)
- `q3_revenue` — Revenue shortfall; focus: data_quality, join_auditor, nl_to_spark, mlops_review
- `churn_risk` — Retention drop; focus: eda, model_selector, leakage_detector, data_quality
- `model_readiness` — Pre-flight model check; focus: leakage_detector, feature_engineering, model_selector, data_quality
- `pipeline_handoff` — New pipeline acceptance; focus: data_quality, join_auditor, mlops_review, nl_to_spark
- `fraud_detection` — Fraud model pre-flight; focus: leakage_detector, model_selector, feature_engineering, data_quality
- `launch_impact` — Did the launch work? Two A/B runs (one broken logging pipeline), an onboarding test with a post-treatment metric, an un-randomised coupon campaign; focus: causal_inference, data_quality

**Coverage-weighted scheduler** (`agents/scheduler.py`): instead of round-robin, `_coverage_weighted_scenario()`
counts benchmark runs per skill and picks whichever scenario's `focus_skills` have the fewest runs.
This prevents DQ/MLOps from dominating while model_selector/leakage_detector stay underrepresented.
Falls back to round-robin if benchmark data is unavailable. Interval: `SIM_INTERVAL_HOURS` (default 0 = manual).

## Local dev

```bash
# Install deps
.venv/bin/pip install -r requirements.txt

# Run app
.venv/bin/python app.py          # port 8050; DEBUG=false in .env turns off dev tools (demos)
# or
LOCAL_DEV_MODE=true .venv/bin/python app.py

# Offline test suite (no API calls; run before every PR)
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest

# Live smoke run of 5 skills (calls the API, prints output)
.venv/bin/python test_skills.py

# Run benchmark suite
.venv/bin/python benchmarks/runner.py

# Compare models/effort on the real skills (scores, spread, $/run)
.venv/bin/python benchmarks/model_compare.py --filter leakage --arms claude-opus-5-5:medium claude-sonnet-5-5:high --trials 3
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
