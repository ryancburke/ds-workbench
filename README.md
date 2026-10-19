# DS Workbench

AI-assisted data science skills that run inside your notebook, plus a team dashboard
that tracks how good those skills are.

- **Skills library** (`skills/`, imported as `ds_workbench`): data quality, EDA, leakage
  detection, feature engineering, model selection, join auditing, NL → PySpark, MLOps
  review, experiment interpretation. Each call returns a result that renders as a
  scannable report in a notebook cell and is a plain `dict` everywhere else.
- **Dash app** (`app.py`, `pages/`): a Workbench page that runs the skills on a session,
  a Benchmarks page with answer-key scores over time, and agent simulations whose
  critiques feed a recommendation backlog.

Skills are built to speak up at the right moment in a data scientist's workflow: bad data
before modelling, target leakage before training. They lead with the verdict and then
give the evidence.

Where the tool stands and what's next — causal inference, recommenders, optimisation — is in
[ROADMAP.md](ROADMAP.md).

## Quick start

Requires Python 3.11+ and an Anthropic API key.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then set ANTHROPIC_API_KEY in .env
.venv/bin/python app.py     # http://localhost:8050
```

With `LOCAL_DEV_MODE=true` (the default) the app uses built-in synthetic datasets, so it
runs without Databricks. Set it to `false` and fill in the `DATABRICKS_*` values to read
real tables.

## Using the skills in a notebook

The skills take pandas DataFrames. Put the repo on `sys.path` (for example as a
Databricks Repo) and make sure the API key is available to `config.py` through a `.env`
file or an environment variable.

```python
from ds_workbench import WorkflowSession, data_quality, leakage_detector, feature_engineering, model_selector, preflight

df = spark.table("main.crm.customers").limit(200_000).toPandas()

session = WorkflowSession(context="Predict which customers churn next quarter")
dq = data_quality.assess(df, table_name="customers", session=session)        # issues ranked by severity
ld = leakage_detector.check(df, target_col="churned", session=session)        # leaks to drop before training
fe = feature_engineering.suggest(df, target_col="churned", session=session)   # plan + runnable PySpark MLlib pipeline
ms = model_selector.select(df, target_col="churned", session=session)         # task type, assumptions, ranked models
preflight.check(session=session)                                               # go / caution / blocked, no API call

X = df.drop(columns=ld["drop_columns"])                                        # columns flagged as critical leakage
```

With a `WorkflowSession`, later skills pick up earlier findings automatically: feature
engineering drops the leakage columns, and model selection factors in data quality.
Without a session, pass the earlier results yourself, for example
`feature_engineering.suggest(df, "churned", leakage_findings=ld)`.

| Skill | Call | Returns |
|---|---|---|
| Data quality | `data_quality.assess(df, table_name)` | findings by severity, score, profile |
| EDA | `eda.analyze(df, table_name, context=)` | insights, recommended charts, candidate targets |
| Leakage | `leakage_detector.check(df, target_col)` | verdict, findings, `drop_columns` |
| Feature engineering | `feature_engineering.suggest(df, target_col)` | per-column plan, `pyspark_code`, `imbalance_handling` |
| Model selection | `model_selector.select(df, target_col)` | task type, assumption checks, ranked models with starter code |
| Join auditor | `join_auditor.audit(left, right, join_keys)` | verdict, fan-out and data-loss findings |
| NL → PySpark | `nl_to_spark.generate(question, {"name": df})` | Spark SQL and DataFrame code |
| MLOps review | `mlops_reviewer.review(code)` | feasibility verdict, findings with fixes |
| Experiment tracker | `experiment_tracker.interpret(metrics)` | verdict, red flags, next experiments |
| Causal inference | `causal_inference.analyze(df, treatment_col, outcome_col, randomized=)` | verdict (valid / caution / not identified), effect with CI, threats such as sample-ratio mismatch, bad controls, confounding |

A failed call returns `{"error": "..."}` with the reason (a refusal, a reply cut off at
`max_tokens`, or a missing tool call) rather than raising.

## Benchmarks: is a skill actually good?

Ground-truth cases live in `benchmarks/answer_keys/<dataset>/<benchmark>.json`. Each one
uses a synthetic dataset with planted issues, such as disguised leakage, high-cardinality
IDs or a 32:1 class imbalance. The cases also include precision cases, where a strong
legitimate feature must *not* be flagged.

```bash
.venv/bin/python benchmarks/runner.py                    # full suite (~$1–2, ~10 min)
.venv/bin/python benchmarks/runner.py --filter leakage   # a subset
.venv/bin/python benchmarks/runner.py --score-only       # re-score cached outputs, no API calls
.venv/bin/python benchmarks/report.py                    # latest results in the terminal

# Compare models or effort levels on the real skills: mean, spread, $/run
.venv/bin/python benchmarks/model_compare.py --filter leakage \
    --arms claude-opus-5-5:medium claude-sonnet-5-5:high --trials 3
```

Results also appear on the app's **Benchmarks** page. Run the suite before shipping a
skill change. Prompts must never contain a benchmark's planted answers; a test enforces
this.

## Agent simulations and the feedback loop

Three personas (a lead data scientist, an ML engineer and an analyst) work through a
business scenario with the skills. They rate each output and finish with improvement
suggestions. Run simulations manually from the **Feedback** page or with
`.venv/bin/python run_sim.py [scenario_id]` (about $0.60 each). Set `SIM_INTERVAL_HOURS`
to schedule them instead.

On the Feedback page, **Approve & Draft** asks Claude for a small, checked edit to the
right file. **Export Patch** writes it to `feedback/patches/` so you can apply it on a
branch. The app never edits its own source.

## Configuration

Everything is read from `.env` by `config.py`; see `.env.example`.

| Setting | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | API access |
| `LOCAL_DEV_MODE` | `true` | synthetic data instead of Databricks |
| `SIM_INTERVAL_HOURS` | `0` (manual) | scheduled simulations |
| `REASONING_MODEL` / `_EFFORT` | Sonnet 5.5 / high | data quality |
| `LEAKAGE_MODEL` / `_EFFORT` | Opus 5.5 / medium | leakage detection |
| `ANALYSIS_MODEL` / `_EFFORT` | Sonnet 5.5 / high | model selection, join auditing |
| `GENERAL_MODEL` / `_EFFORT` | Sonnet 5.5 / high | EDA, FE, NL → PySpark, MLOps, tracker |
| `STRUCTURED_MODEL` | Haiku 5.5 | structured extraction (second pass) |
| `SIM_MODEL`, `SIM_JUDGE_MODEL`, `CODE_MODEL` | Sonnet / Opus / Sonnet 5.5 | simulation and drafting |

Model IDs are never hardcoded outside `config.py`; skills ask for a role via
`skills.models.role()`.

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest          # offline suite (~1s, no API calls)
```

- Work on a feature branch and open a pull request; don't push to `main`.
- Always use the virtualenv (`.venv/bin/python`); the shell doesn't activate it.
- Live runs cost money; keep their outputs in the git-ignored `runs/` folder.
- `CLAUDE.md` has the architecture notes and the settled design decisions.

```
skills/            skill implementations (pure Python, no Dash), notebook rendering (result.py)
ds_workbench/      the notebook import surface
pages/             Dash pages; pages/home.py is the Workbench
workbench/         Workbench renderers, callbacks and session panels
agents/            simulation personas and the scheduler/feedback loop
benchmarks/        runner, scorers, answer keys, model comparison
tests/             offline pytest suite
```
