# Roadmap

Where DS Workbench is today, and where it's going. Update this file as skills land.

## Where we are (21 October 2026)

The tool is a strong assistant for **supervised machine learning and causal analysis on
tables**: it checks data quality, catches target leakage, plans features, picks models, reads
experiments and observational data, and reviews training results, before the expensive
mistakes happen. Every skill is benchmarked against answer keys and critiqued by agent
simulations.

It is now becoming a **decision-support workflow** (the current priority, below): from a
business question to framing, data discovery, verification, a planned analysis, and a
decision summary that says when the evidence isn't enough.

| Archetype | Status | Evidence |
|---|---|---|
| Binary classification (churn, fraud, attrition, conversion) | **Covered, tested** | 5 datasets; leakage, FE and model-selection cases, including disguised leakage and precision cases |
| Data quality, joins, EDA on tables | **Covered, tested** | 7 DQ cases incl. hidden missing values, sentinels, future dates; join and EDA lightly tested |
| Training-result review | **Covered, tested** | 4 experiment-tracker cases incl. a no-false-alarm case |
| Regression | Weakly tested | one case (sales forecasting) |
| Time-series forecasting, multiclass, clustering, anomaly detection | Advertised by the model selector, **untested** | no benchmark cases |
| Question → framing → data discovery → verification → plan | **Built (M0–M4), tested** | 12 framing cases, 5 scout cases with a keyword-search baseline, deterministic verification and planner tests; live end-to-end runs on a synthetic catalog |
| Causal inference: A/B tests, observational, difference-in-differences | **Covered, tested** | 7 cases with known true effects (SRM, bad control, confounding, underpowered, clean, DiD rollout, DiD pre-trend trap); Workbench section; simulation scenario |
| Recommendation and ranking | Not covered | Phase 2 |
| Optimisation (pricing, scheduling, allocation) | Not covered | Phase 3 |
| Text, images, geospatial | Not covered | not planned |

Known limits:
- **All benchmark data is synthetic.** The suite measures whether skills find problems we
  planted, not problems in your tables. The Data Scout's synthetic cases prove its guardrails,
  not its usefulness on a real catalog: that is what the pilot is for.
- **Most cases score 100 for every model tried**, so the suite catches regressions but can't
  yet separate good models from great ones (issue #4).

## Where we're going

### Current priority — from question to decision
An investigation workflow on top of the existing skills, so a data scientist gets help with
*which* data and *which* analysis, and an honest answer about whether the evidence supports
the decision. Architecture (agreed 21 Oct 2026): a **deterministic controller** with
structured model calls at the judgment points; the only agent loop is the Data Scout's
catalog search, bounded and read-only. Code enforces the rules; people make the calls that
carry responsibility. Notebook first; the Workbench and Sessions pages are the view.

| Milestone | What | Status |
|---|---|---|
| M0 | Approved-model routing (`ANTHROPIC_PROVIDER`), declared required checks, failed checks never dropped, quoted SQL names | shipped (#38) |
| M1 | **Problem Framer**: question → ProblemSpec (decision vs questions, typed, assumptions, gaps, answerability, assignment) | shipped (#39) |
| M2 | **Data Scout**: catalog interface, synthetic catalog, metadata-only shortlist; no invented tables, columns or joins | shipped (#40) |
| M3 | **Verification**: the tables a person selects, checked as aggregates (freshness, fill, join match and cardinality, causal timing) | shipped (#41) |
| M4 | **Planner**: steps per question type, method choice, analysis-table recipe and contract, `run_next` | shipped (#42) |
| M5 | **Decision synthesis**: decision readiness computed in code (separate from analytical readiness), a cited narrative, optional cost/ROI | shipped (#44) |
| M6 | **Unity Catalog adapter + pilot harness**: the catalog interface over Unity Catalog (read-only, allow-listed, service principal), contract-tested against the synthetic catalog; blinded Scout-vs-search pilot scoring | shipped (#45) |

Decisions taken: an organisation-approved Anthropic model (the client route is a setting);
pilot access through a read-only service principal on an allow-listed sandbox first, moving
to on-behalf-of-user before wider rollout; notebook first; cost and ROI inputs optional;
lineage optional; confirming the spec recommended, not required (`REQUIRE_CONFIRMED_SPEC`).

**Enterprise pilot.** The public repo stays organisation-agnostic: the adapter is generic, and
company vocabulary, allow-lists and caps live in private configuration. Evaluation: 15–25 real
past questions where data scientists know the right tables; compare the Scout with plain
catalog search for the same terms (recall@5, expert-rated relevance and suitability notes,
time to a shortlist, confident-but-wrong claims), rated blind.

**Next: run the pilot** in the approved sandbox (read-only service principal, allow-listed schemas, 15–25 real past questions). **Open for later:** lineage access in the pilot environment; when to require spec
confirmation; a Workbench view of investigations (and the confirm-availability button, #36).

### Phase 1 — Causal inference (shipped)
A `causal_inference` skill for randomised experiments (A/B tests) and observational data. It
leads with a verdict (VALID / CAUTION / NOT IDENTIFIED), then the estimate.
- **Computed in code:** sample-ratio mismatch, covariate balance and propensity overlap, power
  and minimum detectable effect, effect estimates with confidence intervals (difference in
  means, regression adjustment, CUPED, inverse propensity weighting).
- **The model's job:** choose the identification strategy, critique assumptions, and spot bad
  controls (covariates measured after treatment, the causal cousin of target leakage).
- **Benchmarks:** synthetic data with a known true effect, so estimates are graded exactly:
  sample-ratio mismatch, confounding, a bad control, an underpowered test, a clean test.
- **Shipped:** `causal_inference.analyze()` (A/B and observational, with an E-value for
  sensitivity to unmeasured confounding) and `causal_inference.did()` (difference-in-differences
  with an event study and pre-trend test); a Workbench section with an event-study chart; the
  "Did the Launch Work?" simulation scenario. 7 benchmark cases, all passing live.
- **Next:** reconcile repeated runs of the same experiment (asked for by the simulation judge),
  staggered rollouts (units adopting at different times), and doubly robust estimation.

### Phase 2 — Recommenders
A skill for interaction logs (user, item, time, rating or click). Computes sparsity, long-tail
share, cold-start share and a popularity baseline; recommends an approach (popularity → Spark
ALS → richer models) and a time-based evaluation protocol (recall@k, NDCG@k, coverage). The
trap it guards against: random splits that leak future interactions into training.

### Phase 3 — Optimisation
A skill that turns a problem description and data tables into a structured formulation
(variables, objective, constraints) that **our code** solves with an open-source solver
(HiGHS via SciPy), then reports feasibility, binding constraints and sensitivity. The model
never runs code. Benchmarks use small problems with known optima. Links to the other skills
through "predict, then optimise".

### Cost visibility
- **Shipped:** a usage ledger for every call (user, skill, run, model, tokens, latency, outcome,
  estimated cost; no content), a Costs page (by day, skill, user, model; cost per successful
  run; spend on failures), and a Delta export for the platform team.
- **Next:** reconcile the estimates with the Anthropic Usage & Cost Admin API; budgets and
  alerts per team or skill; write directly to Delta from the deployed app.

### Cross-cutting
- **Real-data pilot:** run the notebook flow on one or two real Databricks tables with a data
  scientist on a live question. The highest-value next step for trust in the tool.
- **Probe the untested archetypes:** one benchmark case each for forecasting (random-split
  leakage on time-ordered data), multiclass, clustering and anomaly detection.
- **Discriminating benchmarks:** cases where informed models disagree, e.g. ambiguous leakage
  timing or quality issues that need domain knowledge.
- **Cleanup:** remove the standalone pages' copies of the Workbench renderers.

### Open questions that shape the designs
1. Causal: answered — both randomised tests and non-randomised launches.
2. Recommenders: implicit (clicks, views) or explicit (ratings) feedback, and at what scale?
3. Optimisation: which problems (scheduling, pricing, allocation, routing), and is there a
   solver already in use?
