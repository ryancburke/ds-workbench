# Roadmap

Where DS Workbench is today, and where it's going. Update this file as skills land.

## Where we are (October 2026)

The tool is a strong assistant for **supervised machine learning on tables**: it checks data
quality, catches target leakage, plans features, picks models and reviews training results,
before the expensive mistakes happen. Every skill is benchmarked against answer keys and
critiqued by agent simulations.

| Archetype | Status | Evidence |
|---|---|---|
| Binary classification (churn, fraud, attrition, conversion) | **Covered, tested** | 5 datasets; leakage, FE and model-selection cases, including disguised leakage and precision cases |
| Data quality, joins, EDA on tables | **Covered, tested** | 7 DQ cases incl. hidden missing values, sentinels, future dates; join and EDA lightly tested |
| Training-result review | **Covered, tested** | 4 experiment-tracker cases incl. a no-false-alarm case |
| Regression | Weakly tested | one case (sales forecasting) |
| Time-series forecasting, multiclass, clustering, anomaly detection | Advertised by the model selector, **untested** | no benchmark cases |
| Causal inference and A/B tests | **In progress** | see Phase 1 |
| Recommendation and ranking | Not covered | Phase 2 |
| Optimisation (pricing, scheduling, allocation) | Not covered | Phase 3 |
| Text, images, geospatial | Not covered | not planned |

Known limits:
- **All benchmark data is synthetic.** The suite measures whether skills find problems we
  planted, not problems in your tables.
- **Most cases score 100 for every model tried**, so the suite catches regressions but can't
  yet separate good models from great ones (issue #4).

## Where we're going

### Phase 1 — Causal inference (in progress)
A `causal_inference` skill for randomised experiments (A/B tests) and observational data. It
leads with a verdict (VALID / CAUTION / NOT IDENTIFIED), then the estimate.
- **Computed in code:** sample-ratio mismatch, covariate balance and propensity overlap, power
  and minimum detectable effect, effect estimates with confidence intervals (difference in
  means, regression adjustment, CUPED, inverse propensity weighting).
- **The model's job:** choose the identification strategy, critique assumptions, and spot bad
  controls (covariates measured after treatment, the causal cousin of target leakage).
- **Benchmarks:** synthetic data with a known true effect, so estimates are graded exactly:
  sample-ratio mismatch, confounding, a bad control, an underpowered test, a clean test.
- Later: difference-in-differences with pre-period trend checks, a Workbench section and a
  simulation scenario.

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

### Cross-cutting
- **Real-data pilot:** run the notebook flow on one or two real Databricks tables with a data
  scientist on a live question. The highest-value next step for trust in the tool.
- **Probe the untested archetypes:** one benchmark case each for forecasting (random-split
  leakage on time-ordered data), multiclass, clustering and anomaly detection.
- **Discriminating benchmarks:** cases where informed models disagree, e.g. ambiguous leakage
  timing or quality issues that need domain knowledge.
- **Cleanup:** remove the standalone pages' copies of the Workbench renderers.

### Open questions that shape the designs
1. Causal: mostly A/B tests, observational studies, or both? (Phase 1 handles both.)
2. Recommenders: implicit (clicks, views) or explicit (ratings) feedback, and at what scale?
3. Optimisation: which problems (scheduling, pricing, allocation, routing), and is there a
   solver already in use?
