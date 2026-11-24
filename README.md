# ExpertRuleFit

## Prior work and attribution

**Expert RuleFit already exists, and the core idea of this library is theirs.** *Expert RuleFit* (ERF) was introduced by **Luisa Ebner** in her MSc thesis *Expert RuleFit – Complementing Rule Ensembles with Expert Knowledge*, with an R implementation at [github.com/LuisaEbner/ERF](https://github.com/LuisaEbner/ERF). It extends RuleFit (Friedman & Popescu 2008) with expert knowledge in two groups:

- **confirmatory** expert rules and linear terms, exempt from penalisation, so they always enter the final model;
- **optional** expert rules and linear terms, with an adjustable penalty factor (adaptive regularisation).

The *confirmatory / optional* terminology, the idea of exempting confirmatory rules from the penalty, and the adaptive penalty for optional rules all come from that work, and this library uses the same name; whether to rename it is still open.

**What this library adds** to Expert RuleFit:

1. **Determinism by design**: internal seeds fixed and the user seed ignored, canonical row and column order (rows since 0.8.0, columns by name since 0.8.1: any permutation of the training rows or of the columns gives the same model), a pinned container recipe, and an explicit execution envelope.
2. **A sign-constrained confirmatory block** (`confirmatory_mode="constrained"`, the default since 0.8.0): one convex problem with confirmatory coefficients unpenalised and bounded by the expert's expected sign. When the data contradict the sign, the coefficient is held at zero and flagged.
3. **Diagnostics and fingerprints**: sign checks, absorption diagnostics (correlated rules a confirmatory rule displaces), and SHA-256 fingerprints of rules, coefficients and predictions in an audit bundle.
4. **A conditioning analysis** of penalty weighting by column rescaling: very small weights make saga stop far from the optimum without a warning; standardisation and the weight choice that avoid it.
5. Also: an optional bootstrap/subsample selection-frequency filter, data-driven interaction rules from EBM (optional, never confirmatory by default), and a stacked EBM + rules model.

**Reproducible interpretable rule ensembles for regulated environments, building on Expert RuleFit (Ebner, MSc thesis; R package ERF).**

ExpertRuleFit is a Python implementation of [RuleFit](https://arxiv.org/abs/0811.1679) (Friedman & Popescu 2008) with Expert RuleFit's confirmatory and optional expert rules, made **deterministic** (same rules and predictions whatever the seed or row order, within the execution envelope below, which includes the CPU model) and with a **sign-constrained confirmatory block**. Integrates with [InterpretML](https://github.com/interpretml/interpret) Explainable Boosting Machines (EBM) to auto-discover interaction rules grounded in shape functions. Built for banking compliance under CSSF/EU AI Act.

## The Problem

Standard RuleFit uses Lasso (L1) for rule selection, which is **inherently unstable**: different random seeds produce different rules, even on the same data. In regulated banking (CSSF, EU AI Act, BCBS 239), this makes the model **non-deployable** — a regulator who re-executes the model must get identical output.

## The Solution

The pipeline (defaults of version 0.8.1):

0. **Canonical column and row order** — columns are sorted by feature name (`column_order="canonical"`, since 0.8.1), then training rows are sorted, so any permutation of the same rows or columns gives the same model (`row_order="canonical"`).
1. **Fixed-seed rule generation** — a gradient-boosting ensemble with internal seed 42 (user `random_state` ignored). By default the candidates are the rules kept by imodels' RuleFit, **which runs its own L1 (Lasso) path and returns at most `max_rules` rules before ExpertRuleFit or the expert rules are involved**. `candidates="trees"` takes every distinct rule from the same trees instead, so selection happens only in the steps below.
2. **Expert rule augmentation** — confirmatory and optional expert rules, as in Expert RuleFit (Ebner); optional rules get a reduced penalty.
3. **Selection-frequency filter** — 10 elastic-net logistic fits on stratified half-samples drawn without replacement in complementary pairs (`resampling="subsample"`; the n-out-of-n bootstrap of earlier versions is `resampling="bootstrap"`); columns selected in ≥ 80 % of the fits are kept, confirmatory rules always. **Off by default since 0.8.0** (`selection="none"`): in the B1 study it made the model less stable (see Validation Results); `selection="bootstrap"` turns it on.
4. **Final elastic-net CV** on the kept columns (standardised; coefficients reported on the original scale).
5. **Confirmatory block** (`confirmatory_mode="constrained"`) — one convex problem: confirmatory coefficients unpenalised, bounded by their `expected_sign` when one is declared, free otherwise. A coefficient held at 0 by its bound is flagged (warning, audit bundle; `strict=True` raises). `confirmatory_mode="refit"` restores the earlier unpenalised fallback refit.

Guaranteed reproducible within a deterministic execution envelope: **pinned dependency versions, single-threaded BLAS/OpenMP, fixed random seeds, the same CPU model, and consistent platform/runtime settings ⇒ identical rule set and identical predictions**. Row and column order are canonicalised (any order of the same rows and columns gives the same model). On a *different* CPU model the rule set and AUC were identical in our tests, but the probabilities differed in their last bits (three CPU models, FINDINGS §20); pinning the BLAS kernel is planned (phase C).

> **Container deployment:** For a step-by-step guide to enforcing this guarantee via Docker (pinned digests, locked wheels, deterministic numerics, schema validation, auditable fingerprints), see **[docs/REPRODUCIBILITY_CONTAINERS.md](docs/REPRODUCIBILITY_CONTAINERS.md)**.

## Validation Results

> **Version note (24/11/2026):** the table and bullets below were produced with 0.8.0. Under 0.8.1 (columns sorted by name by default), Table 1 was rerun on GitHub Actions in the pinned image ([`experiments/results/table1_v081/`](experiments/results/table1_v081/)): ExpertRuleFit runs were identical in 3/3 runs on every dataset, with test AUC 0.798 (German), 0.763 (Taiwan) and 0.904 (HMEQ), against RuleFit means of 0.788, 0.764 and 0.903, so within 0.010 of RuleFit. Rule sets and AUC were also identical on two other CPU models, but the probability bytes were not: bitwise identity holds per machine (FINDINGS §20). B1 under 0.8.1 is running.

Seed sensitivity on **3 credit-scoring datasets** (one stratified 80/20 split, preprocessing fitted on the training split; pinned stack in [`docker/`](docker/): Python 3.11, scikit-learn 1.7.2, imodels 3.0.4). RuleFit is fitted with 100 seeds. ExpertRuleFit 0.8.0 (library defaults, no expert rules) ignores the seed by design, so it is run 3 times in separate processes and checked for bitwise-identical output (rule-set hash and SHA-256 of test probabilities).

| Dataset | RuleFit: runs with the most frequent rule set | ExpertRuleFit: identical runs | RuleFit AUC (mean ± sd over seeds) | ExpertRuleFit AUC |
|---------|:---:|:---:|:---:|:---:|
| German Credit (UCI) | 1/100 | **3/3** | 0.788 ± 0.009 | 0.794 |
| Taiwan Credit (UCI, 5,000-row subsample) | 1/100 | **3/3** | 0.764 ± 0.003 | 0.757 |
| HMEQ | 1/100 | **3/3** | 0.903 ± 0.003 | 0.902 |

- Standard RuleFit returned a different rule set for every seed (100 distinct sets per dataset); mean pairwise Jaccard 0.036–0.058.
- ExpertRuleFit returned the same rules and the same probabilities in every run, with test AUC within 0.007 of RuleFit's mean.
- This measures **determinism**, which holds by construction, not stability when the training data change.
- **Stability under perturbation (B1, German Credit, 25 stratified 90 % subsamples):** ExpertRuleFit is *not* more stable than RuleFit with a fixed seed (semantic rule similarity 0.527 vs 0.530; prediction SD 0.079 vs 0.056). The bootstrap selection-frequency filter made it less stable (0.414; 0.087) at ~8× the runtime, so since 0.8.0 it is off by default (`selection="none"`; `selection="bootstrap"` restores it). Results: [`experiments/results/b1/`](experiments/results/b1/).
- Reproduce: `python experiments/fetch_data.py`, then `experiments/table1.py` in the `docker/` image. Results: [`experiments/results/table1_v080/`](experiments/results/table1_v080/) (0.7.0 run: `table1/`). One ExpertRuleFit fit took 24 s (German) to 6.6 min (HMEQ) single-threaded.

## Quick Start

```python
from expertrulefit import ExpertRuleFit

# Fit (auto rules only — reproducible)
erf = ExpertRuleFit(max_rules=50)
erf.fit(X_train, y_train, feature_names=feature_names)

# Predict
proba = erf.predict_proba(X_test)[:, 1]

# Inspect the rules
for rule in erf.get_rule_importance()[:5]:
    print(f"coef={rule['coefficient']:+.4f} | {rule['name']}")

# Summary
erf.summary()
```

### With Confirmatory Rules (regulatory requirements)

```python
# Rules the expert requires in the model (unpenalised; optional expected_sign bound)
confirmatory = [
    {
        "name": "CSSF: High cash ratio",
        "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
    },
    {
        "name": "BCBS: Country risk threshold",
        "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 2.5).astype(float),
    },
]

erf = ExpertRuleFit(max_rules=50)
erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)

# Verify all regulatory rules were preserved
assert erf.confirmatory_all_active_, "COMPLIANCE FAILURE: confirmatory rule eliminated!"
erf.summary()  # Shows [ACTIVE] status for each confirmatory rule
```

**How it works:** Candidate columns are first standardised (`standardize=True`, statistics from the training data, stored for prediction; coefficients are reported on the original scale). Confirmatory rules then get a small penalty weight through feature scaling: we fit on `Z_j / sqrt(w_j)`, which reduces the effective elastic-net penalty on their coefficients (L2 scaled by `w_j`, L1 by `sqrt(w_j)`). The default is `w_c = 1e-2`. Much smaller values (the former default `1e-8`) make one column 10⁴ times larger than the others; the saga solver then stops far from the optimum without warning and can collapse the rest of the model (see [findings](docs/refinement/notes/FINDINGS_2026-11-22.md)). **Important:** A confirmatory coefficient can still be near/at zero if the rule carries no incremental signal, is perfectly collinear, or due to solver tolerances. ExpertRuleFit detects this and performs a post-hoc refit to enforce inclusion.

### Expected sign and absorption diagnostics

A confirmatory rule can declare the direction the expert expects (`+1`: the rule should raise P(y=1); `-1`: lower it). The sign is **checked, not enforced**: after `fit`, a contradiction sets `confirmatory_sign_ok_ = False` and emits a `UserWarning`, but the rule stays in the model.

```python
confirmatory = [
    {
        "name": "High cash ratio",
        "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
        "expected_sign": +1,
    }
]
erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)

erf.confirmatory_sign_ok_  # False if any declared sign is contradicted
erf.diagnose_confirmatory()  # per confirmatory rule (training data):
#   max_abs_corr / most_correlated  – strongest correlation with any other candidate column
#   vif                             – variance inflation factor (among retained columns)
#   displaced_rules                 – correlated candidates (|r| >= 0.5) inactive in the
#                                     final model (dropped by the bootstrap filter or
#                                     zeroed), with their bootstrap frequency
#   correlated_coef_shift           – change in those rules' coefficients when the
#                                     confirmatory rule is removed (ridge refits)
erf.diagnose_confirmatory(X_val, y_val, corr_threshold=0.7)  # recompute on other data
```

`RuleSpec(..., expected_sign=+1)` carries the sign through `to_rule_dict()` and JSON save/load. `summary()` and `audit_bundle()` include the sign status and the diagnostics. A near-unpenalised confirmatory rule that duplicates generated rules will take their weight; the diagnostics make that visible, they do not prevent it.

### Choosing `confirmatory_penalty` (w_c)

Preliminary sensitivity results (German Credit only, one split, pinned stack; `experiments/results/sensitivity/`):

- **w_c ≥ 1e-2 is the reliable range; 1e-2 is the default.** It gave no saga ConvergenceWarnings, test AUC 0.784–0.797, and fits 3× faster than at 1e-4. Below 1e-2, warnings appear (3–6 per fit at 1e-4, hundreds at 1e-8), and coefficients of confirmatory rules are not trustworthy.
- **Avoid 1e-8** (the pre-0.7 default). Even with standardisation it lowered AUC by about 0.03–0.04 and degraded the Brier score (0.229 vs about 0.16).
- **Expect sign contradictions** when a confirmatory rule duplicates terms the model already has. "Checking account < 0 DM" got a negative conditional coefficient because the checking-status linear term and generated rules carried the effect. Check `confirmatory_sign_ok_` and `diagnose_confirmatory()`.

After standardisation, solver conditioning depends on the 1/√w_c ratio rather than on the dataset; the HMEQ and Taiwan runs are confirmation.

### Sign-constrained confirmatory rules (`confirmatory_mode="constrained"`)

With `confirmatory_mode="refit"`, a confirmatory rule is kept in the model but its sign is only checked. With `confirmatory_mode="constrained"` (the default since 0.8.0), the final model is one convex problem: confirmatory coefficients are **unpenalised** and bounded by their `expected_sign` (≥ 0 for +1, ≤ 0 for −1), the other columns keep the elastic net chosen by cross-validation, and optional rules keep their penalty factor. A deterministic proximal-gradient solver (FISTA) solves it.

```python
erf = ExpertRuleFit(confirmatory_mode="constrained")  # or constrained_penalty="ridge"
erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)
erf.confirmatory_status_[0]["bound_active"]  # True: held at 0 because, given the other
# terms, the data push against the expert's sign
erf.constrained_info_  # C, l1_ratio, iterations, KKT residual, bounds
```

A bound-active rule is reported, not hidden: its coefficient is 0, so it is not *included* in the sense of the paper's Definition 1, and a warning names it. `constrained_penalty="ridge"` makes the problem strictly convex (unique solution) even when generated rules duplicate a confirmatory rule. Default since 0.8.0; `confirmatory_mode="refit"` keeps the earlier behaviour. `strict=True` turns the bound-active warning into a `ConfirmatoryBoundError`.

### Re-execution checks

- `erf.fingerprint(X_test)` returns SHA-256 hashes of the active rule set, of the intercept and coefficients, and of the predicted probabilities on `X_test`. `audit_bundle(X_reference=X_test)` includes them. Two runs are the same model if, and only if, the fingerprints match.
- **Row order.** Boosting subsamples, resampling draws and CV folds index rows by position, so with `row_order="as_given"` shuffling the training rows changes the model much as changing a seed would (see `experiments/results/row_order/`). The default since 0.8.0, `row_order="canonical"`, sorts the rows deterministically before fitting: any permutation of the same rows gives bitwise-identical output. Expert rules must therefore be row-wise functions of the features.
- **Column order (since 0.8.1).** The column order changed the fit as well (rule names, the row sort and the trees; FINDINGS §17), so `column_order="canonical"` (default) sorts the columns by feature name before anything else: any permutation of the columns, with `feature_names` permuted alike, gives bitwise-identical output (`tests/test_columns_calibration.py`, `experiments/results/hashcheck_v081/`). Names must be unique. Expert rules are still evaluated on X in the order given to `fit`, so `RuleSpec` rules and callables that look columns up by name work unchanged; pass X to `predict` in that same order. `feature_names_` is the canonical order, `input_feature_names_` the order given, `column_order_` the permutation; the audit bundle records all three and `fingerprint()` adds `schema_sha256` (column order mode and canonical names). `column_order="as_given"` restores 0.8.0 behaviour.

### With EBM-Discovered Interactions (automated pipeline)

```python
from expertrulefit import ExpertRuleFit, discover_interaction_rules

# EBM discovers which feature pairs interact, then finds optimal thresholds.
# Returned rules are OPTIONAL by default (rule_type="optional").
rules, ebm = discover_interaction_rules(X_train, y_train, feature_names, top_k=3)

# Data-discovered rules enter as optional rules (reduced penalty);
# they are not confirmatory unless an expert reviews them.
erf = ExpertRuleFit(max_rules=50)
erf.fit(
    X_train,
    y_train,
    feature_names=fn,
    confirmatory_rules=confirmatory,  # expert-imposed rules only
    optional_rules=rules,
)  # EBM-discovered rules
```

**Pipeline:** EBM (GA2M) identifies which feature pairs interact → quantile scan finds discriminative thresholds → ExpertRuleFit treats them as optional candidates.

> **Expert-imposed vs data-discovered.** Confirmatory status is reserved for rules an expert imposes. Rules found by EBM are optional unless a domain expert reviews and promotes them (pass `rule_type="confirmatory"`, or move the dict into `confirmatory_rules`). Promoting an unreviewed data-discovered rule would lock a possibly accidental correlation into the model. Each returned dict carries `"source": "ebm_discovered"` for audit trails.

### Dual Model — Full Stacking Architecture (§5 of spec)

```python
from expertrulefit import DualModel

# EBM (continuous effects) + ExpertRuleFit (discrete rules) → LogisticRegression stacking
dm = DualModel()
dm.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)

# Meta-classifier combines both scores with interpretable weights
proba = dm.predict_proba(X_test)[:, 1]
dm.summary()  # Shows: EBM weight=91%, ERF weight=9%, intercept

# Per-sample explanation: shape functions + active rules + meta-weights
explanations = dm.explain(X_test[:1])
```

> **Leakage.** Each stacking fold fits EBM and ExpertRuleFit on its training rows only (tested in `tests/test_leakage.py`). Do not pass rules from `discover_interaction_rules` run on the whole training set: their thresholds would have seen the out-of-fold rows. Pass the discovery step instead, and it is rerun inside each fold:
> `dm.fit(X_train, y_train, feature_names=fn, rule_discovery=lambda X, y, f: discover_interaction_rules(X, y, f)[0])`

**Architecture:** `Features → EBM(p_ebm) + ExpertRuleFit(p_erf) → LogisticRegression → final_score`. Every layer is interpretable — no post-hoc approximations. Cross-validated OOF predictions prevent stacking overfitting.

## Prediction & Classification

ExpertRuleFit is a **binary classifier** (classes {0, 1}). It outputs hard labels and logistic probabilities; post-hoc calibration is optional (`calibration="sigmoid"` or `"isotonic"`; in the default constrained mode only `"sigmoid"`).

### How Prediction Works

```
Training data
  → Fixed-seed GradientBoosting (candidate rules + linear features)
  → Augment with expert rules (confirmatory / optional, adaptive penalty weighting)
  → [optional, selection="bootstrap"] subsample × weighted LogisticRegressionCV, keep columns in ≥ 80 %
  → Final LogisticRegressionCV (chooses λ, α)
  → Constrained confirmatory block (unpenalised, sign-bounded; bound-active flagged)
  → Logistic sigmoid → P(y=1)  (optional post-hoc calibration)
```

### predict() — Hard Class Labels

```python
erf = ExpertRuleFit(max_rules=50)
erf.fit(X_train, y_train, feature_names=fn)

y_pred = erf.predict(X_test)  # array of {0, 1}
```

### predict_proba() — Probabilities

```python
proba = erf.predict_proba(X_test)  # shape (n_samples, 2)
p_default = proba[:, 1]  # P(y=1) — e.g., probability of default
```

Probabilities are logistic-regression outputs; their calibration has not yet been evaluated on the benchmark datasets. If calibrated scores matter (e.g. PD estimates), set `calibration="sigmoid"` or `"isotonic"` and check reliability on held-out data. **In the default `confirmatory_mode="constrained"` only `"sigmoid"` is allowed** (since 0.8.1): Platt scaling maps the block's log-odds η(x) to a·η(x) + b, with a and b fitted on out-of-fold log-odds (5 stratified folds, same penalty, weights and sign bounds) and a > 0, so the calibrated log-odds stay additive over the rules and every term keeps its sign; the fitted coefficients are unchanged and a, b are in `calibration_info_` and the audit bundle. `"isotonic"` is monotone but not affine, so it would break that additivity, and it raises `ValueError` in constrained mode (use `confirmatory_mode="refit"`). Each row sums to 1.0. Use `proba[:, 1]` for AUC, ranking, and threshold tuning.

### Custom Decision Threshold

The default threshold is 0.5. For cost-sensitive applications (e.g., credit scoring where false negatives are costly), adjust:

```python
threshold = 0.3  # more conservative — catches more positives
y_custom = (erf.predict_proba(X_test)[:, 1] >= threshold).astype(int)
```

### Inspecting What Drives Predictions

```python
# Global: top rules by |coefficient|
for r in erf.get_rule_importance()[:5]:
    print(f"coef={r['coefficient']:+.4f} [{r['category']}] {r['name']}")

# Selection frequency of each column (all 1.0 when selection="none")
print(erf.bootstrap_frequencies_)

# Full model summary (rules, confirmatory status, importance)
erf.summary()
```

### DualModel — Stacked Prediction with Explanations

```python
from expertrulefit import DualModel

dm = DualModel()
dm.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)

proba = dm.predict_proba(X_test)[:, 1]

# Per-sample explanation: EBM contributions + active rules + meta-weights
explanations = dm.explain(X_test[:1], top_n=5)
exp = explanations[0]
print(f"Final P(y=1): {exp['final_score']:.4f}")
print(f"EBM score:    {exp['ebm_score']:.4f}")
print(f"ERF score:    {exp['erf_score']:.4f}")
```

### Feature Matrix Structure

Internally, the final logistic regression operates on an augmented feature matrix:

```
[linear:feat_0, ..., linear:feat_n, rule_1, ..., rule_k, confirmatory:..., optional:...]
```

The prediction is: `P(y=1) = sigmoid(intercept + sum(coef_j * feature_j))`

- **Linear features**: one per input column (continuous contribution)
- **Rules**: binary {0, 1} from tree-extracted conditions (e.g., `X_3 > 0.4 and X_4 <= 0.15`)
- **Confirmatory**: expert-imposed rules, unpenalised; held at 0 and flagged if the data contradict their declared sign
- **Optional**: analyst rules with reduced penalty (may be eliminated)

> See `examples/prediction_basics.py`, `examples/classification_with_rules.py`, and `examples/prediction_inspection.py` for runnable demonstrations.

## Two Guarantees

1. **Reproducibility** — within the execution envelope (same set of rows, pinned stack, same CPU model; any column order since 0.8.1), same rules and same predictions whatever the seed or row order
2. **Confirmatory rules kept in the model** (mechanism from Expert RuleFit, Ebner) — in the default constrained mode, a rule whose declared sign the data contradict is held at zero and flagged rather than given the wrong sign

## Features

- **Seed-independent** — identical rules and predictions in every run (hash-checked on 3 datasets)
- **Confirmatory rules** (Expert RuleFit, Ebner) — unpenalised, with optional sign bounds; `strict=True` raises if a bound is active
- **EBM bridge** — discover interactions with InterpretML and add them as optional rules
- **Dual stacking** — EBM + ExpertRuleFit + LogisticRegression meta-classifier, per-sample explanations
- **Deterministic by design** — fixed internal seeds, single-threaded BLAS
- **Interpretable** — rule-based model, transparent by design (EU AI Act Art. 13)
- **Auditable** — SHA-256 fingerprints of rules, coefficients and predictions in an audit bundle
- **Built on proven foundations** — extends [imodels](https://github.com/csinva/imodels) (Cynthia Rudin, Duke)

## Installation

```bash
pip install imodels scikit-learn numpy
pip install interpret pandas  # optional, for EBM bridge & DualModel
```

Then clone this repo:

```bash
git clone https://github.com/hdjebar/ExpertRuleFit
cd ExpertRuleFit
```

## Run Validation Benchmark

The full 100-seed × 3-dataset benchmark (produces figures, CSVs, report):

```bash
python expertrulefit_validation.py
```

Output goes to `output/` with:
- `figures/` — 5 PNGs + combined PDF
- `data/` — 3 CSV result files
- `ExpertRuleFit_Validation_Report.md` — full markdown report

## Architecture

```
expertrulefit/
    __init__.py              # Package entry point
    expert_rulefit.py        # ExpertRuleFit class
    ebm_bridge.py            # EBM → confirmatory rule extraction
    dual_model.py            # DualModel: EBM + ExpertRuleFit stacking
expertrulefit_validation.py  # Full benchmark (3 datasets × 100 seeds)
examples/
    quick_demo.py                # Quick 10-seed reproducibility demo
    ebm_pipeline.py              # EBM → ExpertRuleFit pipeline demo
    dual_model_demo.py           # Full dual architecture demo
    prediction_basics.py         # predict() and predict_proba() usage
    classification_with_rules.py # Confirmatory + optional rule effects
    prediction_inspection.py     # Rule importance, per-sample explanation
tests/
    test_expert_rulefit.py   # ExpertRuleFit tests (9 tests)
    test_dual_model.py       # DualModel tests (4 tests)
docs/
    SOTA.md                  # State-of-the-art survey (2023--2026)
    REPRODUCIBILITY_CONTAINERS.md  # Container-based reproducibility guide
output/                      # Benchmark results (figures, CSVs, report)
```

## How It Works

### Standard RuleFit (unstable)
```
Random trees (seed-dependent) → Lasso → rule selection
Different seed → different trees → different rules
```

### ExpertRuleFit (deterministic, 0.8.1 defaults)
```
Sort columns by name, then rows (canonical order)
  → Fixed-seed GradientBoosting (BASE_SEED=42) → RuleFit's rules + linear features
  → augment with expert rules (adaptive penalty weighting)
  → final LogisticRegressionCV (elastic net)
  → constrained confirmatory block (FISTA, no randomness)
Same set of rows → same trees → same rules → same output
```

### What determinism does and does not give

Fixing every seed and the row order removes the run-to-run variation of RuleFit; it does not make the selected rules more stable when the training data change. On German Credit, ExpertRuleFit was not more stable under subsampling than RuleFit with a fixed seed (B1).

## Regulatory Compliance

| Requirement | Standard | How ExpertRuleFit can support it (no compliance claim) |
|-------------|----------|---------------------------|
| EU AI Act Art. 9 | Risk management | Re-running the model gives the same risk assessment (within the envelope) |
| EU AI Act Art. 12 | Automatic logging | Same model produces identical results at each audit |
| EU AI Act Art. 13 | Transparency | Seed-independent rules can be documented and explained consistently |
| BCBS 239 Principle 3 | Accuracy & integrity | Verifiable results across re-executions |
| CSSF Circular 12/552 | Model governance | Regulator can re-execute and obtain identical output |

## State of the Art

For a comprehensive survey of recent research (2023--2026) on RuleFit extensions, competing rule-based methods, stability techniques, Rashomon sets, regulatory AI, and how ExpertRuleFit is positioned relative to the field, see **[docs/SOTA.md](docs/SOTA.md)**.

**Key findings:**
- **Direct precedent:** Expert RuleFit (Ebner) introduced confirmatory/optional expert rules in RuleFit (see *Prior work and attribution* above)
- **Related:** SIRUS (stability), FIRE/CRE (sparsity), stability selection (subsample filtering)
- **Regulatory window:** ECB July 2025 Guide to Internal Models + EU AI Act Aug 2026 applicability align directly with ExpertRuleFit's design goals

## References

1. Friedman & Popescu (2008) — *Predictive Learning via Rule Ensembles*, Annals of Applied Statistics
2. Zou & Hastie (2005) — *Regularization and variable selection via the elastic net*, JRSS-B
3. Singh et al. (2021) — *imodels: a python package for fitting interpretable models*, JOSS
4. Zou (2006) — *The Adaptive LASSO and Its Oracle Properties*, JASA
5. Nori et al. (2019) — *InterpretML: A Unified Framework for Machine Learning Interpretability*, arXiv:1909.09223
6. Benard et al. (2021) — *SIRUS: Stable and Interpretable RUle Sets*, EJS
7. Liu & Mazumder (2023) — *FIRE: Fast Interpretable Rule Extraction*, KDD
8. Nalenz & Augustin (2022) — *Compressed Rule Ensembles*, AISTATS
9. Meinshausen & Bühlmann (2010) — *Stability Selection*, JRSS-B
10. Rudin et al. (2024) — *Amazing Things Come From Having Many Good Models*, ICML

## License

MIT

## Author

Djebar Hammouche — AI & Data Engineer
