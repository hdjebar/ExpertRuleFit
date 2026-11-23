# ExpertRuleFit

**Reproducible interpretable ML for regulated banking environments — EBM-grounded rules that survive any regularization.**

ExpertRuleFit extends [RuleFit](https://arxiv.org/abs/0811.1679) (Friedman & Popescu 2008) with **Elastic Net + Bootstrap Stabilization** to guarantee 100/100 seed reproducibility, plus **confirmatory rules** that survive any regularization strength. Integrates with [InterpretML](https://github.com/interpretml/interpret) Explainable Boosting Machines (EBM) to auto-discover interaction rules grounded in shape functions. Built for banking compliance under CSSF/EU AI Act.

## The Problem

Standard RuleFit uses Lasso (L1) for rule selection, which is **inherently unstable**: different random seeds produce different rules, even on the same data. In regulated banking (CSSF, EU AI Act, BCBS 239), this makes the model **non-deployable** — a regulator who re-executes the model must get identical output.

## The Solution

ExpertRuleFit replaces Lasso with a **deterministic-by-design** pipeline:

1. **Fixed-seed Gradient Boosting** — deterministic tree ensemble (`_BASE_SEED=42`) generates candidate rules + linear features via imodels RuleFit, independent of user `random_state`
2. **Expert rule augmentation** — append confirmatory (regulatory) and optional (analyst) rules with adaptive penalty weighting (Zou 2006)
3. **Bootstrap × LogisticRegressionCV (elastic net)** — 10 deterministic bootstrap iterations, each fitting weighted Elastic Net (L1+L2) with per-feature penalty scaling
4. **Frequency-based filtering** — keep only rules selected in ≥ 80% of bootstraps; confirmatory rules are force-included
5. **Final LogisticRegressionCV** — refit on stable features only with adaptive weighting
6. **Confirmatory enforcement** — if any regulatory rule is zeroed by the solver, a post-hoc constrained refit (unpenalized logistic regression) guarantees inclusion

Guaranteed reproducible within a deterministic execution envelope: **pinned dependency versions, single-threaded BLAS/OpenMP, fixed random seeds, validated input schema with stable feature order, and consistent platform/runtime settings ⇒ identical rule set and identical predictions**.

> **Container deployment:** For a step-by-step guide to enforcing this guarantee via Docker (pinned digests, locked wheels, deterministic numerics, schema validation, auditable fingerprints), see **[docs/REPRODUCIBILITY_CONTAINERS.md](docs/REPRODUCIBILITY_CONTAINERS.md)**.

## Validation Results

Seed sensitivity on **3 credit-scoring datasets** (one stratified 80/20 split, preprocessing fitted on the training split; pinned stack in [`docker/`](docker/): Python 3.11, scikit-learn 1.7.2, imodels 3.0.4). RuleFit is fitted with 100 seeds. ExpertRuleFit ignores the seed by design, so it is run 3 times in separate processes and checked for bitwise-identical output (rule-set hash and SHA-256 of test probabilities).

| Dataset | RuleFit: runs with the most frequent rule set | ExpertRuleFit: identical runs | RuleFit AUC (mean ± sd over seeds) | ExpertRuleFit AUC |
|---------|:---:|:---:|:---:|:---:|
| German Credit (UCI) | 1/100 | **3/3** | 0.788 ± 0.009 | 0.790 |
| Taiwan Credit (UCI, 5,000-row subsample) | 1/100 | **3/3** | 0.764 ± 0.003 | 0.769 |
| HMEQ | 1/100 | **3/3** | 0.903 ± 0.003 | 0.903 |

- Standard RuleFit returned a different rule set for every seed (100 distinct sets per dataset); mean pairwise Jaccard 0.036–0.058.
- ExpertRuleFit returned the same rules and the same probabilities in every run, with test AUC within 0.005 of RuleFit's mean.
- This measures **determinism**, which holds by construction, not stability when the training data change (that is the B1 study in [`experiments/`](experiments/)).
- Reproduce: `python experiments/fetch_data.py`, then `experiments/table1.py` in the `docker/` image. Results: [`experiments/results/table1/`](experiments/results/table1/). One ExpertRuleFit fit took 10 min (German) to 3.5 h (HMEQ) single-threaded.

## Quick Start

```python
from expertrulefit import ExpertRuleFit

# Fit (auto rules only — reproducible)
erf = ExpertRuleFit(max_rules=50, n_bootstrap=10, rule_threshold=0.8)
erf.fit(X_train, y_train, feature_names=feature_names)

# Predict
proba = erf.predict_proba(X_test)[:, 1]

# Inspect stable rules
for rule in erf.get_rule_importance()[:5]:
    print(f"coef={rule['coefficient']:+.4f} | {rule['name']}")

# Summary
erf.summary()
```

### With Confirmatory Rules (regulatory requirements)

```python
# Define rules that MUST survive regularization
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
confirmatory = [{
    "name": "High cash ratio",
    "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
    "expected_sign": +1,
}]
erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)

erf.confirmatory_sign_ok_        # False if any declared sign is contradicted
erf.diagnose_confirmatory()      # per confirmatory rule (training data):
#   max_abs_corr / most_correlated  – strongest correlation with any other candidate column
#   vif                             – variance inflation factor (among retained columns)
#   displaced_rules                 – correlated candidates (|r| >= 0.5) inactive in the
#                                     final model (dropped by the bootstrap filter or
#                                     zeroed), with their bootstrap frequency
#   correlated_coef_shift           – change in those rules' coefficients when the
#                                     confirmatory rule is removed (ridge refits)
erf.diagnose_confirmatory(X_val, y_val, corr_threshold=0.7)   # recompute on other data
```

`RuleSpec(..., expected_sign=+1)` carries the sign through `to_rule_dict()` and JSON save/load. `summary()` and `audit_bundle()` include the sign status and the diagnostics. A near-unpenalised confirmatory rule that duplicates generated rules will take their weight; the diagnostics make that visible, they do not prevent it.

### Choosing `confirmatory_penalty` (w_c)

Preliminary sensitivity results (German Credit only, one split, pinned stack; `experiments/results/sensitivity/`):

- **w_c ≥ 1e-2 is the reliable range; 1e-2 is the default.** It gave no saga ConvergenceWarnings, test AUC 0.784–0.797, and fits 3× faster than at 1e-4. Below 1e-2, warnings appear (3–6 per fit at 1e-4, hundreds at 1e-8), and coefficients of confirmatory rules are not trustworthy.
- **Avoid 1e-8** (the pre-0.7 default). Even with standardisation it lowered AUC by about 0.03–0.04 and degraded the Brier score (0.229 vs about 0.16).
- **Expect sign contradictions** when a confirmatory rule duplicates terms the model already has. "Checking account < 0 DM" got a negative conditional coefficient because the checking-status linear term and generated rules carried the effect. Check `confirmatory_sign_ok_` and `diagnose_confirmatory()`.

After standardisation, solver conditioning depends on the 1/√w_c ratio rather than on the dataset; the HMEQ and Taiwan runs are confirmation.

### With EBM-Discovered Interactions (automated pipeline)

```python
from expertrulefit import ExpertRuleFit, discover_interaction_rules

# EBM discovers which feature pairs interact, then finds optimal thresholds.
# Returned rules are OPTIONAL by default (rule_type="optional").
rules, ebm = discover_interaction_rules(X_train, y_train, feature_names, top_k=3)

# Data-discovered rules enter as optional: reduced penalty, but they must
# still survive bootstrap filtering like any generated rule.
erf = ExpertRuleFit(max_rules=50)
erf.fit(X_train, y_train, feature_names=fn,
        confirmatory_rules=confirmatory,   # expert-imposed rules only
        optional_rules=rules)              # EBM-discovered rules
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

**Architecture:** `Features → EBM(p_ebm) + ExpertRuleFit(p_erf) → LogisticRegression → final_score`. Every layer is interpretable — no post-hoc approximations. Cross-validated OOF predictions prevent stacking overfitting.

## Prediction & Classification

ExpertRuleFit is a **binary classifier** (classes {0, 1}). It outputs hard labels and logistic probabilities; post-hoc calibration is optional (`calibration="sigmoid"` or `"isotonic"`).

### How Prediction Works

```
Training data
  → Fixed-seed GradientBoosting (candidate rules + linear features)
  → Augment with expert rules (confirmatory / optional, adaptive penalty weighting)
  → Bootstrap × weighted LogisticRegressionCV (elastic net stability filtering)
  → Frequency filter (≥ 80%, confirmatory force-included)
  → Final LogisticRegressionCV on stable features
  → Constrained refit if confirmatory rules zeroed
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
p_default = proba[:, 1]            # P(y=1) — e.g., probability of default
```

Probabilities are logistic-regression outputs; their calibration has not yet been evaluated on the benchmark datasets. If calibrated scores matter (e.g. PD estimates), set `calibration="sigmoid"` or `"isotonic"` and check reliability on held-out data. Each row sums to 1.0. Use `proba[:, 1]` for AUC, ranking, and threshold tuning.

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

# Bootstrap stability of each rule
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
- **Confirmatory**: regulatory rules with near-zero penalty (guaranteed active)
- **Optional**: analyst rules with reduced penalty (may be eliminated)

> See `examples/prediction_basics.py`, `examples/classification_with_rules.py`, and `examples/prediction_inspection.py` for runnable demonstrations.

## Two Guarantees

1. **Reproducibility** — same data → same rules → same predictions (100/100 seeds)
2. **Rule preservation** — confirmatory (regulatory) rules survive any regularization strength

## Features

- **100/100 reproducible** — identical rules across 100 random seeds on 3 datasets
- **Confirmatory rules** — regulatory rules that are never eliminated by regularization
- **EBM bridge** — auto-discover interactions with InterpretML, inject as confirmatory rules
- **Dual stacking** — EBM + ExpertRuleFit + LogisticRegression meta-classifier, per-sample explanations
- **Deterministic by design** — fixed internal seeds, single-threaded BLAS
- **Interpretable** — rule-based model, transparent by design (EU AI Act Art. 13)
- **Auditable** — stable output enables consistent regulatory reporting
- **Production-ready** — no seed sensitivity in deployment
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

### ExpertRuleFit (deterministic + rule preservation)
```
Fixed-seed GradientBoosting (BASE_SEED=42) → extract rules + linear features
  → augment with expert rules (adaptive penalty weighting)
  → Bootstrap × weighted LogisticRegressionCV (elastic net)
  → frequency filter (≥ 80%, confirmatory force-included)
  → final LogisticRegressionCV on stable features
  → constrained refit if confirmatory rules zeroed
Same data → same trees → same stable rules → same output
Confirmatory rules: penalty ≈ 0 → never eliminated; constrained refit as fallback
```

### Why Elastic Net?

The L2 component of Elastic Net provides two key benefits:
1. **Group stability** — correlated rules are kept together (vs Lasso's arbitrary selection)
2. **Smooth coefficient paths** — small data perturbations don't cause rule switching

Combined with bootstrap frequency filtering, this eliminates the selection instability
that makes standard RuleFit non-reproducible.

## Regulatory Compliance

| Requirement | Standard | How ExpertRuleFit Complies |
|-------------|----------|---------------------------|
| EU AI Act Art. 9 | Risk management | Reproducible output guarantees consistent risk assessment |
| EU AI Act Art. 12 | Automatic logging | Same model produces identical results at each audit |
| EU AI Act Art. 13 | Transparency | Stable rules can be documented and explained consistently |
| BCBS 239 Principle 3 | Accuracy & integrity | Verifiable results across re-executions |
| CSSF Circular 12/552 | Model governance | Regulator can re-execute and obtain identical output |

## State of the Art

For a comprehensive survey of recent research (2023--2026) on RuleFit extensions, competing rule-based methods, stability techniques, Rashomon sets, regulatory AI, and how ExpertRuleFit is positioned relative to the field, see **[docs/SOTA.md](docs/SOTA.md)**.

**Key findings:**
- **Closest competitors:** SIRUS (stability), FIRE/CRE (sparsity), Stability Selection (bootstrap filtering) — none combine all four of ExpertRuleFit's components
- **Unique niche:** No existing method integrates RuleFit rule generation + bootstrap stability selection + weighted regularization for confirmatory rules + EBM interaction discovery
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
