# ExpertRuleFit

**Reproducible interpretable ML for regulated banking environments — EBM-grounded rules that survive any regularization.**

ExpertRuleFit extends [RuleFit](https://arxiv.org/abs/0811.1679) (Friedman & Popescu 2008) with **Elastic Net + Bootstrap Stabilization** to guarantee 100/100 seed reproducibility, plus **confirmatory rules** that survive any regularization strength. Integrates with [InterpretML](https://github.com/interpretml/interpret) Explainable Boosting Machines (EBM) to auto-discover interaction rules grounded in shape functions. Built for banking compliance under CSSF/EU AI Act.

## The Problem

Standard RuleFit uses Lasso (L1) for rule selection, which is **inherently unstable**: different random seeds produce different rules, even on the same data. In regulated banking (CSSF, EU AI Act, BCBS 239), this makes the model **non-deployable** — a regulator who re-executes the model must get identical output.

## The Solution

ExpertRuleFit replaces Lasso with a **deterministic-by-design** pipeline:

1. **Fixed-seed tree generation** — same candidate rules regardless of external `random_state`
2. **Bootstrap × ElasticNetCV** — 10 bootstrap samples with Elastic Net (L1+L2)
3. **Frequency-based filtering** — keep only rules selected in >= 80% of bootstraps
4. **Final ElasticNetCV** — refit on stable features only

This guarantees: **same data → same rules → same predictions → audit-ready**.

## Validation Results

Benchmarked on **3 credit scoring datasets** with **100 random seeds** each:

| Dataset | RuleFit Stability | ExpertRuleFit Stability | RuleFit AUC | ExpertRuleFit AUC |
|---------|:---:|:---:|:---:|:---:|
| German Credit (UCI) | 1/100 | **100/100** | 0.788 | 0.801 |
| Taiwan Credit (UCI) | 1/100 | **100/100** | 0.764 | 0.763 |
| HMEQ | 1/100 | **100/100** | 0.903 | 0.903 |

**Key metrics:**
- ExpertRuleFit: **100/100 stability**, Jaccard similarity **1.000**, AUC std **0.0000**
- Standard RuleFit: **1/100 stability**, Jaccard similarity ~0.05, AUC std ~0.005

> ExpertRuleFit matches RuleFit's predictive performance while guaranteeing perfect reproducibility.

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

**How it works:** confirmatory rules get near-zero regularization penalty (1e-8 vs 1.0), making it mathematically impossible for Elastic Net to eliminate them. This is implemented via feature scaling: `X_j * 1/sqrt(w_j)` where `w_j ~ 0`.

### With EBM-Discovered Interactions (automated pipeline)

```python
from expertrulefit import ExpertRuleFit, discover_interaction_rules

# EBM discovers which feature pairs interact, then finds optimal thresholds
rules, ebm = discover_interaction_rules(X_train, y_train, feature_names, top_k=3)

# Feed EBM interactions as confirmatory rules → guaranteed to survive
erf = ExpertRuleFit(max_rules=50)
erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=rules)
assert erf.confirmatory_all_active_
```

**Pipeline:** EBM (GA2M) identifies which feature pairs interact → quantile scan finds discriminative thresholds → ExpertRuleFit preserves them with near-zero penalty.

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

ExpertRuleFit is a **binary classifier** (classes {0, 1}). It outputs both hard labels and calibrated probabilities via logistic regression.

### How Prediction Works

```
Training data
  → Fixed-seed RuleFit (candidate rules + linear features)
  → Append expert rules (confirmatory / optional)
  → Bootstrap × ElasticNetCV (stability filtering)
  → Final LogisticRegressionCV on stable features
  → Logistic sigmoid → calibrated P(y=1)
```

### predict() — Hard Class Labels

```python
erf = ExpertRuleFit(max_rules=50)
erf.fit(X_train, y_train, feature_names=fn)

y_pred = erf.predict(X_test)  # array of {0, 1}
```

### predict_proba() — Calibrated Probabilities

```python
proba = erf.predict_proba(X_test)  # shape (n_samples, 2)
p_default = proba[:, 1]            # P(y=1) — e.g., probability of default
```

Probabilities are **calibrated** (true logistic outputs, not post-hoc). Each row sums to 1.0. Use `proba[:, 1]` for AUC, ranking, and threshold tuning.

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
Fixed-seed trees → append expert rules → Bootstrap × weighted ElasticNetCV → frequency filter → final fit
Same data → same trees → same stable rules → same output
Confirmatory rules: penalty ≈ 0 → never eliminated
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

## References

1. Friedman & Popescu (2008) — *Predictive Learning via Rule Ensembles*, Annals of Applied Statistics
2. Zou & Hastie (2005) — *Regularization and variable selection via the elastic net*, JRSS-B
3. Singh et al. (2021) — *imodels: a python package for fitting interpretable models*, JOSS
4. Zou (2006) — *The Adaptive LASSO and Its Oracle Properties*, JASA
5. Nori et al. (2019) — *InterpretML: A Unified Framework for Machine Learning Interpretability*, arXiv:1909.09223

## License

MIT

## Author

Djebar Hammouche — AI & Data Engineer
