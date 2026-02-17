# ExpertRuleFit

**Reproducible interpretable ML for regulated banking environments.**

ExpertRuleFit extends [RuleFit](https://arxiv.org/abs/0811.1679) (Friedman & Popescu 2008) with **Elastic Net + Bootstrap Stabilization** to guarantee 100/100 seed reproducibility. Built for banking compliance under CSSF/EU AI Act.

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

# Fit
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

## Features

- **100/100 reproducible** — identical rules across 100 random seeds on 3 datasets
- **Deterministic by design** — fixed internal seeds, single-threaded BLAS
- **Interpretable** — rule-based model, transparent by design (EU AI Act Art. 13)
- **Auditable** — stable output enables consistent regulatory reporting
- **Production-ready** — no seed sensitivity in deployment
- **Built on proven foundations** — extends [imodels](https://github.com/csinva/imodels) (Cynthia Rudin, Duke)

## Installation

```bash
pip install imodels scikit-learn numpy
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
expertrulefit_validation.py  # Full benchmark (3 datasets × 100 seeds)
examples/
    quick_demo.py            # Quick 10-seed reproducibility demo
tests/
    test_expert_rulefit.py   # Unit tests
output/                      # Benchmark results (figures, CSVs, report)
```

## How It Works

### Standard RuleFit (unstable)
```
Random trees (seed-dependent) → Lasso → rule selection
Different seed → different trees → different rules
```

### ExpertRuleFit (deterministic)
```
Fixed-seed trees → Bootstrap × ElasticNetCV → frequency filter → final fit
Same data → same trees → same stable rules → same output
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

## License

MIT

## Author

Djebar Hammouche — AI & Data Engineer
