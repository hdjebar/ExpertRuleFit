# ExpertRuleFit

**Compliance-native interpretable ML for regulated environments.**

ExpertRuleFit extends [RuleFit](https://arxiv.org/abs/0811.1679) (Friedman & Popescu 2008) with a **weighted Lasso** that guarantees preservation of regulatory rules while learning from data. Built for banking compliance under CSSF/EU AI Act.

## The Problem

Standard L1 regularization (Lasso) can eliminate rules that are **legally required** — even if they are weakly correlated with the target. In regulated banking (CSSF, GAFI, EU AI Act), this is unacceptable: compliance rules must **always** be present in the scoring model.

## The Solution

ExpertRuleFit uses the [adaptive Lasso](https://doi.org/10.1198/016214506000000735) (Zou 2006) with three penalty levels:

| Rule Type | Penalty Weight | Behavior |
|-----------|---------------|----------|
| **Confirmatory** (regulatory) | ~0 | Never eliminated by Lasso |
| **Optional** (analyst) | 0.3 | Preferred, kept if any signal |
| **Auto** (data-driven) | 1.0 | Standard Lasso selection |

### Mathematical Formulation

```
Standard Lasso:  minimize ||y - Xb||^2 + lambda * sum(|b_j|)
Weighted Lasso:  minimize ||y - Xb||^2 + lambda * sum(w_j * |b_j|)

Implementation: scale standardized features by 1/sqrt(w_j)
=> effective penalty on b_j = lambda * w_j
=> w_j ~ 0 for confirmatory rules => NEVER eliminated
```

## Validation Results

Tested on synthetic banking data (KYC/AML scoring) with **100 random seeds**:

| Metric | ExpertRuleFit | Standard Lasso |
|--------|:---:|:---:|
| Confirmatory rules preserved | **100/100 (100%)** | 18/100 (18%) |
| Mean AUC-ROC | 0.816 | 0.839 |
| AUC delta (compliance cost) | **-2.3 points** | — |

**Trade-off**: 2.3 AUC points in exchange for **100% regulatory rule preservation**. This trade-off is documented, measured, and defensible before regulators.

## Quick Start

```python
from expertrulefit import ExpertRuleFit

# Define regulatory rules (CSSF — must NEVER be eliminated)
confirmatory_rules = [
    {
        "name": "CSSF: Country risk > 1.5",
        "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 1.5).astype(float),
        "sql_condition": "country_risk > 1.5",
    },
]

# Define analyst rules (preferred but not mandatory)
optional_rules = [
    {
        "name": "Analyst: Night TX + Cash ratio",
        "evaluate": lambda X, fn: (
            (X[:, fn.index("night_tx_ratio")] > 0.2) &
            (X[:, fn.index("cash_ratio")] > 0.3)
        ).astype(float),
    },
]

# Fit
erf = ExpertRuleFit(confirmatory_penalty=1e-6, random_state=42)
erf.fit(X_train, y_train,
        feature_names=feature_names,
        confirmatory_rules=confirmatory_rules,
        optional_rules=optional_rules)

# Verify compliance
assert erf.confirmatory_all_active_, "COMPLIANCE FAILURE"

# Predict
proba = erf.predict_proba(X_test)[:, 1]

# Explain (EU AI Act Article 13)
for exp in erf.explain(X_test[0])[:5]:
    print(f"[{exp['category']:>12}] {exp['contribution']:+.4f} | {exp['rule']}")

# Export as SQL for real-time scoring
print(erf.export_sql("scoring_input"))
```

## Features

- **Guaranteed rule preservation** — confirmatory rules survive any regularization strength
- **Transparent by design** — shape functions, not post-hoc SHAP approximations
- **SQL export** — scoring function exportable as SQL for real-time production use
- **EU AI Act Article 13 compliant** — native explainability, no black-box wrapper needed
- **Built on proven foundations** — extends [imodels](https://github.com/csinva/imodels) (Cynthia Rudin, Duke)

## Installation

```bash
pip install imodels scikit-learn numpy
```

Then clone this repo:

```bash
git clone https://github.com/hdjebar/ExpertRuleFit.git
cd ExpertRuleFit
```

## Run Validation

```bash
python examples/validate_100_seeds.py
```

## Architecture

```
expertrulefit/
    __init__.py          # Package entry point
    expert_rulefit.py    # ExpertRuleFit class (~350 lines)
examples/
    validate_100_seeds.py  # 100-seed reproducibility experiment
tests/
    test_expert_rulefit.py # Unit tests
```

## References

1. Friedman & Popescu (2008) — *Predictive Learning via Rule Ensembles*, Annals of Applied Statistics
2. Zou (2006) — *The Adaptive LASSO and Its Oracle Properties*, JASA
3. Singh et al. (2021) — *imodels: a python package for fitting interpretable models*, JOSS
4. Lou et al. (2013) — *Accurate Intelligible Models with Pairwise Interactions*, KDD

## EU AI Act Compliance

ExpertRuleFit is designed for **high-risk AI systems** under the EU AI Act (Annex III, Area 5b — credit scoring):

| Article | Requirement | How ExpertRuleFit Complies |
|---------|-------------|---------------------------|
| Art. 13 | Transparency & explainability | Rules ARE the model — native, not post-hoc |
| Art. 11 | Technical documentation | `summary()` + `export_sql()` for audit |
| Art. 14 | Human oversight | Rule coefficients readable by compliance officers |
| Art. 9 | Risk management | `confirmatory_all_active_` assertion in production |

## License

MIT

## Author

Djebar Hammouche — AI & Data Engineer
