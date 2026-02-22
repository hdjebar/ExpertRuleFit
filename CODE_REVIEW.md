# ExpertRuleFit — Code Review (Historical Record)

**Initial review date**: 2026-02-21
**Last updated**: 2026-02-22
**Status**: All critical and high-priority findings resolved

---

## Findings & Resolution Log

### Critical Bugs (all resolved)

| ID | Finding | Status | Resolution |
|----|---------|--------|------------|
| B1 | `predict_proba` used `ElasticNetCV` (regression), clipped to [0,1] — not true probabilities | **Resolved** | Upgraded to `LogisticRegressionCV(penalty="elasticnet", solver="saga")` — native logistic probabilities. Commit `1cf2245`. |
| B2 | Optional rules force-included in stable set (same as confirmatory) | **Resolved** | Only confirmatory rules are now force-included; optional rules must pass bootstrap threshold. Commit `1cf2245`. |
| B3 | Rule name collisions from 60-char truncation | **Resolved** | Index-based unique names: `rule[{i}]:{rule_str}`. Commit `9c287fe`. |
| B4 | Silent total bootstrap failure — all rules kept if all bootstraps fail | **Resolved** | `RuntimeError` raised if all fail; `UserWarning` if partial. Commit `9c287fe`. |
| B5 | Coefficients reported in weighted space (confirmatory ~10000x off) | **Resolved** | `_get_true_coefficients()` recovers original-space coefficients. Commit `1cf2245`. |
| B6 | `DualModel.explain` assumed 1D `.coef_` (ElasticNet) | **Resolved** | Uses `_get_true_coefficients()` for correct 2D indexing. Commit `9c287fe`. |

### Design Issues

| ID | Finding | Status | Resolution |
|----|---------|--------|------------|
| D1 | Three diverged copies of core logic | **Resolved** | Consolidated into single `expertrulefit/expert_rulefit.py`. |
| D2 | No packaging configuration | **Resolved** | Added `pyproject.toml` with dependencies and optional extras. Commit `105bd6d`. |
| D3 | Models not serializable (lambda evaluate functions) | **Addressed** | `RuleSpec` dataclass introduced with `save()`/`load()` support. |
| D4 | BLAS threading not controlled | **Documented** | README notes single-threaded BLAS for reproducibility. |
| D5 | `__repr__` default comparison bug | **Resolved** | Removed custom `__repr__`, uses sklearn `BaseEstimator.__repr__`. |
| D6 | No input validation | **Resolved** | `_validate_inputs()` and `_validate_expert_rules()` added. Commit `9c287fe`. |

### Test Coverage Gaps

| Finding | Status |
|---------|--------|
| `eval_rule_on_data` edge cases | Covered in `test_expert_rulefit.py` |
| `build_rule_feature_matrix` | Covered via integration tests |
| `_refit_with_confirmatory` | Covered via confirmatory preservation tests |
| Error conditions (bootstrap fail, zero features) | Covered |
| Optional rule elimination | Covered in `test_optional_rules_can_be_eliminated` |

---

## Strengths (unchanged from initial review)

1. **Mathematical foundation** — weighted feature scaling for confirmatory rules is well-grounded in adaptive LASSO theory (Zou 2006)
2. **Docstrings** — thorough parameter/return documentation
3. **Reproducibility by construction** — fixed-seed design is sound for regulatory use
4. **Bootstrap stabilization** — frequency-based filtering with configurable threshold
5. **EBM bridge** — auto-discovering interactions and injecting as confirmatory rules
6. **DualModel stacking** — OOF predictions prevent stacking overfitting; per-sample explanations
7. **Validation benchmark** — 100-seed x 3-dataset benchmark with figures and reports

---

## Architecture Decisions

### Why LogisticRegressionCV over ElasticNetCV?

`ElasticNetCV` is a regression estimator — its `predict()` returns unbounded values, not
probabilities. For PD models in banking, regulators expect true logistic probabilities.
`LogisticRegressionCV(penalty="elasticnet", solver="saga")` provides the same L1+L2
regularization with native sigmoid outputs.

### Why adaptive penalty weighting via feature scaling?

Standard `LogisticRegressionCV` applies uniform regularization. By scaling feature j by
`1/sqrt(w_j)`, the effective penalties become:
- L1: `l1 * sqrt(w_j) * |b_j|`
- L2: `l2 * w_j * b_j^2`

With `w_j = 1e-8` for confirmatory rules, elimination is mathematically near-impossible.
This is grounded in Zou (2006) "The Adaptive LASSO and Its Oracle Properties".

### Why constrained refit fallback?

The `1/sqrt(w_j)` scaling gives only `sqrt(w_j)` reduction on L1 (not exact). In rare
cases the solver can still zero a confirmatory rule. The post-hoc unpenalized logistic
regression fallback (`_refit_with_confirmatory`) guarantees inclusion at the cost of
sparsity.

### Why bootstrap frequency filtering?

Elastic net alone is more stable than Lasso but still sensitive to data perturbations.
Bootstrap resampling (10 iterations, threshold >= 80%) ensures only consistently selected
rules survive, producing identical rule sets across external random seeds.
