# ExpertRuleFit — Code Review

**Date**: 2026-02-21
**Scope**: Full codebase review of `expertrulefit/`, tests, examples, and validation benchmark

## Project Summary

ExpertRuleFit is a reproducible rule-based classifier for regulated banking environments.
It extends `imodels.RuleFitClassifier` with bootstrap-stabilized Elastic Net to guarantee
identical rule selection across random seeds, with confirmatory rules that survive any
regularization strength.

---

## Critical Bugs

### B1: `predict_proba` does NOT output probabilities (on-disk version)

**File**: `expertrulefit/expert_rulefit.py:396-411`

The current code uses `ElasticNetCV` (a regression model) and fakes probabilities:

```python
raw = self.final_model_.predict(X_stable_weighted)
raw = np.clip(raw, 0, 1)
return np.column_stack([1 - raw, raw])
```

`ElasticNetCV.predict()` returns unbounded regression values, not probabilities. Clipping
to [0,1] does not produce calibrated probabilities. For a PD model in banking, regulators
expect logistic probabilities, not clipped regression outputs.

**Fix**: Upgrade to `LogisticRegressionCV(penalty="elasticnet", solver="saga")`, which
natively outputs calibrated logistic probabilities.

### B2: Optional rules bypass bootstrap filtering

**File**: `expertrulefit/expert_rulefit.py:339-341`

```python
for i in range(n_expert):
    self.stable_mask_[n_auto_features + i] = True
```

ALL expert rules (confirmatory AND optional) are force-included in the stable set,
contradicting the documented behavior that optional rules "can be eliminated."

**Fix**: Only force-include confirmatory rules.

### B3: Rule name collisions

**File**: `expertrulefit/expert_rulefit.py:530-532`

```python
names.append(f"rule:{rule_str[:60]}")
```

Rule strings truncated to 60 characters can collide, causing incorrect rule tracking.

**Fix**: Use index-based unique names: `f"rule[{i}]:{rule_str}"`.

### B4: Silent total bootstrap failure

**File**: `expertrulefit/expert_rulefit.py:331-332`

```python
except Exception:
    continue
```

If all bootstraps fail silently, the fallback keeps ALL rules, defeating bootstrap filtering.

**Fix**: Track successful bootstrap count; raise `RuntimeError` if all fail, warn if some fail.

### B5: `get_rule_importance` reports coefficients in wrong space

**File**: `expertrulefit/expert_rulefit.py:464-483`

Coefficients are reported in the weighted feature space. For confirmatory rules (scaled by
`1/sqrt(1e-8) ~ 10000`), fitted coefficients appear ~10000x smaller than their true effect.

**Fix**: Add `_get_true_coefficients()` to recover original-space coefficients:
`true_coef = fitted_coef * inv_sqrt_w`.

### B6: `DualModel.explain` will break after LogisticRegression upgrade

**File**: `expertrulefit/dual_model.py:274`

```python
final_coefs = self.erf_.final_model_.coef_
```

`ElasticNetCV.coef_` is 1D; `LogisticRegressionCV.coef_` is 2D. After upgrading the core
class, `DualModel.explain` will silently produce wrong results.

**Fix**: Use `self.erf_.final_model_.coef_[0]` or `self.erf_._get_true_coefficients()`.

---

## Design Issues

### D1: Three diverged copies of core logic

- `expertrulefit/expert_rulefit.py` — `ExpertRuleFit` (ElasticNet)
- `expertrulefit_validation.py` — `ExpertRuleFitClassifier` (separate class, no expert rules)
- Shared updated version — `ExpertRuleFit` (LogisticRegression, sparse, validation)

`eval_rule_on_data` and `build_rule_feature_matrix` are duplicated. Bug fixes won't
propagate. The validation script should import from the package.

### D2: No packaging configuration

No `setup.py` or `pyproject.toml`. Scripts use `sys.path.insert(0, ...)` hacks.
No dependency declaration, version pinning, or installability.

### D3: Models are not serializable

Expert rules use lambdas for `evaluate`, which cannot be pickled. A fitted model with
confirmatory rules cannot be saved with `pickle`/`joblib.dump()`. Consider named functions
or a rule specification DSL.

### D4: BLAS threading not controlled in package

The validation script sets `OPENBLAS_NUM_THREADS=1`, but the package does not. Users
importing `ExpertRuleFit` without these settings may get non-deterministic results from
multi-threaded BLAS reductions, undermining the reproducibility guarantee.

### D5: `__repr__` comparison bug (shared version)

```python
if v != self.__class__.__init__.__defaults__
```

Compares each value against the entire defaults tuple (always false). `__repr__` will
always show all parameters.

### D6: No input validation (on-disk version)

No validation for X/y shape compatibility, binary target check, hyperparameter ranges,
or expert rule output shape/type.

---

## Test Coverage Gaps

Existing: 13 tests covering basic fit/predict, reproducibility, rule extraction, confirmatory preservation.

Missing:
1. **`eval_rule_on_data`** — negative thresholds, scientific notation, OOB columns, malformed rules
2. **`build_rule_feature_matrix`** — sparse input, empty rules, shape correctness
3. **`_refit_with_confirmatory`** — zero test coverage on the most complex fallback path
4. **Error conditions** — all bootstraps failing, zero stable features, shape mismatches
5. **`ebm_bridge.py`** — `discover_interaction_rules`, `_find_best_thresholds`, `_make_rule_fn`
6. **Optional rule elimination** — test only checks count, not that optional rules CAN be dropped
7. **Test infrastructure** — `sys.path.insert` hack instead of proper package installation

---

## Positive Aspects

1. **Mathematical foundation** — weighted feature scaling for confirmatory rules is well-grounded in adaptive LASSO theory (Zou 2006)
2. **Docstrings** — thorough parameter/return documentation
3. **Reproducibility by construction** — fixed-seed design is sound for regulatory use
4. **Bootstrap stabilization** — frequency-based filtering with configurable threshold is practical
5. **EBM bridge** — auto-discovering interactions and injecting as confirmatory rules is a useful workflow
6. **DualModel stacking** — OOF predictions prevent stacking overfitting; per-sample explanations are useful
7. **Validation benchmark** — 100-seed x 3-dataset benchmark with figures and reports is thorough

---

## Recommended Priority Actions

| Priority | Issue | Action |
|----------|-------|--------|
| **P0** | B1: Fake probabilities | Upgrade to `LogisticRegressionCV` |
| **P0** | B2: Optional rules not filterable | Only force-include confirmatory rules |
| **P0** | B5: Wrong coefficient space | Add `_get_true_coefficients()` |
| **P1** | D1: Code duplication | Make validation script import from package |
| **P1** | D2: No packaging | Add `pyproject.toml` |
| **P1** | B3: Rule name collisions | Use index-based unique names |
| **P1** | B4: Silent bootstrap failure | Raise/warn on failures |
| **P1** | B6: DualModel coef indexing | Fix for 2D `.coef_` |
| **P2** | D3: Non-serializable models | Replace lambdas with named functions |
| **P2** | D4: BLAS threading | Document or enforce in package |
| **P2** | D6: Input validation | Add validation methods |
| **P3** | Test gaps | Add missing test cases |
| **P3** | D5: `__repr__` | Fix default comparison logic |
