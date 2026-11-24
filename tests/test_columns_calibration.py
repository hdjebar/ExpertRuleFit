"""0.8.1 — canonical column order (decision c) and Platt scaling in
constrained mode (decision d)."""

import numpy as np
import pytest
from scipy.special import logit

from expertrulefit import ExpertRuleFit, RuleSpec

FAST = dict(n_estimators=30, max_rules=10, n_bootstrap=2)
NAMES = ["income", "age", "debt", "zeta"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(11)
    n = 300
    X = np.column_stack(
        [
            rng.normal(size=n),
            rng.uniform(18, 70, n),
            rng.uniform(0, 5, n),
            rng.normal(size=n),
        ]
    )
    eta = -1.0 * (X[:, 0] > 0) + 0.6 * X[:, 2] - 1.5 + rng.normal(scale=0.7, size=n)
    y = (eta > 0).astype(int)
    return X, y


def by_name(fn):
    """Confirmatory rule that finds its column by name (any column order)."""
    return [
        {
            "name": "income>0",
            "evaluate": lambda X, names: (X[:, names.index("income")] > 0).astype(
                float
            ),
            "expected_sign": -1,
        }
    ]


def spec_rule(fn):
    """The same rule as a RuleSpec compiled against the given name order."""
    spec = RuleSpec(
        name="income>0",
        clauses=[{"feature": "income", "op": ">", "threshold": 0.0}],
        expected_sign=-1,
    )
    return [spec.to_rule_dict(fn)]


def fit(X, y, fn, rules=by_name, **kw):
    return ExpertRuleFit(**FAST, **kw).fit(
        X, y, feature_names=fn, confirmatory_rules=rules(fn)
    )


@pytest.fixture(scope="module")
def reference(data):
    X, y = data
    return fit(X, y, NAMES)


# ---------------------------------------------------------------- (c) columns


def test_canonical_columns_are_sorted_by_name(reference):
    assert reference.column_order == "canonical"
    assert reference.feature_names_ == sorted(NAMES)
    assert reference.input_feature_names_ == NAMES
    assert [NAMES[j] for j in reference.column_order_] == sorted(NAMES)
    linear = [n for n in reference.rule_names_ if n.startswith("linear:")]
    assert linear == [f"linear:{f}" for f in sorted(NAMES)]


@pytest.mark.parametrize("rules", [by_name, spec_rule], ids=["callable", "rulespec"])
@pytest.mark.parametrize("perm_seed", [0, 1, 2, "reversed"])
def test_any_column_permutation_is_bitwise_identical(data, reference, rules, perm_seed):
    X, y = data
    ref = reference if rules is by_name else fit(X, y, NAMES, rules=rules)
    if perm_seed == "reversed":
        perm = np.arange(len(NAMES))[::-1]
    else:
        perm = np.random.RandomState(perm_seed).permutation(len(NAMES))
    fn = [NAMES[j] for j in perm]
    m = fit(X[:, perm], y, fn, rules=rules)
    assert m.feature_names_ == ref.feature_names_
    assert m.rule_names_ == ref.rule_names_
    assert m.get_selected_rules() == ref.get_selected_rules()
    np.testing.assert_array_equal(m.predict_proba(X[:, perm]), ref.predict_proba(X))
    assert m.fingerprint(X[:, perm]) == ref.fingerprint(X)


def test_rows_and_columns_permuted_together(data, reference):
    X, y = data
    rows = np.random.RandomState(5).permutation(len(y))
    cols = np.array([3, 1, 0, 2])
    fn = [NAMES[j] for j in cols]
    m = fit(X[rows][:, cols], y[rows], fn)
    assert m.fingerprint(X[:, cols]) == reference.fingerprint(X)


@pytest.mark.parametrize("candidates", ["trees"])
def test_permutation_invariance_other_paths(data, candidates):
    X, y = data
    perm = np.array([2, 0, 3, 1])
    fn = [NAMES[j] for j in perm]
    a = fit(X, y, NAMES, candidates=candidates)
    b = fit(X[:, perm], y, fn, candidates=candidates)
    assert a.fingerprint(X) == b.fingerprint(X[:, perm])


def test_column_order_in_fingerprint_and_bundle(data, reference):
    X, y = data
    fp = reference.fingerprint()
    assert set(fp) == {"rules_sha256", "coefficients_sha256", "schema_sha256"}
    bundle = reference.audit_bundle(X_reference=X)
    assert bundle["column_order"] == "canonical"
    assert bundle["feature_names"] == sorted(NAMES)
    assert bundle["input_feature_names"] == NAMES
    given = fit(X, y, NAMES, column_order="as_given")
    assert given.feature_names_ == NAMES
    assert given.audit_bundle()["column_order"] == "as_given"
    assert given.fingerprint()["schema_sha256"] != fp["schema_sha256"]


def test_as_given_depends_on_column_order(data):
    X, y = data
    perm = np.array([3, 2, 1, 0])
    fn = [NAMES[j] for j in perm]
    a = fit(X, y, NAMES, column_order="as_given")
    b = fit(X[:, perm], y, fn, column_order="as_given")
    assert b.feature_names_ == fn
    # same data, so predictions must still be valid probabilities
    p = b.predict_proba(X[:, perm])[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert a.fingerprint()["schema_sha256"] != b.fingerprint()["schema_sha256"]
    # the canonical default exists because this differs:
    assert (
        a.fingerprint(X)["predictions_sha256"]
        != (b.fingerprint(X[:, perm])["predictions_sha256"])
    )


def test_save_load_keeps_column_order(data, reference, tmp_path):
    X, _ = data
    path = tmp_path / "m.joblib"
    reference.save(path)
    loaded = ExpertRuleFit.load(path, confirmatory_rules=by_name(NAMES))
    assert loaded.column_order == "canonical"
    np.testing.assert_array_equal(loaded.column_order_, reference.column_order_)
    assert loaded.fingerprint(X) == reference.fingerprint(X)


def test_column_errors(data, reference):
    X, y = data
    with pytest.raises(ValueError, match="unique"):
        ExpertRuleFit(**FAST).fit(X, y, feature_names=["a", "b", "a", "c"])
    with pytest.raises(ValueError, match="column_order"):
        ExpertRuleFit(column_order="sorted", **FAST).fit(X, y, feature_names=NAMES)
    with pytest.raises(ValueError, match="feature_names"):
        ExpertRuleFit(**FAST).fit(X, y, feature_names=NAMES[:3])
    with pytest.raises(ValueError, match="columns"):
        reference.predict_proba(X[:, :3])


# ------------------------------------------------------------ (d) calibration


@pytest.fixture(scope="module")
def platt(data):
    X, y = data
    return fit(X, y, NAMES, calibration="sigmoid")


def test_sigmoid_allowed_in_constrained_mode(platt, reference, data):
    X, _ = data
    status = {st["name"]: st for st in reference.confirmatory_status_}
    assert not status["income>0"].get("bound_active")
    info = platt.calibration_info_
    assert platt.confirmatory_mode == "constrained"
    assert info["method"] == "sigmoid" and info["applied"] and info["a"] > 0
    assert info["n_folds"] == 5
    # The fitted block is untouched: same terms, coefficients, sign bounds.
    assert (
        platt.fingerprint()["coefficients_sha256"]
        == (reference.fingerprint()["coefficients_sha256"])
    )
    assert platt.constrained_info_ == reference.constrained_info_
    assert not np.array_equal(platt.predict_proba(X), reference.predict_proba(X))


def test_platt_keeps_log_odds_additive(platt, reference, data):
    X, _ = data
    a, b = platt.calibration_info_["a"], platt.calibration_info_["b"]
    eta = logit(reference.predict_proba(X)[:, 1])
    eta_cal = logit(platt.predict_proba(X)[:, 1])
    np.testing.assert_allclose(eta_cal, a * eta + b, rtol=1e-9, atol=1e-9)
    # additive decomposition: intercept + sum of term contributions
    Z = platt._build_predict_matrix(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    fm = platt.final_model_
    contrib = Z * fm.coef_[0]
    np.testing.assert_allclose(
        eta_cal, a * (fm.intercept_[0] + contrib.sum(axis=1)) + b, rtol=1e-9, atol=1e-9
    )
    assert platt.audit_bundle()["calibration_info"] == platt.calibration_info_


def test_platt_is_deterministic_and_saved(platt, data, tmp_path):
    X, y = data
    again = fit(X, y, NAMES, calibration="sigmoid")
    assert again.calibration_info_ == platt.calibration_info_
    assert again.fingerprint(X) == platt.fingerprint(X)
    path = tmp_path / "p.joblib"
    platt.save(path)
    loaded = ExpertRuleFit.load(path, confirmatory_rules=by_name(NAMES))
    assert loaded.calibration_info_ == platt.calibration_info_
    np.testing.assert_array_equal(loaded.predict_proba(X), platt.predict_proba(X))


def test_platt_column_permutation_invariant(platt, data):
    X, y = data
    perm = np.array([1, 3, 0, 2])
    fn = [NAMES[j] for j in perm]
    m = fit(X[:, perm], y, fn, calibration="sigmoid")
    assert m.fingerprint(X[:, perm]) == platt.fingerprint(X)


def test_isotonic_still_raises_in_constrained_mode(data):
    X, y = data
    with pytest.raises(ValueError, match="isotonic"):
        fit(X, y, NAMES, calibration="isotonic")
