"""Tests for internal column standardisation and the 1e-2 confirmatory default."""

import joblib
import numpy as np
import pytest

from expertrulefit import ExpertRuleFit

FAST = dict(n_estimators=30, max_rules=10, n_bootstrap=2)


@pytest.fixture(scope="module")
def raw_data():
    """Unscaled features (one lognormal column in the thousands)."""
    rng = np.random.RandomState(42)
    n = 300
    X = np.column_stack([
        rng.uniform(0, 5, n),
        rng.lognormal(8, 1.5, n),
        rng.beta(2, 5, n),
        rng.beta(1.5, 8, n),
    ])
    logit = 0.8 * (X[:, 2] > 0.4) + 0.5 * (X[:, 3] > 0.15) + 0.4 * (X[:, 0] - 2.5) \
        + rng.normal(0, 0.3, n) - 0.5
    y = (logit > 0).astype(int)
    return X, y, ["risk", "volume", "cash", "night"]


CONF = [{"name": "risk>2.5",
         "evaluate": lambda X, fn: (X[:, 0] > 2.5).astype(float)}]


@pytest.fixture(scope="module")
def fitted(raw_data):
    X, y, fn = raw_data
    return ExpertRuleFit(**FAST).fit(X, y, feature_names=fn,
                                     confirmatory_rules=CONF)


def test_defaults():
    m = ExpertRuleFit()
    assert m.standardize is True
    assert m.confirmatory_penalty == 1e-2


def test_statistics_stored(fitted, raw_data):
    X, _, _ = raw_data
    p = X.shape[1]
    assert fitted.col_center_.shape == fitted.col_sd_.shape
    # linear columns centred on training means, rule/expert columns not centred
    np.testing.assert_allclose(fitted.col_center_[:p], X.mean(axis=0))
    assert np.all(fitted.col_center_[p:] == 0)
    np.testing.assert_allclose(fitted.col_sd_[:p], X.std(axis=0))
    assert len(fitted.stable_center_) == fitted.n_stable_rules_


def test_original_scale_logit_matches_predict_proba(fitted, raw_data):
    """intercept + raw stable values @ coefficients reproduces the model."""
    X, _, _ = raw_data
    Z = fitted._build_stable_raw_matrix(X)
    logit = fitted._get_true_intercept() + Z @ fitted._get_true_coefficients()
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)),
                               fitted.predict_proba(X)[:, 1], rtol=1e-8, atol=1e-10)
    assert fitted.audit_bundle()["intercept"] == pytest.approx(
        fitted._get_true_intercept())


def test_reason_codes_add_up_to_logit(fitted, raw_data):
    X, _, _ = raw_data
    reasons = fitted.reason_codes(X[:5], top_k=10_000)
    p1 = fitted.predict_proba(X[:5])[:, 1]
    for r, p in zip(reasons, p1):
        logit = fitted._get_true_intercept() + sum(c["contribution"] for c in r)
        assert logit == pytest.approx(np.log(p / (1 - p)), rel=1e-6, abs=1e-8)
        for c in r:   # feature values are on the original scale
            if c["category"] in ("confirmatory", "rule"):
                assert c["feature_value"] in (0.0, 1.0)


def test_invariant_to_feature_units(raw_data):
    """Expressing a feature in other units (x1000) leaves predictions unchanged."""
    X, y, fn = raw_data
    X2 = X.copy()
    X2[:, 1] *= 1000.0
    m1 = ExpertRuleFit(**FAST).fit(X, y, feature_names=fn)
    m2 = ExpertRuleFit(**FAST).fit(X2, y, feature_names=fn)
    assert len(m1.get_selected_rules()) == len(m2.get_selected_rules())
    np.testing.assert_allclose(m1.predict_proba(X), m2.predict_proba(X2),
                               rtol=1e-4, atol=1e-6)


def test_deterministic(raw_data, fitted):
    X, y, fn = raw_data
    m2 = ExpertRuleFit(**FAST).fit(X, y, feature_names=fn,
                                   confirmatory_rules=CONF)
    np.testing.assert_array_equal(fitted.predict_proba(X), m2.predict_proba(X))
    assert fitted.get_selected_rules() == m2.get_selected_rules()


def test_save_load_roundtrip(fitted, raw_data, tmp_path):
    X, _, _ = raw_data
    path = tmp_path / "m.joblib"
    fitted.save(path)
    m2 = ExpertRuleFit.load(path, confirmatory_rules=CONF)
    assert m2.standardize is True
    np.testing.assert_array_equal(m2.stable_center_, fitted.stable_center_)
    np.testing.assert_array_equal(m2.predict_proba(X), fitted.predict_proba(X))


def test_load_model_saved_before_standardisation(raw_data, tmp_path):
    """A pre-0.7 file (no centre/scale keys, no 'standardize' param) loads
    and predicts as an unstandardised model."""
    X, y, fn = raw_data
    m = ExpertRuleFit(standardize=False, **FAST).fit(X, y, feature_names=fn)
    path = tmp_path / "old.joblib"
    m.save(path)
    state = joblib.load(path)
    for key in ("stable_center_", "stable_scale_", "col_center_", "col_sd_"):
        state.pop(key)
    state["params"].pop("standardize")
    joblib.dump(state, path)

    old = ExpertRuleFit.load(path)
    assert old.standardize is False
    np.testing.assert_allclose(old.predict_proba(X), m.predict_proba(X))


def test_standardize_false_keeps_previous_transform(raw_data):
    X, y, fn = raw_data
    m = ExpertRuleFit(standardize=False, **FAST).fit(X, y, feature_names=fn)
    assert np.all(m.col_center_ == 0) and np.all(m.col_sd_ == 1)
    np.testing.assert_array_equal(m.stable_scale_, m.stable_inv_sqrt_w_)
