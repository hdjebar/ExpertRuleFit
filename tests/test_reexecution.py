"""A6 — re-execution checks: row-order invariance and SHA-256 fingerprints."""

import numpy as np
import pytest

from expertrulefit import ExpertRuleFit

FAST = dict(n_estimators=30, max_rules=10, n_bootstrap=2)


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(3)
    n = 300
    X = np.column_stack([rng.normal(size=n), rng.uniform(0, 5, n), rng.normal(size=n)])
    logit = 1.5 * (X[:, 0] > 0) + 0.4 * X[:, 1] - 1.0 + rng.normal(scale=0.5, size=n)
    y = (logit > 0).astype(int)
    return X, y, ["a", "b", "c"]


CONF = [
    {
        "name": "b>2.5",
        "evaluate": lambda X, fn: (X[:, 1] > 2.5).astype(float),
        "expected_sign": 1,
    }
]


@pytest.fixture(scope="module")
def canonical(data):
    X, y, fn = data
    return ExpertRuleFit(row_order="canonical", **FAST).fit(
        X, y, feature_names=fn, confirmatory_rules=CONF
    )


@pytest.mark.parametrize("perm_seed", [0, 1])
def test_canonical_row_order_is_permutation_invariant(data, canonical, perm_seed):
    X, y, fn = data
    perm = np.random.RandomState(perm_seed).permutation(len(y))
    m = ExpertRuleFit(row_order="canonical", **FAST).fit(
        X[perm], y[perm], feature_names=fn, confirmatory_rules=CONF
    )
    assert m.get_selected_rules() == canonical.get_selected_rules()
    # bitwise: same rules, same coefficients, same predictions on the same X
    assert m.fingerprint(X) == canonical.fingerprint(X)


def test_fingerprint_contents_and_bundle(data, canonical):
    X, _, _ = data
    fp = canonical.fingerprint()
    assert set(fp) == {"rules_sha256", "coefficients_sha256"}
    fpx = canonical.fingerprint(X)
    assert fpx["n_predictions"] == len(X)
    assert all(
        len(fpx[k]) == 64
        for k in ("rules_sha256", "coefficients_sha256", "predictions_sha256")
    )
    bundle = canonical.audit_bundle(X_reference=X)
    assert bundle["fingerprint"] == fpx
    assert bundle["row_order"] == "canonical"
    assert "predictions_sha256" not in canonical.audit_bundle()["fingerprint"]


def test_fingerprint_detects_a_changed_model(data, canonical):
    X, y, fn = data
    other = ExpertRuleFit(row_order="canonical", **FAST).fit(X, y, feature_names=fn)
    fp1, fp2 = canonical.fingerprint(X), other.fingerprint(X)
    assert fp1["coefficients_sha256"] != fp2["coefficients_sha256"]
    assert fp1["predictions_sha256"] != fp2["predictions_sha256"]


def test_fingerprint_survives_save_load(data, canonical, tmp_path):
    X, _, _ = data
    path = tmp_path / "m.joblib"
    canonical.save(path)
    loaded = ExpertRuleFit.load(path, confirmatory_rules=CONF)
    assert loaded.fingerprint(X) == canonical.fingerprint(X)


def test_as_given_is_repeatable(data):
    X, y, fn = data
    m1 = ExpertRuleFit(**FAST).fit(X, y, feature_names=fn)
    m2 = ExpertRuleFit(**FAST).fit(X, y, feature_names=fn)
    assert m1.fingerprint(X) == m2.fingerprint(X)


def test_invalid_row_order():
    X = np.random.RandomState(0).normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="row_order"):
        ExpertRuleFit(row_order="sorted", **FAST).fit(X, y)
