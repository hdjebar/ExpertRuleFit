"""Tests for confirmatory-rule sign checks and absorption diagnostics (A2)."""

import warnings

import numpy as np
import pytest

from expertrulefit import ExpertRuleFit, RuleSpec
from expertrulefit.rule_spec import load_rule_specs, save_rule_specs

# Small, fast configuration: these tests exercise the diagnostics, not the
# rule generator.
# These tests exercise the refit-mode sign check and diagnostics.
FAST = dict(n_estimators=30, max_rules=10, n_bootstrap=2, confirmatory_mode="refit",
            selection="bootstrap")


@pytest.fixture(scope="module")
def signal_data():
    """Standardised data where y is driven by b > 0 (and weakly by a)."""
    rng = np.random.RandomState(7)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    X = np.column_stack([a, b, c])
    logit = 2.5 * (b > 0) + 0.5 * a - 1.25 + rng.normal(scale=0.5, size=n)
    y = (logit > 0).astype(int)
    return X, y, ["a", "b", "c"]


def _rule_b_pos(expected_sign=None):
    rule = {
        "name": "b>0",
        "evaluate": lambda X, fn: (X[:, 1] > 0).astype(float),
    }
    if expected_sign is not None:
        rule["expected_sign"] = expected_sign
    return rule


def _noise_rule():
    # Parity of the third decimal of feature c (independent noise):
    # unrelated to y and nearly uncorrelated with the features. It is a
    # row-wise function of X, as rules must be (rows are sorted before
    # fitting with the default row_order="canonical").
    return {
        "name": "noise",
        "evaluate": lambda X, fn: (np.floor(np.abs(X[:, 2]) * 1000) % 2).astype(float),
    }


# ---------------------------------------------------------------------------
# Sign check
# ---------------------------------------------------------------------------


def test_sign_ok_when_expectation_matches(signal_data):
    X, y, fn = signal_data
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = ExpertRuleFit(**FAST).fit(
            X, y, feature_names=fn, confirmatory_rules=[_rule_b_pos(+1)]
        )
    assert m.confirmatory_sign_ok_ is True
    st = m.confirmatory_status_[0]
    assert st["expected_sign"] == 1 and st["sign_ok"] is True
    assert st["coefficient"] > 0
    assert not [w for w in rec if "expected_sign" in str(w.message)]


def test_sign_flip_warns_and_flags(signal_data):
    """Rule raises risk in the data; expert declared -1 -> flagged."""
    X, y, fn = signal_data
    with pytest.warns(UserWarning, match=r"expected_sign=-1"):
        m = ExpertRuleFit(**FAST).fit(
            X, y, feature_names=fn, confirmatory_rules=[_rule_b_pos(-1)]
        )
    assert m.confirmatory_sign_ok_ is False
    st = m.confirmatory_status_[0]
    assert st["sign_ok"] is False and st["coefficient"] > 0
    # The rule is still kept: the sign is checked, not enforced.
    assert m.confirmatory_all_active_
    text = m.summary(return_string=True)
    assert "CONTRADICTS" in text
    assert "contradict their expected sign" in text
    assert m.audit_bundle()["confirmatory_sign_ok"] is False


def test_no_expected_sign_means_no_check(signal_data):
    X, y, fn = signal_data
    m = ExpertRuleFit(**FAST).fit(
        X, y, feature_names=fn, confirmatory_rules=[_rule_b_pos()]
    )
    assert m.confirmatory_sign_ok_ is True
    assert m.confirmatory_status_[0]["sign_ok"] is None


def test_invalid_expected_sign_raises(signal_data):
    X, y, fn = signal_data
    bad = _rule_b_pos()
    bad["expected_sign"] = 2
    with pytest.raises(ValueError, match="expected_sign"):
        ExpertRuleFit(**FAST).fit(X, y, feature_names=fn, confirmatory_rules=[bad])


# ---------------------------------------------------------------------------
# Absorption diagnostics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def absorbing_model(signal_data):
    """Confirmatory b>0 duplicates the strongest generated signal.

    The generated rules on b are absorbed (at the bootstrap filter or in the
    final fit); the diagnostics must report them.
    """
    X, y, fn = signal_data
    m = ExpertRuleFit(**FAST).fit(
        X, y, feature_names=fn,
        confirmatory_rules=[_rule_b_pos(+1), _noise_rule()],
    )
    return m, X, y


def test_absorption_detected_for_duplicating_rule(absorbing_model):
    m, _, _ = absorbing_model
    diags = {d["name"]: d for d in m.diagnose_confirmatory()}
    d = diags["b>0"]
    # Candidate rules on b are near-duplicates of b>0 ...
    assert d["max_abs_corr"] > 0.8
    assert d["most_correlated"].startswith(("rule[", "linear:b"))
    assert d["n_correlated"] >= 1
    # ... and none of them is active: b>0 took their weight.
    assert d["n_displaced"] >= 1
    assert set(d["displaced_rules"]) == set(d["displaced_bootstrap_frequency"])
    assert all(0.0 <= f <= 1.0 for f in d["displaced_bootstrap_frequency"].values())


def test_no_absorption_for_unrelated_rule(absorbing_model):
    m, _, _ = absorbing_model
    d = {d["name"]: d for d in m.diagnose_confirmatory()}["noise"]
    assert d["max_abs_corr"] < 0.3
    assert d["n_correlated"] == 0
    assert d["absorption_ratio"] == 0.0
    assert d["n_displaced"] == 0
    assert d["vif"] < 1.5


def test_diagnostics_in_summary_and_bundle(absorbing_model):
    m, _, _ = absorbing_model
    text = m.summary(return_string=True)
    assert "Confirmatory diagnostics" in text
    assert "absorption ratio" in text
    bundle = m.audit_bundle()
    assert len(bundle["confirmatory_diagnostics"]) == 2


def test_diagnostics_recompute_on_new_data(absorbing_model):
    m, X, y = absorbing_model
    stored = m.diagnose_confirmatory()
    again = m.diagnose_confirmatory(X, y)
    for s, a in zip(stored, again):
        assert s["name"] == a["name"]
        assert a["max_abs_corr"] == pytest.approx(s["max_abs_corr"])
        assert a["vif"] == pytest.approx(s["vif"], nan_ok=True)
    strict = m.diagnose_confirmatory(X, y, corr_threshold=0.99)
    assert all(d["corr_threshold"] == 0.99 for d in strict)
    with pytest.raises(ValueError):
        m.diagnose_confirmatory(corr_threshold=0.9)


def test_diagnostics_are_deterministic(signal_data, absorbing_model):
    X, y, fn = signal_data
    m1, _, _ = absorbing_model
    m2 = ExpertRuleFit(**FAST).fit(
        X, y, feature_names=fn,
        confirmatory_rules=[_rule_b_pos(+1), _noise_rule()],
    )
    assert m1.diagnose_confirmatory() == m2.diagnose_confirmatory()


def test_save_load_keeps_diagnostics(absorbing_model, tmp_path):
    m, X, _ = absorbing_model
    path = tmp_path / "m.joblib"
    m.save(path)
    m2 = ExpertRuleFit.load(
        path, confirmatory_rules=[_rule_b_pos(+1), _noise_rule()]
    )
    assert m2.confirmatory_sign_ok_ == m.confirmatory_sign_ok_
    assert m2.diagnose_confirmatory() == m.diagnose_confirmatory()
    np.testing.assert_array_equal(m2.predict_proba(X), m.predict_proba(X))


# ---------------------------------------------------------------------------
# RuleSpec.expected_sign
# ---------------------------------------------------------------------------


def test_rulespec_expected_sign_roundtrip(tmp_path):
    spec = RuleSpec(
        name="High cash",
        clauses=[{"feature": "cash", "op": ">", "threshold": 0.4}],
        expected_sign=1,
    )
    d = spec.to_rule_dict(["cash"])
    assert d["expected_sign"] == 1
    path = tmp_path / "specs.json"
    save_rule_specs([spec], path)
    assert load_rule_specs(path)[0].expected_sign == 1

    no_sign = RuleSpec(
        name="x", clauses=[{"feature": "cash", "op": "<", "threshold": 1}]
    )
    assert "expected_sign" not in no_sign.to_rule_dict(["cash"])
    assert RuleSpec.from_dict(
        {"name": "x", "clauses": no_sign.clauses}
    ).expected_sign is None


@pytest.mark.parametrize("bad", [0, 2, "+", True, 0.5])
def test_rulespec_rejects_bad_expected_sign(bad):
    with pytest.raises(ValueError, match="expected_sign"):
        RuleSpec(
            name="x",
            clauses=[{"feature": "f", "op": ">", "threshold": 0}],
            expected_sign=bad,
        )
