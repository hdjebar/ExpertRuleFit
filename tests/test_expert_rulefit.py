"""Unit tests for ExpertRuleFit."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from expertrulefit import ExpertRuleFit
from expertrulefit.expert_rulefit import eval_rule_on_data, build_rule_feature_matrix


@pytest.fixture
def credit_data():
    """Generate synthetic credit scoring data for testing."""
    rng = np.random.RandomState(42)
    n = 500
    X = np.column_stack([
        rng.uniform(0, 5, n),       # feature_0: country_risk
        rng.exponential(30, n),      # feature_1: payment_delay
        rng.lognormal(8, 1.5, n),   # feature_2: tx_volume
        rng.beta(2, 5, n),          # feature_3: cash_ratio
        rng.beta(1.5, 8, n),        # feature_4: night_tx_ratio
        rng.uniform(0, 20, n),      # feature_5: tenure
    ])
    feature_names = ["country_risk", "payment_delay", "tx_volume",
                     "cash_ratio", "night_tx_ratio", "tenure"]

    logit = (0.8 * (X[:, 3] > 0.4) + 0.5 * (X[:, 4] > 0.15)
             + 0.1 * (X[:, 0] > 2.5) + rng.normal(0, 0.3, n) - 0.8)
    y = (logit > 0).astype(int)

    return X, y, feature_names


def test_fit_and_predict(credit_data):
    """Test basic fit/predict cycle."""
    X, y, fn = credit_data

    erf = ExpertRuleFit(max_rules=20, n_bootstrap=3)
    erf.fit(X, y, feature_names=fn)

    proba = erf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)

    labels = erf.predict(X)
    assert set(np.unique(labels)).issubset({0, 1})


def test_reproducibility(credit_data):
    """Test that two fits with different random_state produce identical rules."""
    X, y, fn = credit_data

    erf1 = ExpertRuleFit(max_rules=20, n_bootstrap=3, random_state=0)
    erf1.fit(X, y, feature_names=fn)

    erf2 = ExpertRuleFit(max_rules=20, n_bootstrap=3, random_state=99)
    erf2.fit(X, y, feature_names=fn)

    rules1 = erf1.get_selected_rules()
    rules2 = erf2.get_selected_rules()
    assert rules1 == rules2, "Different random_state should produce identical rules"

    proba1 = erf1.predict_proba(X)
    proba2 = erf2.predict_proba(X)
    np.testing.assert_array_equal(proba1, proba2)


def test_get_selected_rules(credit_data):
    """Test rule extraction."""
    X, y, fn = credit_data

    erf = ExpertRuleFit(max_rules=30, n_bootstrap=5, rule_threshold=0.6)
    erf.fit(X, y, feature_names=fn)

    rules = erf.get_selected_rules()
    assert isinstance(rules, set)
    # Model should have stable features (at minimum linear features)
    assert erf.n_stable_rules_ > 0


def test_get_rule_importance(credit_data):
    """Test rule importance retrieval."""
    X, y, fn = credit_data

    erf = ExpertRuleFit(max_rules=30, n_bootstrap=5, rule_threshold=0.6)
    erf.fit(X, y, feature_names=fn)

    importance = erf.get_rule_importance()
    assert isinstance(importance, list)
    # Should be sorted by importance (descending) if any rules selected
    for i in range(len(importance) - 1):
        assert importance[i]["abs_importance"] >= importance[i + 1]["abs_importance"]
    for r in importance:
        assert "name" in r
        assert "coefficient" in r
        assert "abs_importance" in r
        assert "bootstrap_frequency" in r
        assert "category" in r


def test_summary(credit_data, capsys):
    """Test summary output (print and return_string modes)."""
    X, y, fn = credit_data

    erf = ExpertRuleFit(max_rules=20, n_bootstrap=3)
    erf.fit(X, y, feature_names=fn)

    # Test print mode
    erf.summary()
    captured = capsys.readouterr()
    assert "ExpertRuleFit" in captured.out
    assert "Stable features" in captured.out

    # Test return_string mode
    text = erf.summary(return_string=True)
    assert isinstance(text, str)
    assert "ExpertRuleFit" in text


def test_stable_mask(credit_data):
    """Test that stable_mask_ is properly set after fit."""
    X, y, fn = credit_data

    erf = ExpertRuleFit(max_rules=20, n_bootstrap=5, rule_threshold=0.8)
    erf.fit(X, y, feature_names=fn)

    assert hasattr(erf, "stable_mask_")
    assert erf.stable_mask_.dtype == bool
    assert erf.n_stable_rules_ == erf.stable_mask_.sum()


def test_confirmatory_rules_preserved(credit_data):
    """Test that confirmatory rules survive regularization."""
    X, y, fn = credit_data

    confirmatory = [
        {
            "name": "High cash ratio",
            "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
        },
    ]

    erf = ExpertRuleFit(max_rules=30, n_bootstrap=5, rule_threshold=0.6)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=confirmatory)

    # Confirmatory rule must be in the stable set
    assert erf.confirmatory_all_active_, "Confirmatory rule was eliminated!"

    # Check it appears in selected rules
    selected = erf.get_selected_rules()
    assert any("confirmatory:" in r for r in selected)

    # Check it appears in importance with correct category
    importance = erf.get_rule_importance()
    confirmatory_entries = [r for r in importance if r["category"] == "confirmatory"]
    assert len(confirmatory_entries) >= 1

    # Verify prediction still works
    proba = erf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


def test_confirmatory_summary(credit_data, capsys):
    """Test that summary shows confirmatory rule status."""
    X, y, fn = credit_data

    confirmatory = [
        {
            "name": "High cash ratio",
            "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
        },
    ]

    erf = ExpertRuleFit(max_rules=20, n_bootstrap=3)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=confirmatory)

    erf.summary()
    captured = capsys.readouterr()
    assert "Confirmatory rules" in captured.out
    assert "High cash ratio" in captured.out


def test_optional_rules(credit_data):
    """Test that optional rules are supported but can be eliminated by bootstrap."""
    X, y, fn = credit_data

    optional = [
        {
            "name": "Night transactions",
            "evaluate": lambda X, fn: (X[:, fn.index("night_tx_ratio")] > 0.15).astype(float),
        },
    ]

    erf = ExpertRuleFit(max_rules=30, n_bootstrap=5, rule_threshold=0.6)
    erf.fit(X, y, feature_names=fn, optional_rules=optional)

    # Optional rule is registered (but may or may not survive bootstrap)
    assert hasattr(erf, "n_expert_")
    assert erf.n_expert_ == 1

    proba = erf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


def test_bootstrap_frequencies(credit_data):
    """Test that bootstrap_frequencies_ is properly set after fit."""
    X, y, fn = credit_data

    erf = ExpertRuleFit(max_rules=20, n_bootstrap=5, rule_threshold=0.8)
    erf.fit(X, y, feature_names=fn)

    assert hasattr(erf, "bootstrap_frequencies_")
    assert len(erf.bootstrap_frequencies_) > 0
    assert np.all(erf.bootstrap_frequencies_ >= 0)
    assert np.all(erf.bootstrap_frequencies_ <= 1)


def test_input_validation():
    """Test that invalid inputs raise proper errors."""
    erf = ExpertRuleFit()

    # Mismatched X and y
    with pytest.raises(ValueError, match="incompatible shapes"):
        erf.fit(np.ones((10, 3)), np.ones(5))

    # Non-binary y
    with pytest.raises(ValueError, match="binary"):
        erf.fit(np.ones((10, 3)), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]).astype(float))

    # Invalid rule_threshold
    erf_bad = ExpertRuleFit(rule_threshold=0)
    with pytest.raises(ValueError, match="rule_threshold"):
        erf_bad.fit(np.ones((10, 3)), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).astype(float))


# ---------------------------------------------------------------------------
# Tests for eval_rule_on_data parse_mode safety (P0 fix)
# ---------------------------------------------------------------------------


class TestEvalRuleSafety:
    """Verify that malformed rules never silently fire for all samples."""

    def test_parse_failure_drop_rule_returns_all_zero(self):
        """Default parse_mode='drop_rule': bad condition -> entire rule = 0."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = eval_rule_on_data("INVALID_JUNK", X, parse_mode="drop_rule")
        assert result.sum() == 0.0, "Malformed rule must evaluate to all-zero"
        assert len(w) == 1  # exactly one warning

    def test_parse_failure_raise_mode(self):
        """parse_mode='raise': bad condition -> ValueError."""
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Failed to parse"):
            eval_rule_on_data("bad_rule", X, parse_mode="raise")

    def test_parse_failure_warn_and_zero(self):
        """parse_mode='warn_and_zero': bad condition -> zeros, rule continues."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        # "X_0 > 0.5 and BAD" — first condition would match both rows,
        # but the unparseable second condition zeros the result.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = eval_rule_on_data(
                "X_0 > 0.5 and BAD_COND", X, parse_mode="warn_and_zero"
            )
        assert result.sum() == 0.0

    def test_valid_rule_unaffected_by_parse_mode(self):
        """parse_mode does not change behavior for well-formed rules."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]])
        for mode in ("drop_rule", "raise", "warn_and_zero"):
            result = eval_rule_on_data("X_0 > 1.5", X, parse_mode=mode)
            np.testing.assert_array_equal(result, [0.0, 1.0, 0.0])

    def test_invalid_parse_mode_raises(self):
        X = np.array([[1.0]])
        with pytest.raises(ValueError, match="parse_mode"):
            eval_rule_on_data("X_0 > 0", X, parse_mode="invalid_option")

    def test_default_parse_mode_is_drop_rule(self):
        """Ensure the default (no keyword) uses the safe 'drop_rule' behavior."""
        X = np.array([[1.0], [2.0]])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = eval_rule_on_data("UNPARSEABLE", X)
        assert result.sum() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
