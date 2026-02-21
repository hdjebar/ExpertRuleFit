"""Unit tests for ExpertRuleFit."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from expertrulefit import ExpertRuleFit


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


def test_summary(credit_data, capsys):
    """Test summary output."""
    X, y, fn = credit_data

    erf = ExpertRuleFit(max_rules=20, n_bootstrap=3)
    erf.fit(X, y, feature_names=fn)

    erf.summary()
    captured = capsys.readouterr()
    assert "ExpertRuleFit" in captured.out
    assert "Stable features" in captured.out


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
    """Test that optional rules are supported but can be eliminated."""
    X, y, fn = credit_data

    optional = [
        {
            "name": "Night transactions",
            "evaluate": lambda X, fn: (X[:, fn.index("night_tx_ratio")] > 0.15).astype(float),
        },
    ]

    erf = ExpertRuleFit(max_rules=30, n_bootstrap=5, rule_threshold=0.6)
    erf.fit(X, y, feature_names=fn, optional_rules=optional)

    # Optional rule should be in stable mask (force-included)
    assert hasattr(erf, "n_expert_")
    assert erf.n_expert_ == 1

    proba = erf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
