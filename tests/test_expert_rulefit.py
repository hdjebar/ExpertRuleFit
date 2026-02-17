"""Unit tests for ExpertRuleFit."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from expertrulefit import ExpertRuleFit


@pytest.fixture
def banking_data():
    """Generate synthetic banking data for testing."""
    rng = np.random.RandomState(42)
    n = 500
    X = np.column_stack([
        rng.uniform(0, 5, n),      # country_risk
        rng.exponential(30, n),      # payment_delay
        rng.lognormal(8, 1.5, n),   # tx_volume
        rng.beta(2, 5, n),          # cash_ratio
        rng.beta(1.5, 8, n),        # night_tx_ratio
        rng.uniform(0, 20, n),      # tenure
    ])
    feature_names = ["country_risk", "payment_delay", "tx_volume",
                     "cash_ratio", "night_tx_ratio", "tenure"]

    logit = (0.8 * (X[:, 3] > 0.4) + 0.5 * (X[:, 4] > 0.15)
             + 0.1 * (X[:, 0] > 2.5) + rng.normal(0, 0.3, n) - 0.8)
    y = (logit > 0).astype(int)

    return X, y, feature_names


@pytest.fixture
def rules():
    """Define test rules."""
    confirmatory = [
        {
            "name": "CSSF: Country risk > 2.5",
            "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 2.5).astype(float),
        },
    ]
    optional = [
        {
            "name": "Analyst: Night TX > 0.15",
            "evaluate": lambda X, fn: (X[:, fn.index("night_tx_ratio")] > 0.15).astype(float),
        },
    ]
    return confirmatory, optional


def test_fit_and_predict(banking_data, rules):
    """Test basic fit/predict cycle."""
    X, y, fn = banking_data
    conf, opt = rules

    erf = ExpertRuleFit(max_rules=20, random_state=42)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=conf, optional_rules=opt)

    proba = erf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)

    labels = erf.predict(X)
    assert set(labels).issubset({0, 1})


def test_confirmatory_preserved(banking_data, rules):
    """Test that confirmatory rules have non-zero coefficients."""
    X, y, fn = banking_data
    conf, opt = rules

    erf = ExpertRuleFit(confirmatory_penalty=1e-6, max_rules=20, random_state=42)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=conf, optional_rules=opt)

    assert erf.confirmatory_all_active_, "Confirmatory rules should be preserved"
    assert all(abs(c) > 1e-10 for c in erf.confirmatory_coefs_)


def test_explain(banking_data, rules):
    """Test explanation output format."""
    X, y, fn = banking_data
    conf, opt = rules

    erf = ExpertRuleFit(max_rules=20, random_state=42)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=conf, optional_rules=opt)

    explanations = erf.explain(X[0])
    assert isinstance(explanations, list)
    for exp in explanations:
        assert "rule" in exp
        assert "category" in exp
        assert "contribution" in exp
        assert "active" in exp
        assert "coefficient" in exp


def test_get_active_rules(banking_data, rules):
    """Test active rules retrieval."""
    X, y, fn = banking_data
    conf, opt = rules

    erf = ExpertRuleFit(max_rules=20, random_state=42)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=conf, optional_rules=opt)

    active = erf.get_active_rules()
    assert isinstance(active, list)
    # At least the confirmatory rule should be active
    categories = [r["category"] for r in active]
    assert "confirmatory" in categories


def test_export_sql(banking_data, rules):
    """Test SQL export."""
    X, y, fn = banking_data
    conf, opt = rules
    conf[0]["sql_condition"] = "country_risk > 2.5"

    erf = ExpertRuleFit(max_rules=20, random_state=42)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=conf, optional_rules=opt)

    sql = erf.export_sql("scoring_input")
    assert "SELECT" in sql
    assert "risk_score" in sql
    assert "scoring_input" in sql


def test_summary(banking_data, rules, capsys):
    """Test summary output."""
    X, y, fn = banking_data
    conf, opt = rules

    erf = ExpertRuleFit(max_rules=20, random_state=42)
    erf.fit(X, y, feature_names=fn, confirmatory_rules=conf, optional_rules=opt)

    erf.summary()
    captured = capsys.readouterr()
    assert "ExpertRuleFit" in captured.out
    assert "Confirmatory" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
