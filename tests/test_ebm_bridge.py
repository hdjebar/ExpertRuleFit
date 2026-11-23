"""Unit tests for ebm_bridge module."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pytest

from expertrulefit.ebm_bridge import (
    discover_interaction_rules,
    _find_best_thresholds,
    _make_rule_fn,
)


@pytest.fixture
def interaction_data():
    """Generate data with a known interaction effect."""
    rng = np.random.RandomState(42)
    n = 500
    X = np.column_stack(
        [
            rng.uniform(0, 5, n),  # feature_0
            rng.exponential(30, n),  # feature_1
            rng.beta(2, 5, n),  # feature_2
            rng.beta(1.5, 8, n),  # feature_3
            rng.uniform(0, 20, n),  # feature_4
        ]
    )
    feature_names = [
        "country_risk",
        "payment_delay",
        "cash_ratio",
        "night_tx_ratio",
        "tenure",
    ]

    # Ground truth includes an interaction between feature_2 and feature_3
    logit = (
        0.8 * (X[:, 2] > 0.4)
        + 0.5 * (X[:, 3] > 0.15)
        + 0.3 * (X[:, 2] > 0.3) * (X[:, 3] > 0.1)
        + rng.normal(0, 0.3, n)
        - 0.8
    )
    y = (logit > 0).astype(int)
    return X, y, feature_names


def test_make_rule_fn():
    """Test that _make_rule_fn creates a correct evaluation function."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 10, (100, 3))
    fn = ["a", "b", "c"]

    rule_fn = _make_rule_fn(0, 5.0, 1, 3.0)
    result = rule_fn(X, fn)

    expected = ((X[:, 0] > 5.0) & (X[:, 1] > 3.0)).astype(np.float64)
    np.testing.assert_array_equal(result, expected)


def test_find_best_thresholds():
    """Test that _find_best_thresholds returns valid output."""
    rng = np.random.RandomState(42)
    n = 300
    X = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n)])
    y = ((X[:, 0] > 5) & (X[:, 1] > 5)).astype(float)

    thresh_a, thresh_b, corr, direction = _find_best_thresholds(X, y, 0, 1)

    assert isinstance(thresh_a, float)
    assert isinstance(thresh_b, float)
    assert corr >= 0
    assert direction in ("risk", "protective")


def test_discover_interaction_rules(interaction_data):
    """Test full interaction discovery pipeline."""
    X, y, fn = interaction_data

    rules, ebm = discover_interaction_rules(
        X,
        y,
        fn,
        top_k=2,
        max_interactions=5,
        random_state=42,
    )

    assert isinstance(rules, list)
    assert len(rules) <= 2
    for rule in rules:
        assert "name" in rule
        assert "evaluate" in rule
        assert callable(rule["evaluate"])
        # Rule should produce valid binary output
        result = rule["evaluate"](X, fn)
        assert result.shape == (X.shape[0],)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    # EBM should be fitted
    assert hasattr(ebm, "term_importances")

    # Data-discovered rules are optional by default (A1) and carry provenance
    for rule in rules:
        assert rule["category"] == "optional"
        assert rule["source"] == "ebm_discovered"


def test_default_rule_type_is_optional():
    """discover_interaction_rules must not promote data rules by default."""
    import inspect

    sig = inspect.signature(discover_interaction_rules)
    assert sig.parameters["rule_type"].default == "optional"


def test_invalid_rule_type_fails_before_fitting():
    X = np.zeros((10, 2))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="rule_type"):
        discover_interaction_rules(X, y, ["a", "b"], rule_type="mandatory")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
