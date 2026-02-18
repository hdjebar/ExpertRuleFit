"""Unit tests for DualModel (EBM + ExpertRuleFit stacking)."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from expertrulefit import DualModel


@pytest.fixture
def credit_data():
    """Generate synthetic credit scoring data."""
    rng = np.random.RandomState(42)
    n = 400
    X = np.column_stack([
        rng.uniform(0, 5, n),
        rng.exponential(30, n),
        rng.beta(2, 5, n),
        rng.beta(1.5, 8, n),
        rng.uniform(0, 20, n),
    ])
    feature_names = ["country_risk", "payment_delay", "cash_ratio",
                     "night_tx_ratio", "tenure"]
    logit = (0.8 * (X[:, 2] > 0.4) + 0.5 * (X[:, 3] > 0.15)
             + 0.1 * (X[:, 0] > 2.5) + rng.normal(0, 0.3, n) - 0.8)
    y = (logit > 0).astype(int)
    return X, y, feature_names


def test_dual_model_fit_predict(credit_data):
    """Test basic DualModel fit/predict cycle."""
    X, y, fn = credit_data

    dm = DualModel(
        ebm_params=dict(interactions=3, outer_bags=5, n_jobs=1),
        erf_params=dict(max_rules=20, n_bootstrap=3),
        meta_cv=3,
    )
    dm.fit(X, y, feature_names=fn)

    proba = dm.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)

    labels = dm.predict(X)
    assert set(np.unique(labels)).issubset({0, 1})


def test_dual_model_meta_weights(credit_data):
    """Test that meta-classifier has interpretable weights."""
    X, y, fn = credit_data

    dm = DualModel(
        ebm_params=dict(interactions=3, outer_bags=5, n_jobs=1),
        erf_params=dict(max_rules=20, n_bootstrap=3),
        meta_cv=3,
    )
    dm.fit(X, y, feature_names=fn)

    assert hasattr(dm, "meta_weights_")
    assert "ebm_weight" in dm.meta_weights_
    assert "erf_weight" in dm.meta_weights_
    assert "intercept" in dm.meta_weights_


def test_dual_model_with_confirmatory(credit_data):
    """Test DualModel with confirmatory rules."""
    X, y, fn = credit_data

    confirmatory = [
        {
            "name": "High cash ratio",
            "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
        },
    ]

    dm = DualModel(
        ebm_params=dict(interactions=3, outer_bags=5, n_jobs=1),
        erf_params=dict(max_rules=20, n_bootstrap=3),
        meta_cv=3,
    )
    dm.fit(X, y, feature_names=fn, confirmatory_rules=confirmatory)

    assert dm.confirmatory_all_active_

    proba = dm.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


def test_dual_model_explain(credit_data):
    """Test per-sample explanation."""
    X, y, fn = credit_data

    dm = DualModel(
        ebm_params=dict(interactions=3, outer_bags=5, n_jobs=1),
        erf_params=dict(max_rules=20, n_bootstrap=3),
        meta_cv=3,
    )
    dm.fit(X, y, feature_names=fn)

    explanations = dm.explain(X[:2], top_n=3)
    assert len(explanations) == 2

    exp = explanations[0]
    assert "ebm_score" in exp
    assert "erf_score" in exp
    assert "final_score" in exp
    assert "meta_weights" in exp
    assert "ebm_top_contributions" in exp
    assert "erf_active_rules" in exp
    assert len(exp["ebm_top_contributions"]) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
