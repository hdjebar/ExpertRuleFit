"""Tests for input validation and eval_rule_on_data edge cases."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest

from expertrulefit import ExpertRuleFit
from expertrulefit.expert_rulefit import eval_rule_on_data


class TestEvalRuleOnData:
    """Tests for the regex-based rule evaluation function."""

    def test_simple_greater_than(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = eval_rule_on_data("X_0 > 2.0", X)
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_simple_less_equal(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = eval_rule_on_data("X_0 <= 3.0", X)
        np.testing.assert_array_equal(result, [1.0, 1.0, 0.0])

    def test_simple_greater_equal(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = eval_rule_on_data("X_0 >= 3.0", X)
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_simple_less_than(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = eval_rule_on_data("X_1 < 4.0", X)
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_compound_rule(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]])
        result = eval_rule_on_data("X_0 > 2.0 and X_1 <= 4.0", X)
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_negative_threshold(self):
        X = np.array([[-2.0], [0.0], [2.0]])
        result = eval_rule_on_data("X_0 > -1.0", X)
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_scientific_notation(self):
        X = np.array([[1e-5], [1e-3], [1.0]])
        result = eval_rule_on_data("X_0 > 1e-4", X)
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    def test_out_of_bounds_column(self):
        X = np.array([[1.0, 2.0]])
        result = eval_rule_on_data("X_5 > 0.0", X)
        np.testing.assert_array_equal(result, [0.0])

    def test_returns_float64(self):
        X = np.array([[1.0, 2.0]])
        result = eval_rule_on_data("X_0 > 0.0", X)
        assert result.dtype == np.float64


class TestInputValidation:
    """Tests for ExpertRuleFit input validation."""

    def test_1d_X_raises(self):
        erf = ExpertRuleFit()
        with pytest.raises(ValueError, match="2-dimensional"):
            erf.fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))

    def test_mismatched_samples_raises(self):
        erf = ExpertRuleFit()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="inconsistent sample counts"):
            erf.fit(X, y)

    def test_empty_X_raises(self):
        erf = ExpertRuleFit()
        X = np.empty((0, 5))
        y = np.empty(0)
        with pytest.raises(ValueError, match="at least one sample"):
            erf.fit(X, y)

    def test_expert_rule_missing_name_raises(self):
        erf = ExpertRuleFit()
        X = np.random.RandomState(0).randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        bad_rules = [{"evaluate": lambda X, fn: np.ones(len(X))}]
        with pytest.raises(ValueError, match="missing required key 'name'"):
            erf.fit(X, y, confirmatory_rules=bad_rules)

    def test_expert_rule_missing_evaluate_raises(self):
        erf = ExpertRuleFit()
        X = np.random.RandomState(0).randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        bad_rules = [{"name": "test"}]
        with pytest.raises(ValueError, match="missing required key 'evaluate'"):
            erf.fit(X, y, confirmatory_rules=bad_rules)

    def test_expert_rule_non_callable_raises(self):
        erf = ExpertRuleFit()
        X = np.random.RandomState(0).randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        bad_rules = [{"name": "test", "evaluate": "not_callable"}]
        with pytest.raises(ValueError, match="must be callable"):
            erf.fit(X, y, confirmatory_rules=bad_rules)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
