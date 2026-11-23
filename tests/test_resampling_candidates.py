"""D4 (resampling without replacement) and D3 (candidate rules from the trees)."""

import numpy as np
import pytest

from expertrulefit import ExpertRuleFit, TreeRuleCandidates
from expertrulefit.expert_rulefit import _resample_indices, eval_rule_on_data

FAST = dict(n_estimators=30, max_rules=10, n_bootstrap=4, selection="bootstrap")


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(5)
    n = 301                                     # odd, unbalanced classes
    X = np.column_stack([rng.normal(size=n), rng.uniform(0, 5, n), rng.normal(size=n)])
    y = ((1.2 * (X[:, 0] > 0) + 0.3 * X[:, 1] - 1.2 + rng.normal(scale=0.6, size=n)) > 0).astype(int)
    return X, y, ["a", "b", "c"]


def test_subsample_draws_are_stratified_complementary_halves(data):
    _, y, _ = data
    draws = _resample_indices(y, 10, 42, "subsample")
    assert len(draws) == 10
    for k in range(5):
        a, b = draws[2 * k], draws[2 * k + 1]
        assert len(np.unique(a)) == len(a) and len(np.unique(b)) == len(b)   # no duplicates
        assert not set(a) & set(b)                                          # complementary
        assert set(a) | set(b) == set(range(len(y)))
        for c in (0, 1):                                                    # stratified
            n_c = (y == c).sum()
            assert abs((y[a] == c).sum() - n_c / 2) <= 1
    again = _resample_indices(y, 10, 42, "subsample")
    assert all(np.array_equal(u, v) for u, v in zip(draws, again))          # deterministic
    assert not np.array_equal(draws[0], draws[2])                           # pairs differ


def test_bootstrap_scheme_kept_for_compatibility(data):
    _, y, _ = data
    draws = _resample_indices(y, 3, 42, "bootstrap")
    assert all(len(d) == len(y) for d in draws)
    assert any(len(np.unique(d)) < len(d) for d in draws)                   # with replacement


def test_resampling_option_changes_the_fit_and_both_are_deterministic(data):
    X, y, fn = data
    sub = [ExpertRuleFit(**FAST).fit(X, y, feature_names=fn).fingerprint(X) for _ in range(2)]
    boot = ExpertRuleFit(resampling="bootstrap", **FAST).fit(X, y, feature_names=fn).fingerprint(X)
    assert sub[0] == sub[1]
    assert boot["predictions_sha256"] != sub[0]["predictions_sha256"]


def test_tree_candidates_skip_rulefit_lasso(data):
    X, y, fn = data
    cand = TreeRuleCandidates(n_estimators=30, tree_size=4, random_state=42).fit(X, y, fn)
    rules = [r.rule for r in cand.rules_without_feature_names_]
    assert rules == sorted(rules)
    cols = np.column_stack([eval_rule_on_data(r, X) for r in rules])
    sup = cols.mean(axis=0)
    assert np.all((sup >= 0.01) & (sup <= 0.99))
    assert len({c.tobytes() for c in cols.T}) == len(rules)                 # no duplicate activations
    assert all(("X_" not in n) for n in cand.rules_)                        # feature names restored

    from imodels import RuleFitClassifier
    rf = RuleFitClassifier(n_estimators=30, tree_size=4, max_rules=10,
                           random_state=42, include_linear=True).fit(X, y, feature_names=fn)
    assert len(rules) > len(rf.rules_without_feature_names_)                # no L1 pre-selection


def test_erf_with_tree_candidates(data):
    X, y, fn = data
    m1 = ExpertRuleFit(candidates="trees", **FAST).fit(X, y, feature_names=fn)
    m2 = ExpertRuleFit(candidates="trees", **FAST).fit(X, y, feature_names=fn)
    assert m1.fingerprint(X) == m2.fingerprint(X)
    assert isinstance(m1.base_rulefit_, TreeRuleCandidates)
    assert m1.audit_bundle()["candidates"] == "trees"


@pytest.mark.parametrize("kw", [dict(resampling="jackknife"), dict(candidates="sirus")])
def test_invalid(kw, data):
    X, y, fn = data
    with pytest.raises(ValueError):
        ExpertRuleFit(**FAST, **kw).fit(X, y)
