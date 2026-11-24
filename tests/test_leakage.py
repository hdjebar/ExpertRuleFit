"""A5 — leakage checks for DualModel stacking and EBM rule discovery.

The base estimators are replaced by recording stand-ins, so the tests are
fast and check only *which rows* each model is fitted on and predicts.
Column 0 of X holds a unique row id.
"""

import numpy as np
import pytest

import expertrulefit.dual_model as dm
from expertrulefit import DualModel, discover_interaction_rules

LOG = []


def _ids(X):
    X = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    return frozenset(np.asarray(X)[:, 0].astype(int).tolist())


class _Recorder:
    """Records fit/predict rows; predicts a constant 0.5."""

    kind = "base"

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, *args, **kwargs):
        self.fit_ids = _ids(X)
        LOG.append((self.kind, "fit", self.fit_ids, self))
        return self

    def predict_proba(self, X):
        LOG.append((self.kind, "predict", _ids(X), self))
        n = len(X)
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class FakeEBM(_Recorder):
    kind = "ebm"


class FakeERF(_Recorder):
    kind = "erf"
    confirmatory_all_active_ = True


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    n = 100
    X = np.column_stack([np.arange(n), rng.normal(size=n), rng.normal(size=n)])
    y = (X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y, ["row_id", "a", "b"]


@pytest.fixture
def fakes(monkeypatch):
    import interpret.glassbox

    LOG.clear()
    monkeypatch.setattr(interpret.glassbox, "ExplainableBoostingClassifier", FakeEBM)
    monkeypatch.setattr(dm, "ExpertRuleFit", FakeERF)
    yield LOG
    LOG.clear()


def _fold_models(log, kind, n_folds):
    fitted = [e[3] for e in log if e[0] == kind and e[1] == "fit"]
    assert len(fitted) == n_folds + 1, "one model per fold + one final refit"
    return fitted[:n_folds], fitted[-1]


@pytest.mark.parametrize("kind", ["ebm", "erf"])
def test_stacking_fold_models_never_see_their_validation_rows(fakes, data, kind):
    X, y, fn = data
    DualModel(meta_cv=5).fit(X, y, feature_names=fn)
    folds, final = _fold_models(fakes, kind, 5)
    predicted = set()
    for model in folds:
        preds = [e[2] for e in fakes if e[1] == "predict" and e[3] is model]
        assert len(preds) == 1, "each fold model predicts its validation fold once"
        val = preds[0]
        assert not (model.fit_ids & val), "fold model predicted rows it was fitted on"
        assert model.fit_ids | val == set(range(len(y)))
        predicted |= val
    assert predicted == set(range(len(y))), (
        "out-of-fold predictions cover every row once"
    )
    assert final.fit_ids == set(range(len(y)))


def test_rule_discovery_runs_on_training_folds_only(fakes, data):
    X, y, fn = data
    calls = []

    def discover(X_tr, y_tr, names):
        calls.append(_ids(X_tr))
        return [
            {
                "name": f"r{len(calls)}",
                "evaluate": lambda X, f: (np.asarray(X)[:, 1] > 0).astype(float),
            }
        ]

    DualModel(meta_cv=5).fit(X, y, feature_names=fn, rule_discovery=discover)
    folds, final = _fold_models(fakes, "erf", 5)
    assert len(calls) == 6
    for ids, model in zip(calls[:5], folds):
        assert ids == model.fit_ids, (
            "rules for a fold are discovered on that fold's training rows"
        )
        val = [e[2] for e in fakes if e[1] == "predict" and e[3] is model][0]
        assert not (ids & val)
    assert calls[5] == set(range(len(y)))


def test_discover_interaction_rules_fits_ebm_on_given_rows_only(monkeypatch, data):
    """The EBM inside discover_interaction_rules sees exactly the rows passed."""
    import interpret.glassbox

    seen = {}

    class RecordingEBM(FakeEBM):
        term_features_ = [(1,), (2,), (1, 2)]
        term_names_ = ["a", "b", "a & b"]

        def term_importances(self):
            return np.array([0.1, 0.1, 0.5])

    def fit(self, X, y, *a, **k):
        seen["ids"] = _ids(X)
        return self

    RecordingEBM.fit = fit
    monkeypatch.setattr(
        interpret.glassbox, "ExplainableBoostingClassifier", RecordingEBM
    )
    X, y, fn = data
    train = np.arange(0, 70)
    rules, _ = discover_interaction_rules(X[train], y[train], fn, top_k=1)
    assert seen["ids"] == set(train.tolist())
    assert len(rules) == 1 and rules[0]["category"] == "optional"
