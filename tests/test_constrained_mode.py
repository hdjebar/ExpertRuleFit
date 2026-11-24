"""C3 — ExpertRuleFit(confirmatory_mode="constrained")."""

import numpy as np
import pytest

from expertrulefit import ExpertRuleFit

FAST = dict(n_estimators=30, max_rules=10, n_bootstrap=2)


@pytest.fixture(scope="module")
def flip_data():
    """x0 = x1 + noise; y driven by +x1 and -(x0 > 0).

    The rule x0 > 0 raises risk marginally (through x1) but lowers it given
    x1: an expert expecting +1 is contradicted by the conditional effect.
    """
    rng = np.random.RandomState(0)
    n = 600
    x1 = rng.normal(size=n)
    x0 = x1 + rng.normal(scale=0.3, size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x0, x1, x2])
    logit = 2.0 * x1 - 1.5 * (x0 > 0) + 0.3 * rng.normal(size=n)
    y = (logit > 0).astype(int)
    assert np.corrcoef(x0 > 0, y)[0, 1] > 0.2  # marginally risk-raising
    return X, y, ["a", "b", "c"]


def rule(sign):
    return [
        {
            "name": "a>0",
            "evaluate": lambda X, fn: (X[:, 0] > 0).astype(float),
            "expected_sign": sign,
        }
    ]


@pytest.fixture(scope="module")
def bound_model(flip_data):
    X, y, fn = flip_data
    with pytest.warns(UserWarning, match="held at 0"):
        m = ExpertRuleFit(confirmatory_mode="constrained", **FAST).fit(
            X, y, feature_names=fn, confirmatory_rules=rule(+1)
        )
    return m


def test_bound_binds_when_conditional_effect_opposes_expert(bound_model):
    st = bound_model.confirmatory_status_[0]
    assert st["bound_active"] is True
    assert st["coefficient"] == 0.0
    assert bound_model.confirmatory_all_active_ is False  # not included (Definition 1)
    info = bound_model.constrained_info_
    assert info["converged"] and info["kkt_residual"] < 1e-6
    assert info["bound_active"] == {"a>0": True}
    assert "BOUND ACTIVE" in bound_model.summary(return_string=True)
    assert bound_model.audit_bundle()["confirmatory_mode"] == "constrained"


def test_free_when_expert_sign_agrees(flip_data):
    X, y, fn = flip_data
    m = ExpertRuleFit(confirmatory_mode="constrained", **FAST).fit(
        X, y, feature_names=fn, confirmatory_rules=rule(-1)
    )
    st = m.confirmatory_status_[0]
    assert st["bound_active"] is False and st["coefficient"] < 0
    assert st["sign_ok"] is True and m.confirmatory_all_active_
    assert m.constrained_info_["converged"]


def test_refit_mode_flags_the_same_case_without_bounding(flip_data):
    X, y, fn = flip_data
    with pytest.warns(UserWarning, match="contradicts"):
        m = ExpertRuleFit(confirmatory_mode="refit", **FAST).fit(
            X, y, feature_names=fn, confirmatory_rules=rule(+1)
        )
    assert m.confirmatory_mode == "refit"
    assert m.confirmatory_status_[0]["coefficient"] < 0
    assert m.confirmatory_status_[0]["bound_active"] is False


def test_constrained_is_deterministic_over_three_runs(flip_data, bound_model):
    X, y, fn = flip_data
    fps = {bound_model.fingerprint(X)["predictions_sha256"]}
    for _ in range(2):
        m = ExpertRuleFit(confirmatory_mode="constrained", **FAST).fit(
            X, y, feature_names=fn, confirmatory_rules=rule(+1)
        )
        fps.add(m.fingerprint(X)["predictions_sha256"])
    assert len(fps) == 1


def test_ridge_option_and_predictions(flip_data):
    X, y, fn = flip_data
    m = ExpertRuleFit(
        confirmatory_mode="constrained", constrained_penalty="ridge", **FAST
    ).fit(X, y, feature_names=fn, confirmatory_rules=rule(-1))
    assert m.constrained_info_["l1_ratio"] == 0.0
    p = m.predict_proba(X)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(1), 1)
    # original-scale logit reproduces predict_proba
    Z = m._build_stable_raw_matrix(X)
    logit = m._get_true_intercept() + Z @ m._get_true_coefficients()
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), p[:, 1], rtol=1e-8, atol=1e-10)


def test_save_load(bound_model, flip_data, tmp_path):
    X, _, _ = flip_data
    path = tmp_path / "c.joblib"
    bound_model.save(path)
    m2 = ExpertRuleFit.load(path, confirmatory_rules=rule(+1))
    assert m2.constrained_info_ == bound_model.constrained_info_
    assert m2.fingerprint(X) == bound_model.fingerprint(X)


@pytest.mark.parametrize(
    "kw",
    [
        dict(confirmatory_mode="exact"),
        dict(constrained_penalty="lasso"),
        dict(confirmatory_mode="constrained", calibration="sigmoid"),
    ],
)
def test_invalid_options(kw):
    X = np.random.RandomState(0).normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError):
        ExpertRuleFit(**FAST, **kw).fit(X, y)


def test_ablation_options_and_timings(flip_data):
    X, y, fn = flip_data
    m = ExpertRuleFit(selection="none", confirmatory_mode="none", **FAST).fit(
        X, y, feature_names=fn, confirmatory_rules=rule(+1)
    )
    assert m.stable_mask_.all()  # no frequency filter
    assert set(m.timings_) == {
        "rule_generation",
        "bootstrap",
        "final_cv",
        "refit_or_constrained",
        "diagnostics",
    }
    assert m.timings_["bootstrap"] < 1.0  # bootstraps skipped
    with pytest.raises(ValueError, match="selection"):
        ExpertRuleFit(selection="cpss", **FAST).fit(X, y)


def test_v080_defaults():
    m = ExpertRuleFit()
    assert m.row_order == "canonical"
    assert m.confirmatory_mode == "constrained"
    assert m.resampling == "subsample"
    assert m.candidates == "rulefit"
    assert m.selection == "none"
    assert m.strict is False


def test_strict_raises_on_bound_active(flip_data):
    from expertrulefit import ConfirmatoryBoundError

    X, y, fn = flip_data
    with pytest.raises(ConfirmatoryBoundError, match="a>0"):
        ExpertRuleFit(strict=True, **FAST).fit(
            X, y, feature_names=fn, confirmatory_rules=rule(+1)
        )


def test_bound_active_listed_in_audit_bundle(bound_model):
    assert bound_model.audit_bundle()["confirmatory_bound_active"] == ["a>0"]


def test_rule_without_expected_sign_is_free(flip_data):
    """No expected_sign: unpenalised and unbounded, so the negative
    conditional effect is estimated, not clipped."""
    X, y, fn = flip_data
    free = [{"name": "a>0", "evaluate": lambda X, fn: (X[:, 0] > 0).astype(float)}]
    m = ExpertRuleFit(**FAST).fit(X, y, feature_names=fn, confirmatory_rules=free)
    st = m.confirmatory_status_[0]
    assert st["coefficient"] < 0 and st["bound_active"] is False
    assert m.confirmatory_all_active_
