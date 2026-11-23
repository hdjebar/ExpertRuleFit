"""C3 — block-penalised, sign-bounded logistic solver."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from expertrulefit.constrained import solve_block_logistic

INF = np.inf


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(0)
    n, p = 400, 6
    Z = rng.normal(size=(n, p))
    beta = np.array([1.0, -0.8, 0.0, 0.5, 0.0, 0.0])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(Z @ beta - 0.3)))).astype(float)
    return Z, y


def test_matches_sklearn_elasticnet_without_bounds(data):
    Z, y = data
    n = len(y)
    C, l1 = 0.5, 0.5
    sk = LogisticRegression(C=C, l1_ratio=l1, solver="saga", tol=1e-12,
                            max_iter=200000, penalty="elasticnet").fit(Z, y)
    res = solve_block_logistic(Z, y, lam=1 / (C * n), alpha=l1, w=np.ones(6),
                               lower=np.full(6, -INF), upper=np.full(6, INF), tol=1e-12)
    assert res["converged"] and res["kkt_residual"] < 1e-6
    np.testing.assert_allclose(res["coef"], sk.coef_[0], atol=1e-4)
    assert res["intercept"] == pytest.approx(sk.intercept_[0], abs=1e-4)


def test_unpenalised_column_matches_plain_logistic(data):
    Z, y = data
    res = solve_block_logistic(Z, y, lam=0.0, alpha=0.5, w=np.zeros(6),
                               lower=np.full(6, -INF), upper=np.full(6, INF), tol=1e-12)
    sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(Z, y)
    np.testing.assert_allclose(res["coef"], sk.coef_[0], atol=1e-4)


def test_sign_bound_binds_when_data_disagree(data):
    """Column 1 has a negative effect; forcing beta_1 >= 0 puts it on the bound."""
    Z, y = data
    w = np.ones(6); w[1] = 0.0
    lo = np.full(6, -INF); lo[1] = 0.0
    res = solve_block_logistic(Z, y, lam=0.01, alpha=0.5, w=w, lower=lo,
                               upper=np.full(6, INF))
    assert res["converged"] and res["kkt_residual"] < 1e-6
    assert res["coef"][1] == 0.0
    assert res["bound_active"][1] and res["bound_active"].sum() == 1


def test_free_case_equals_unbounded_solution(data):
    """Column 0 has a positive effect; the bound beta_0 >= 0 is inactive."""
    Z, y = data
    w = np.ones(6); w[0] = 0.0
    lo = np.full(6, -INF); lo[0] = 0.0
    bounded = solve_block_logistic(Z, y, 0.01, 0.5, w, lo, np.full(6, INF))
    free = solve_block_logistic(Z, y, 0.01, 0.5, w, np.full(6, -INF), np.full(6, INF))
    assert not bounded["bound_active"].any()
    assert bounded["coef"][0] > 0
    np.testing.assert_allclose(bounded["coef"], free["coef"], atol=1e-7)


def test_ridge_rest_gives_unique_solution_under_collinearity():
    """A penalised column duplicates the unpenalised one; with alpha < 1 the
    optimum is unique and independent of the starting point."""
    rng = np.random.RandomState(1)
    n = 300
    c = (rng.normal(size=n) > 0).astype(float)
    c = (c - c.mean()) / c.std()
    Z = np.column_stack([c, c, rng.normal(size=n)])        # col 1 == col 0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-c))).astype(float)
    w = np.array([0.0, 1.0, 1.0])
    lo, hi = np.array([0.0, -INF, -INF]), np.full(3, INF)
    a = solve_block_logistic(Z, y, 0.05, 0.0, w, lo, hi, tol=1e-12)
    b = solve_block_logistic(Z, y, 0.05, 0.0, w, lo, hi, tol=1e-12,
                             beta_init=np.array([2.0, -1.0, 0.5]))
    np.testing.assert_allclose(a["coef"], b["coef"], atol=1e-6)
    # ridge pushes the duplicate's coefficient to zero: the unpenalised column takes it all
    assert abs(a["coef"][1]) < 1e-5 and a["coef"][0] > 0


def test_deterministic(data):
    Z, y = data
    w = np.ones(6); w[1] = 0.0
    lo = np.full(6, -INF); lo[1] = 0.0
    r1 = solve_block_logistic(Z, y, 0.01, 0.5, w, lo, np.full(6, INF))
    r2 = solve_block_logistic(Z, y, 0.01, 0.5, w, lo, np.full(6, INF))
    assert r1["coef"].tobytes() == r2["coef"].tobytes()
    assert r1["n_iter"] == r2["n_iter"]
