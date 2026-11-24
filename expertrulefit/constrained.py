"""Block-penalised logistic regression with sign-bounded unpenalised columns (C3).

Solves, for standardised columns Z (n x p) and labels y in {0, 1}::

    minimise_{b0, beta}  (1/n) sum_i log(1 + exp(-s_i (b0 + z_i^T beta)))
                         + lam * sum_j w_j * (alpha * |beta_j| + (1 - alpha)/2 * beta_j^2)
    subject to           lo_j <= beta_j <= hi_j

with s_i = 2 y_i - 1. Columns with w_j = 0 (confirmatory rules) are not
penalised; their bounds come from ``expected_sign`` ([0, inf) for +1,
(-inf, 0] for -1, unbounded otherwise). Other columns have infinite bounds.

The problem is convex. With alpha < 1 and the unpenalised columns plus the
intercept of full column rank, it is strictly convex and the solution is
unique; with alpha = 1 and collinear columns it may not be.

Solver: FISTA (accelerated proximal gradient) with a fixed step 1/L,
L = ||[1, Z]||_2^2 / (4 n), and gradient-based adaptive restart. The
proximal step is exact: elastic-net shrinkage for penalised columns,
projection onto the box for bounded ones. No randomness is involved, so the
result is deterministic for a given input and BLAS configuration.
"""

import numpy as np
from scipy.special import expit

__all__ = ["solve_block_logistic", "LinearLogit"]


def _loss_grad(b0, beta, Z, s, n):
    m = b0 + Z @ beta
    sm = s * m
    loss = np.logaddexp(0.0, -sm).sum() / n
    r = -s * expit(-sm) / n  # d loss / d m
    return loss, float(r.sum()), Z.T @ r


def _penalty(beta, lam, w, alpha):
    return lam * float(np.sum(w * (alpha * np.abs(beta) + 0.5 * (1 - alpha) * beta**2)))


def _prox(v, t, lam, w, alpha, lo, hi):
    """Prox of t * penalty followed by the box (exact: penalty is 0 where boxed)."""
    thr = t * lam * w * alpha
    out = (
        np.sign(v)
        * np.maximum(np.abs(v) - thr, 0.0)
        / (1.0 + t * lam * w * (1 - alpha))
    )
    return np.clip(out, lo, hi)


def solve_block_logistic(
    Z,
    y,
    lam,
    alpha,
    w,
    lower,
    upper,
    *,
    tol=1e-9,
    max_iter=200_000,
    b0_init=0.0,
    beta_init=None,
):
    """Solve the block-penalised, box-constrained logistic problem.

    Parameters
    ----------
    Z : ndarray (n, p)       standardised design (dense)
    y : ndarray (n,)         labels in {0, 1}
    lam : float              overall penalty (>= 0)
    alpha : float            l1 share of the elastic net, in [0, 1]
    w : ndarray (p,)         per-column penalty factors (0 = unpenalised)
    lower, upper : ndarray   box bounds per column (use -inf / inf for none)

    Returns
    -------
    result : dict with ``intercept``, ``coef``, ``n_iter``, ``converged``,
        ``objective``, ``kkt_residual`` (max violation of the optimality
        conditions, scaled like a gradient) and ``bound_active`` (bool per
        column: at a finite bound with the gradient pushing outward).
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = Z.shape
    s = 2.0 * y - 1.0
    w = np.asarray(w, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)

    A = np.column_stack([np.ones(n), Z])
    L = float(np.linalg.norm(A, 2) ** 2 / (4.0 * n)) + lam * float(
        np.max(w * (1 - alpha), initial=0.0)
    )
    t = 1.0 / max(L, 1e-12)

    b0 = float(b0_init)
    beta = (
        np.zeros(p)
        if beta_init is None
        else np.clip(np.asarray(beta_init, float), lo, hi)
    )
    yb0, ybeta = b0, beta.copy()
    theta = 1.0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        _, g0, g = _loss_grad(yb0, ybeta, Z, s, n)
        nb0 = yb0 - t * g0
        nbeta = _prox(ybeta - t * g, t, lam, w, alpha, lo, hi)
        step = max(abs(nb0 - b0), float(np.max(np.abs(nbeta - beta), initial=0.0)))
        scale = max(1.0, abs(nb0), float(np.max(np.abs(nbeta), initial=0.0)))
        # adaptive restart (O'Donoghue & Candes): momentum pointing uphill
        if (yb0 - nb0) * (nb0 - b0) + float((ybeta - nbeta) @ (nbeta - beta)) > 0:
            theta = 1.0
        ntheta = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
        mom = (theta - 1.0) / ntheta
        yb0 = nb0 + mom * (nb0 - b0)
        ybeta = nbeta + mom * (nbeta - beta)
        b0, beta, theta = nb0, nbeta, ntheta
        if step <= tol * scale:
            converged = True
            break

    loss, g0, g = _loss_grad(b0, beta, Z, s, n)
    kkt, bound_active = _kkt(beta, g0, g, lam, alpha, w, lo, hi)
    return {
        "intercept": b0,
        "coef": beta,
        "n_iter": it,
        "converged": converged,
        "objective": loss + _penalty(beta, lam, w, alpha),
        "kkt_residual": kkt,
        "bound_active": bound_active,
    }


def _kkt(beta, g0, g, lam, alpha, w, lo, hi, atol=1e-10):
    """Optimality residual and active bounds.

    The residual is the prox-gradient mapping with unit step,
    max(|g0|, max_j |beta_j - prox(beta - g)_j|), which is zero exactly at
    a solution. A bound is *active* when the coefficient sits on it and
    the unconstrained proximal step would leave the box, i.e. the data
    push the coefficient past the expert's sign.
    """
    free = _prox(beta - g, 1.0, lam, w, alpha, -np.inf, np.inf)
    boxed = np.clip(free, lo, hi)
    resid = max(abs(g0), float(np.max(np.abs(beta - boxed), initial=0.0)))
    bound_active = ((beta <= lo + atol) & (free < lo - atol)) | (
        (beta >= hi - atol) & (free > hi + atol)
    )
    return resid, bound_active


class LinearLogit:
    """Minimal fitted logistic model (sklearn-like predict / predict_proba)."""

    def __init__(self, intercept, coef, C, l1_ratio, info=None):
        self.intercept_ = np.array([float(intercept)])
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.classes_ = np.array([0.0, 1.0])
        self.C = C
        self.l1_ratio = l1_ratio
        self.info_ = info or {}

    def decision_function(self, X):
        m = X @ self.coef_[0] if not hasattr(X, "toarray") else X.dot(self.coef_[0])
        return np.asarray(m).ravel() + self.intercept_[0]

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(np.float64)
