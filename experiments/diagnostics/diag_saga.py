"""Is the w_c collapse caused by saga's step size? (collapsing case from FINDINGS §1)"""

import warnings, json, sys
import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imodels import RuleFitClassifier
from expertrulefit.expert_rulefit import build_rule_feature_matrix

rng = np.random.RandomState(42)
n = 500
X = np.column_stack(
    [
        rng.uniform(0, 5, n),
        rng.exponential(30, n),
        rng.lognormal(8, 1.5, n),
        rng.beta(2, 5, n),
        rng.beta(1.5, 8, n),
        rng.uniform(0, 20, n),
    ]
)
logit = (
    0.8 * (X[:, 3] > 0.4)
    + 0.5 * (X[:, 4] > 0.15)
    + 0.1 * (X[:, 0] > 2.5)
    + rng.normal(0, 0.3, n)
    - 0.8
)
y = (logit > 0).astype(float)
Xs = StandardScaler().fit_transform(X)
thr = (2.5 - X[:, 0].mean()) / X[:, 0].std()
conf = (Xs[:, 0] > thr).astype(float)

rf = RuleFitClassifier(
    n_estimators=250, tree_size=4, max_rules=20, random_state=42, include_linear=True
)
rf.fit(Xs, y, feature_names=[f"f{i}" for i in range(6)])
Z = sparse.hstack(
    [build_rule_feature_matrix(rf, Xs), sparse.csc_matrix(conf).T], format="csr"
).toarray()
p = Z.shape[1]
jc = p - 1
print(f"candidate columns: {p} (confirmatory = last)")


def objective(beta, b0, Zw, C, l1):
    m = Zw @ beta + b0
    nll = np.sum(np.logaddexp(0, m) - y * m)
    return C * nll + l1 * np.abs(beta).sum() + 0.5 * (1 - l1) * beta @ beta


def fit(Zw, C, l1, tol, max_iter):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = LogisticRegression(
            solver="saga", C=C, l1_ratio=l1, tol=tol, max_iter=max_iter, random_state=42
        ).fit(Zw, y)
    conv = sum(issubclass(w.category, ConvergenceWarning) for w in rec)
    return m, int(m.n_iter_[0]), conv


C, l1 = 1.0, 0.5  # fixed, so CV choice of C is not a factor
rows = []
for label, wc in [("w_c=1e-8", 1e-8), ("w_c=1e-4", 1e-4), ("unrescaled (w=1)", 1.0)]:
    w = np.ones(p)
    w[jc] = wc
    Zw = Z / np.sqrt(w)
    L = (Zw**2).sum(axis=1).max()  # max squared row norm; saga step ∝ 1/L
    m, it, cw = fit(Zw, C, l1, 1e-4, 10000)
    m_ref, it_ref, cw_ref = fit(
        Zw, C, l1, 1e-12, 2_000_000
    )  # run saga to (near) true optimum
    beta, beta_ref = m.coef_[0], m_ref.coef_[0]
    nz = int((np.abs(beta[:jc]) > 1e-10).sum())
    nz_ref = int((np.abs(beta_ref[:jc]) > 1e-10).sum())
    f, f_ref = (
        objective(beta, m.intercept_[0], Zw, C, l1),
        objective(beta_ref, m_ref.intercept_[0], Zw, C, l1),
    )
    rows.append(
        dict(
            setting=label,
            max_sq_row_norm=float(L),
            n_iter_default=it,
            conv_warn_default=cw,
            nonzero_other_default=nz,
            n_iter_strict=it_ref,
            conv_warn_strict=cw_ref,
            nonzero_other_strict=nz_ref,
            objective_gap=float(f - f_ref),
            max_coef_diff=float(
                np.abs(beta / np.sqrt(w) - beta_ref / np.sqrt(w)).max()
            ),
            conf_coef_default=float(beta[jc] / np.sqrt(w[jc])),
            conf_coef_strict=float(beta_ref[jc] / np.sqrt(w[jc])),
        )
    )
    print(json.dumps(rows[-1]), flush=True)
json.dump(
    rows,
    open(str(__import__("pathlib").Path(__file__).with_name("diag_saga.json")), "w"),
    indent=1,
)
