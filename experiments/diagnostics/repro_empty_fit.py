"""Does canonical row order + unshuffled CV select an empty model?

Synthetic data shaped like German Credit: column "a1" (sorted first) is a
strong 4-level ordinal predictor, plus weaker numeric columns. Fits
ExpertRuleFit with defaults on several subsamples and reports the number
of active terms; then repeats with the final-model CV folds shuffled with
a fixed seed (monkeypatched) to test the proposed fix.
"""

import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import expertrulefit.expert_rulefit as er
from expertrulefit import ExpertRuleFit

warnings.filterwarnings("ignore")
mode = sys.argv[1] if len(sys.argv) > 1 else "default"

if mode == "shuffled":
    orig = er._make_logistic_cv

    def patched(**kw):
        k = kw.pop("cv", 5)
        kw["cv"] = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        return orig(**kw)

    er._make_logistic_cv = patched

rng = np.random.RandomState(0)
n = 1000
a1 = rng.choice([0.0, 1.0, 2.0, 3.0], size=n, p=[0.27, 0.27, 0.06, 0.40])
X = np.column_stack(
    [
        (a1 - a1.mean()) / a1.std(),
        rng.normal(size=n),
        rng.normal(size=n),
        rng.normal(size=n),
        rng.choice([0.0, 1.0], size=n),
    ]
)
eta = -1.3 * (a1 == 3) + 0.4 * (a1 == 0) + 0.5 * X[:, 1] - 0.3 * X[:, 2] - 0.9
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
names = ["a1", "b2", "c3", "d4", "e5"]
te = np.arange(800, 1000)
for k in range(6):
    r = np.random.RandomState(100 + k)
    tr = np.sort(r.choice(800, 720, replace=False))
    m = ExpertRuleFit(n_estimators=100, max_rules=30).fit(
        X[tr], y[tr], feature_names=names
    )
    coefs = m._get_true_coefficients()
    active = int(np.sum(np.abs(coefs) > 1e-10))
    auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    C = m.constrained_info_["C"] if m.constrained_info_ else None
    print(f"{mode} k={k}: active terms {active}, AUC {auc:.3f}, C {C}")
