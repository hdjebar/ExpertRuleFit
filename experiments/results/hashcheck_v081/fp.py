import json
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from expertrulefit import ExpertRuleFit

rng = np.random.RandomState(7)
n = 400
X = np.column_stack(
    [rng.normal(size=n), rng.uniform(0, 5, n), rng.normal(size=n), rng.uniform(size=n)]
)
y = (
    (1.2 * (X[:, 0] > 0) + 0.5 * X[:, 1] - 1.5 + rng.normal(scale=0.6, size=n)) > 0
).astype(int)
names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["x0", "x1", "x2", "x3"]
perm = [["x0", "x1", "x2", "x3"].index(f) for f in names]
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
conf = [
    {
        "name": "x1>2.5",
        "evaluate": lambda X, fn: (X[:, fn.index("x1")] > 2.5).astype(float),
        "expected_sign": 1,
    }
]
m = ExpertRuleFit(n_estimators=60, max_rules=20, **kw).fit(
    X[:, perm], y, feature_names=names, confirmatory_rules=conf
)
print(json.dumps(m.fingerprint(X[:, perm]), sort_keys=True))
