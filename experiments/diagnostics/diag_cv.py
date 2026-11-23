"""Pipeline-equivalent final LogisticRegressionCV (cv=5, saga) on the collapsing case."""
import warnings, json, time
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
exec(open(__import__("pathlib").Path(__file__).with_name("diag_saga.py")).read().split("def objective")[0])  # rebuild Z, y
out = []
for label, wc in [("w_c=1e-8", 1e-8), ("w_c=1e-4", 1e-4), ("unrescaled (w=1)", 1.0)]:
    w = np.ones(p); w[jc] = wc; Zw = Z / np.sqrt(w)
    t = time.time()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = LogisticRegressionCV(solver="saga", l1_ratios=[0.1,0.3,0.5,0.7,0.9,1.0], cv=5,
                                 random_state=42, tol=1e-4, max_iter=10000).fit(Zw, y)
    cw = sum(issubclass(r.category, ConvergenceWarning) for r in rec)
    beta = m.coef_[0] / np.sqrt(w)
    it = np.asarray(m.n_iter_).ravel()
    out.append(dict(setting=label, C=float(np.ravel(m.C_)[0]), l1_ratio=float(np.ravel(m.l1_ratio_)[0]),
                    n_iter_min=int(it.min()), n_iter_median=float(np.median(it)), n_iter_max=int(it.max()),
                    conv_warnings=cw, nonzero_other=int((np.abs(beta[:jc])>1e-10).sum()),
                    conf_coef=float(beta[jc]), seconds=round(time.time()-t,1)))
    print(json.dumps(out[-1]), flush=True)
json.dump(out, open(str(__import__("pathlib").Path(__file__).with_name("diag_cv.json")),"w"), indent=1)
