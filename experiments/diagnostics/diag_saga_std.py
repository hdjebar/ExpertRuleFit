"""Same fixed-C saga test as diag_saga.py, but on standardised columns (library default since 0.7)."""
import warnings, json
import numpy as np
src = open(__import__("pathlib").Path(__file__).with_name("diag_saga.py")).read()
exec(src.split("C, l1 = 1.0, 0.5")[0])          # builds Z, y, fit(), objective()
nlin = 6
mean, sd = Z.mean(0), Z.std(0); sd[sd < 1e-12] = 1
Zs = Z / sd; Zs[:, :nlin] -= (mean / sd)[:nlin]
C, l1 = 1.0, 0.5; rows = []
for wc in [1e-8, 1e-4, 1e-2, 1e-1, 1.0]:
    w = np.ones(p); w[jc] = wc; Zw = Zs / np.sqrt(w)
    m, it, cw = fit(Zw, C, l1, 1e-4, 10000)
    mr, itr, cwr = fit(Zw, C, l1, 1e-12, 2_000_000)
    b, br = m.coef_[0], mr.coef_[0]
    rows.append(dict(w_c=wc, max_sq_row_norm=float((Zw**2).sum(1).max()), n_iter_default=it, conv_warn_default=cw,
                     n_iter_strict=itr, conv_warn_strict=cwr,
                     objective_gap=float(objective(b, m.intercept_[0], Zw, C, l1) - objective(br, mr.intercept_[0], Zw, C, l1)),
                     max_coef_diff=float(np.abs((b - br) / np.sqrt(w) / sd).max()),
                     conf_coef_default=float(b[jc]/np.sqrt(w[jc])/sd[jc]), conf_coef_strict=float(br[jc]/np.sqrt(w[jc])/sd[jc])))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(str(__import__("pathlib").Path(__file__).with_name("diag_saga_std.json")), "w"), indent=1)
