"""B3 — Sensitivity of ExpertRuleFit to the penalty weights w_c and w_o.

For each dataset: one stratified 80/20 split (seed 42, as in the benchmark),
preprocessing fitted on the training split without scaling (ExpertRuleFit
standardises internally, so expert rules use raw units), one confirmatory
and one optional expert rule (see EXPERT_RULES), and one ExpertRuleFit fit
per grid point:

    w_c in {1e-8, 1e-6, 1e-4, 1e-2, 1}   x   w_o in {0.1, 0.3, 1}

Reported per fit: test AUC and Brier score; rule set and its Jaccard
similarity to the default fit (w_c = 1e-4, w_o = 0.3); number of active
terms; confirmatory coefficient and sign check; optional rule kept or not;
absorption diagnostics (max |corr|, rules displaced, VIF); final-model C and
saga iteration counts; and whether saga raised ConvergenceWarnings.

The expert rules are illustrative: standard credit-risk criteria with
thresholds checked against the data, not rules supplied by domain experts.

Every fit is deterministic (fixed internal seeds, single-threaded). Fits are
appended to <out>/fits.jsonl and skipped on rerun.

Usage:
    python experiments/fetch_data.py
    python experiments/sensitivity.py --jobs 2
    python experiments/sensitivity.py --summarize-only
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perturbation import (DATASETS, Preprocessor, jaccard, load_dataset,  # noqa: E402
                          normalise_rule)

W_C_GRID = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
W_O_GRID = (0.1, 0.3, 1.0)
DEFAULT = (1e-4, 0.3)

# (feature, op, value, label); value is a category label for encoded columns.
EXPERT_RULES = {
    "german": {
        "confirmatory": ("Attribute1", "==", "A11", "Checking account < 0 DM"),
        "optional": ("Attribute2", ">", 24, "Duration > 24 months"),
    },
    "taiwan": {
        "confirmatory": ("X6", ">=", 2, "Last repayment >= 2 months late"),
        "optional": ("X1", "<", 50000, "Credit limit < 50,000 NT$"),
    },
    "hmeq": {
        "confirmatory": ("DELINQ", ">", 0, "Any delinquent credit line"),
        "optional": ("DEBTINC", ">", 43, "Debt-to-income > 43%"),
    },
}
_OPS = {"==": np.equal, ">": np.greater, ">=": np.greater_equal, "<": np.less}


def make_rule(pre, spec, expected_sign=+1):
    feature, op, value, label = spec
    j = pre.column(feature)
    v = pre.code(feature, value) if isinstance(value, str) else float(value)
    fn = _OPS[op]
    return {
        "name": label,
        "evaluate": lambda X, names, j=j, v=v, fn=fn: fn(X[:, j], v).astype(float),
        "expected_sign": expected_sign,
    }


def run_fit(task):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    from expertrulefit import ExpertRuleFit

    ds, w_c, w_o = task["dataset"], task["w_c"], task["w_o"]
    df, y = load_dataset(ds, task["data_dir"])
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      random_state=42, stratify=y)
    pre = Preprocessor(scale=False).fit(df.iloc[idx_tr])
    X_tr, X_te = pre.transform(df.iloc[idx_tr]), pre.transform(df.iloc[idx_te])
    y_tr, y_te = y[idx_tr], y[idx_te]
    conf = make_rule(pre, EXPERT_RULES[ds]["confirmatory"])
    opt = make_rule(pre, EXPERT_RULES[ds]["optional"])

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = ExpertRuleFit(confirmatory_penalty=w_c, optional_penalty=w_o,
                          tol=task["tol"])
        m.fit(X_tr, y_tr, feature_names=pre.feature_names,
              confirmatory_rules=[conf], optional_rules=[opt])
    fit_s = time.perf_counter() - t0
    conv = sum(issubclass(w.category, ConvergenceWarning) for w in rec)

    p = m.predict_proba(X_te)[:, 1]
    rules = sorted(normalise_rule(r) for r in m.get_selected_rules())
    st = m.confirmatory_status_[0]
    d = m.diagnose_confirmatory()[0]
    fm = m.final_model_
    n_iter = np.asarray(getattr(fm, "n_iter_", [np.nan])).ravel()
    return {
        "dataset": ds, "w_c": w_c, "w_o": w_o,
        "auc": float(roc_auc_score(y_te, p)),
        "brier": float(brier_score_loss(y_te, p)),
        "rules": rules, "n_active": len(rules),
        "n_stable": int(m.n_stable_rules_),
        "confirmatory_coef": st["coefficient"],
        "confirmatory_sign_ok": st["sign_ok"],
        "optional_active": f"optional:{opt['name']}" in m.get_selected_rules(),
        "optional_coef": next((r["coefficient"] for r in m.get_rule_importance()
                               if r["name"] == f"optional:{opt['name']}"), 0.0),
        "max_abs_corr": d["max_abs_corr"], "most_correlated": d["most_correlated"],
        "n_correlated": d["n_correlated"], "n_displaced": d["n_displaced"],
        "vif": d["vif"], "absorption_ratio": d["absorption_ratio"],
        "final_model": type(fm).__name__,
        "final_C": float(m._final_hyperparams()[0]),
        "final_n_iter_median": float(np.nanmedian(n_iter)),
        "final_n_iter_min": float(np.nanmin(n_iter)),
        "convergence_warnings": int(conv),
        "fit_seconds": round(fit_s, 1),
    }


def load_fits(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fits = load_fits(out_dir / "fits.jsonl")
    if not fits:
        return None
    df = pd.DataFrame(fits)
    rows = []
    for ds, g in df.groupby("dataset", sort=False):
        ref = g[(g.w_c == DEFAULT[0]) & (g.w_o == DEFAULT[1])]
        ref_rules = set(ref.rules.iloc[0]) if len(ref) else None
        for r in g.itertuples():
            rows.append({**{k: getattr(r, k) for k in (
                "dataset", "w_c", "w_o", "auc", "brier", "n_active", "n_stable",
                "confirmatory_coef", "confirmatory_sign_ok", "optional_active",
                "optional_coef", "max_abs_corr", "n_correlated", "n_displaced",
                "vif", "final_C", "final_n_iter_median", "convergence_warnings",
                "fit_seconds")},
                "jaccard_vs_default": jaccard(set(r.rules), ref_rules)
                if ref_rules is not None else np.nan})
    summ = pd.DataFrame(rows).sort_values(["dataset", "w_o", "w_c"])
    summ.to_csv(out_dir / "summary.csv", index=False)

    datasets = list(summ.dataset.unique())
    fig, axes = plt.subplots(3, len(datasets), figsize=(4 * len(datasets), 8),
                             squeeze=False, sharex=True)
    for c, ds in enumerate(datasets):
        g = summ[summ.dataset == ds]
        for w_o, h in g.groupby("w_o"):
            h = h.sort_values("w_c")
            for r_, col in enumerate(("auc", "jaccard_vs_default", "n_active")):
                axes[r_, c].plot(h.w_c, h[col], marker="o", label=f"w_o={w_o:g}")
        axes[0, c].set_title(ds)
        for r_, lab in enumerate(("test AUC", "Jaccard vs default", "active terms")):
            axes[r_, c].set_xscale("log")
            axes[r_, c].axvline(DEFAULT[0], ls=":", c="grey", lw=0.8)
            if c == 0:
                axes[r_, c].set_ylabel(lab)
        axes[2, c].set_xlabel("w_c (confirmatory penalty weight)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity.png", dpi=150)
    plt.close(fig)

    md = ["| Dataset | w_c | w_o | AUC | Brier | Active | Jaccard vs default | Conf. coef | Sign OK "
          "| Optional kept | Displaced | Final C | saga iters (median) | ConvergenceWarnings | Fit s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        md.append(f"| {r.dataset} | {r.w_c:g} | {r.w_o:g} | {r.auc:.3f} | {r.brier:.4f} | "
                  f"{r.n_active} | {r.jaccard_vs_default:.2f} | {r.confirmatory_coef:+.3f} | "
                  f"{r.confirmatory_sign_ok} | {r.optional_active} | {r.n_displaced} | "
                  f"{r.final_C:.3g} | {r.final_n_iter_median:.0f} | "
                  f"{r.convergence_warnings} | {r.fit_seconds:.0f} |")
    (out_dir / "table.md").write_text("\n".join(md) + "\n")
    return summ


def main():
    ap = argparse.ArgumentParser(description="B3 penalty-weight sensitivity")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--w-c", nargs="+", type=float, default=list(W_C_GRID))
    ap.add_argument("--w-o", nargs="+", type=float, default=list(W_O_GRID))
    ap.add_argument("--tol", type=float, default=1e-4, help="ExpertRuleFit tol (library default)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "results" / "sensitivity"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        done = {(f["dataset"], f["w_c"], f["w_o"]) for f in load_fits(out / "fits.jsonl")}
        # default point first, so a partial run still has its reference
        grid = sorted(((d, wc, wo) for d in args.datasets for wc in args.w_c for wo in args.w_o),
                      key=lambda t: (args.datasets.index(t[0]), (t[1], t[2]) != DEFAULT))
        tasks = [{"dataset": d, "w_c": wc, "w_o": wo, "tol": args.tol,
                  "data_dir": args.data_dir} for d, wc, wo in grid if (d, wc, wo) not in done]
        print(f"{len(tasks)} fits to run ({args.jobs} process(es))", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_fit, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                with open(out / "fits.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  {i}/{len(tasks)} {rec['dataset']} w_c={rec['w_c']:g} w_o={rec['w_o']:g} "
                      f"AUC={rec['auc']:.3f} active={rec['n_active']} {rec['fit_seconds']}s "
                      f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    summ = summarize(out)
    if summ is not None:
        print((out / "table.md").read_text())


if __name__ == "__main__":
    main()
