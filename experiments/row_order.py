"""A6 — what changes when the training rows are shuffled (German Credit).

Fits ExpertRuleFit (library defaults, no expert rules) on the Table 1
training split in its original row order and in ``--n-perm`` random
permutations, with ``row_order="as_given"`` and ``row_order="canonical"``.
Reports rule-set Jaccard against the original order (rule names without
ExpertRuleFit's positional ``rule[i]:`` prefix), max |difference| of
test probabilities, test AUC, and whether the SHA-256 fingerprints match.

    docker run --rm --network none -v "$PWD":/work -w /work \\
        -e PYTHONPATH=/work erf-pinned python experiments/row_order.py --jobs 2
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def run(task):
    import warnings

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    import table1
    from expertrulefit import ExpertRuleFit
    from expertrulefit_validation import preprocess

    warnings.filterwarnings("ignore")
    X, y = table1._load(task["dataset"], task["data_dir"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_te = preprocess(X_tr, X_te)
    if task["perm"] >= 0:
        perm = np.random.RandomState(1000 + task["perm"]).permutation(len(y_tr))
        X_tr, y_tr = X_tr[perm], np.asarray(y_tr)[perm]
    names = [f"f{i}" for i in range(X_tr.shape[1])]
    t0 = time.perf_counter()
    m = ExpertRuleFit(row_order=task["mode"]).fit(X_tr, y_tr, feature_names=names)
    p = m.predict_proba(X_te)[:, 1]
    import re
    rules = sorted({re.sub(r"^rule\[\d+\]:", "", r) for r in m.get_selected_rules()})
    return {**task, "rules": rules, "proba": p.tolist(),
            "auc": float(roc_auc_score(y_te, p)), "fingerprint": m.fingerprint(X_te),
            "fit_seconds": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "results" / "row_order"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [{"dataset": args.dataset, "mode": mode, "perm": k, "data_dir": args.data_dir}
             for mode in ("as_given", "canonical") for k in range(-1, args.n_perm)]
    with ProcessPoolExecutor(args.jobs) as ex:
        res = list(ex.map(run, tasks))
    rows = []
    for mode in ("as_given", "canonical"):
        ref = next(r for r in res if r["mode"] == mode and r["perm"] == -1)
        for r in (r for r in res if r["mode"] == mode):
            a, b = set(ref["rules"]), set(r["rules"])
            rows.append({
                "mode": mode, "perm": r["perm"], "n_rules": len(b),
                "jaccard_vs_original": len(a & b) / len(a | b) if a | b else 1.0,
                "max_abs_proba_diff": float(np.max(np.abs(np.array(ref["proba"]) - np.array(r["proba"])))),
                "auc": r["auc"], "fingerprint_equal": r["fingerprint"] == ref["fingerprint"],
                "predictions_sha256": r["fingerprint"]["predictions_sha256"],
                "fit_seconds": r["fit_seconds"]})
    (out / f"{args.dataset}.json").write_text(json.dumps(rows, indent=1))
    (out / f"{args.dataset}_fits.json").write_text(json.dumps(
        [{k: r[k] for k in ("mode", "perm", "rules", "auc", "fingerprint", "fit_seconds")} for r in res],
        indent=1))
    md = ["| Mode | Permutation | Rules | Jaccard vs original | Max abs prob. diff | AUC | Fingerprint equal |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['mode']} | {'original' if r['perm'] < 0 else r['perm']} | {r['n_rules']} | "
                  f"{r['jaccard_vs_original']:.3f} | {r['max_abs_proba_diff']:.2e} | {r['auc']:.3f} | "
                  f"{r['fingerprint_equal']} |")
    (out / f"{args.dataset}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
