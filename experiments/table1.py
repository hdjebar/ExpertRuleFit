"""Regenerate Table 1 (seed sensitivity) under the corrected protocol.

Protocol, as in ``expertrulefit_validation.py`` except where noted:

- German Credit, Taiwan Credit (5,000-row subsample of 30,000, seed 42) and
  HMEQ; one stratified 80/20 split (seed 42).
- Preprocessing (imputation, ordinal encoding, standardisation) fitted on
  the training split only (the original run fitted it on the full data).
- RuleFit: imodels ``RuleFitClassifier`` (250 trees, depth 4, at most 50
  rules), ``random_state`` = 0..``--rulefit-seeds``-1.
- ExpertRuleFit: benchmark settings (tol 1e-6), run ``--erf-runs`` times
  with ``random_state`` = 0, 1, 2, ... (ignored by design), each in its own
  process. The script fails unless every run gives the same rule-set hash
  and the same SHA-256 of the test-set probabilities.

Run it in the pinned image (``docker/``)::

    docker run --rm --network none -v "$PWD":/work -w /work \\
        -e PYTHONPATH=/work erf-pinned python experiments/table1.py --jobs 2

Fits are appended to ``<out>/fits.jsonl`` and skipped on rerun.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
DATASETS = {"german": "German Credit", "taiwan": "Taiwan Credit", "hmeq": "HMEQ"}


def _bench():
    """Import the benchmark module lazily (it configures logging on import)."""
    import expertrulefit_validation as v
    return v


def _load(ds, data_dir):
    """Raw data: local CSV from fetch_data.py if present, else the benchmark loader."""
    import pandas as pd

    path = Path(data_dir) / f"{ds}.csv"
    if path.exists():
        df = pd.read_csv(path)
        y = df.pop("target").to_numpy().astype(int)
        if ds == "taiwan" and len(df) > 5000:
            idx = np.random.RandomState(42).choice(len(df), size=5000, replace=False)
            df, y = df.iloc[idx].reset_index(drop=True), y[idx]
        return df, y
    v = _bench()
    X, y, _ = {"german": v.load_german_credit, "taiwan": v.load_taiwan_credit,
               "hmeq": v.load_hmeq}[ds]()
    return X, np.asarray(y, dtype=int)


def run_fit(task):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")
    v = _bench()
    ds, model, seed = task["dataset"], task["model"], task["seed"]
    X, y = _load(ds, task["data_dir"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    X_tr, X_te = v.preprocess(X_tr, X_te)
    names = [f"f{i}" for i in range(X_tr.shape[1])]

    t0 = time.perf_counter()
    if model == "rulefit":
        from imodels import RuleFitClassifier

        m = RuleFitClassifier(n_estimators=250, tree_size=4, max_rules=50,
                              random_state=seed, include_linear=True)
        m.fit(X_tr, y_tr, feature_names=names)
        rules = v.extract_rulefit_rules(m)
    else:
        m = v.ExpertRuleFitClassifier(n_estimators=250, tree_size=4, max_rules=50,
                                      random_state=seed, n_bootstrap=10,
                                      rule_threshold=0.8, tol=1e-6)
        m.fit(X_tr, y_tr, feature_names=names)
        rules = v.extract_expert_rules(m)
    fit_s = time.perf_counter() - t0
    p = np.ascontiguousarray(m.predict_proba(X_te)[:, 1], dtype=np.float64)
    return {
        "dataset": ds, "model": model, "seed": seed,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "auc": float(roc_auc_score(y_te, p)),
        "rules": sorted(rules), "n_rules": len(rules),
        "rules_hash": v.rules_hash(rules),
        "proba_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
        "fit_seconds": round(fit_s, 1),
    }


def load_fits(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def summarize(out, datasets, n_rf, n_erf):
    import sklearn
    import imodels

    fits = load_fits(out / "fits.jsonl")
    rows, problems = [], []
    for ds in datasets:
        rf = sorted((f for f in fits if f["dataset"] == ds and f["model"] == "rulefit"),
                    key=lambda f: f["seed"])
        erf = sorted((f for f in fits if f["dataset"] == ds and f["model"] == "erf"),
                     key=lambda f: f["seed"])
        if not rf or not erf:
            continue
        sets = [set(f["rules"]) for f in rf]
        jac = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        modal = Counter(f["rules_hash"] for f in rf).most_common(1)[0][1]
        erf_rule_hashes = {f["rules_hash"] for f in erf}
        erf_proba_hashes = {f["proba_sha256"] for f in erf}
        identical = len(erf_rule_hashes) == 1 and len(erf_proba_hashes) == 1
        if not identical:
            problems.append(ds)
        rows.append({
            "dataset": ds, "n_train": rf[0]["n_train"], "n_test": rf[0]["n_test"],
            "rulefit_runs": len(rf), "rulefit_modal_count": modal,
            "rulefit_unique_sets": len({f["rules_hash"] for f in rf}),
            "rulefit_jaccard_mean": float(np.mean(jac)), "rulefit_jaccard_sd": float(np.std(jac)),
            "rulefit_auc_mean": float(np.mean([f["auc"] for f in rf])),
            "rulefit_auc_sd": float(np.std([f["auc"] for f in rf], ddof=1)),
            "erf_runs": len(erf), "erf_identical": identical,
            "erf_rules_hash": sorted(erf_rule_hashes), "erf_proba_sha256": sorted(erf_proba_hashes),
            "erf_auc": erf[0]["auc"], "erf_n_rules": erf[0]["n_rules"],
            "erf_fit_seconds": [f["fit_seconds"] for f in erf],
        })
    import expertrulefit
    from expertrulefit._version import __version__ as erf_version
    env = {"expertrulefit": f"{erf_version} ({Path(expertrulefit.__file__).parent})",
           "python": platform.python_version(), "numpy": np.__version__,
           "scikit-learn": sklearn.__version__,
           "imodels": __import__("importlib.metadata").metadata.version("imodels"),
           "platform": platform.platform()}
    (out / "table1.json").write_text(json.dumps({"environment": env, "rows": rows}, indent=1))

    md = ["| Dataset | RuleFit: identical rule sets | RuleFit: distinct sets | RuleFit: mean Jaccard "
          "| ERF: identical runs | RuleFit AUC (mean ± sd) | ERF AUC | ERF rules |",
          "|---|---|---|---|---|---|---|---|"]
    tex = [r"Dataset & RuleFit & \textsc{ExpertRuleFit} & RuleFit & \textsc{ExpertRuleFit}\\"]
    for r in rows:
        md.append(f"| {DATASETS[r['dataset']]} | {r['rulefit_modal_count']}/{r['rulefit_runs']} | "
                  f"{r['rulefit_unique_sets']} | {r['rulefit_jaccard_mean']:.3f} | "
                  f"{r['erf_runs'] if r['erf_identical'] else 'NOT'}/{r['erf_runs']} | "
                  f"{r['rulefit_auc_mean']:.3f} ± {r['rulefit_auc_sd']:.3f} | {r['erf_auc']:.3f} | "
                  f"{r['erf_n_rules']} |")
        tex.append(f"{DATASETS[r['dataset']]} & {r['rulefit_modal_count']}/{r['rulefit_runs']} & "
                   f"{r['erf_runs']}/{r['erf_runs']} & {r['rulefit_auc_mean']:.3f} & "
                   f"{r['erf_auc']:.3f}\\\\")
    md.append("")
    md.append("Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()))
    (out / "table1.md").write_text("\n".join(md) + "\n")
    (out / "table1.tex").write_text("\n".join(tex) + "\n")
    print((out / "table1.md").read_text())
    if problems:
        raise AssertionError(f"ExpertRuleFit runs were not identical on: {problems}")


def main():
    ap = argparse.ArgumentParser(description="Regenerate Table 1")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--rulefit-seeds", type=int, default=100)
    ap.add_argument("--erf-runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "results" / "table1"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        done = {(f["dataset"], f["model"], f["seed"]) for f in load_fits(out / "fits.jsonl")}
        tasks = [{"dataset": d, "model": "erf", "seed": s, "data_dir": args.data_dir}
                 for d in args.datasets for s in range(args.erf_runs)]
        tasks += [{"dataset": d, "model": "rulefit", "seed": s, "data_dir": args.data_dir}
                  for d in args.datasets for s in range(args.rulefit_seeds)]
        tasks = [t for t in tasks if (t["dataset"], t["model"], t["seed"]) not in done]
        print(f"{len(tasks)} fits to run ({args.jobs} process(es))", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_fit, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                with open(out / "fits.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                if r["model"] == "erf" or i % 25 == 0:
                    print(f"  {i}/{len(tasks)} {r['dataset']} {r['model']} seed={r['seed']} "
                          f"AUC={r['auc']:.3f} rules={r['n_rules']} hash={r['rules_hash']} "
                          f"{r['fit_seconds']}s [{time.perf_counter() - t0:.0f}s]", flush=True)
    summarize(out, args.datasets, args.rulefit_seeds, args.erf_runs)


if __name__ == "__main__":
    main()
