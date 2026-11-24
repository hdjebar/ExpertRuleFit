"""Check that results reused from the 0.8.0 Table 1 run still hold here.

Table 1 under 0.8.1 reuses the 300 RuleFit fits of ``table1_v080`` (a
different machine and data download). This script, run where the 0.8.1
ERF fits run, refits:

- RuleFit with seeds 0..``--seeds``-1 and compares rule-set hash, AUC and
  the SHA-256 of the test probabilities with the reused fits;
- ExpertRuleFit with ``column_order="as_given"`` (the 0.8.0 behaviour)
  and compares the same fields with the 0.8.0 ERF fit.

Identical hashes mean the reused fits are valid here (same data, same
numerics). Writes ``<out>/reuse_check_<dataset>.json``.

    python experiments/check_reuse.py --dataset german --out out/table1
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import table1  # noqa: E402

FIELDS = ("rules_hash", "auc", "proba_sha256")


def erf_as_given(dataset, data_dir):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from expertrulefit import ExpertRuleFit

    warnings.filterwarnings("ignore")
    v = table1._bench()
    X, y = table1._load(dataset, data_dir)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_te = v.preprocess(X_tr, X_te)
    names = [f"f{i}" for i in range(X_tr.shape[1])]
    m = ExpertRuleFit(random_state=0, column_order="as_given")
    m.fit(X_tr, y_tr, feature_names=names)
    rules = v.extract_expert_rules(m)
    p = np.ascontiguousarray(m.predict_proba(X_te)[:, 1], dtype=np.float64)
    return {
        "rules_hash": v.rules_hash(rules),
        "auc": float(roc_auc_score(y_te, p)),
        "proba_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--reference", default=str(HERE / "results" / "table1_v080"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = {
        (f["model"], f["seed"]): f
        for f in table1.load_fits(Path(args.reference) / "fits.jsonl")
        if f["dataset"] == args.dataset
    }
    checks = []
    for s in range(args.seeds):
        new = table1.run_fit(
            {
                "dataset": args.dataset,
                "model": "rulefit",
                "seed": s,
                "data_dir": args.data_dir,
            }
        )
        old = ref[("rulefit", s)]
        checks.append(
            {"model": "rulefit", "seed": s, **{k: new[k] == old[k] for k in FIELDS}}
        )
    old_erf = ref[("erf", 0)]
    new_erf = erf_as_given(args.dataset, args.data_dir)
    checks.append(
        {
            "model": "erf_0.8.0_as_given",
            "seed": 0,
            **{k: new_erf[k] == old_erf[k] for k in FIELDS},
        }
    )
    result = {
        "dataset": args.dataset,
        "reference": args.reference,
        "checks": checks,
        "all_identical": all(all(c[k] for k in FIELDS) for c in checks),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"reuse_check_{args.dataset}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
