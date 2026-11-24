"""B1 — stability under data perturbation (revised protocol, review 4).

One stratified 80/20 split (seed 42, as Table 1). The training split is
perturbed ``--n-sub`` times by stratified 90 % subsampling without
replacement (seed BASE_SEED + k); preprocessing (imputation, ordinal
encoding, standardisation) is refitted inside every subsample. Each arm is
also fitted once on the full training split (reference for discrepancy).
No expert rules: this isolates rule generation and selection.

Arms
----
rulefit_fixed_seed  imodels RuleFit, internal seed fixed at 42 (key control)
erf_default         ExpertRuleFit library defaults. Up to 0.8.0 results
                    (results/b1/german*) this had the filter on; since the
                    0.8.0 B1 decision the default is selection="none".
erf_no_filter       ExpertRuleFit, selection="none"
erf_filter          ExpertRuleFit, selection="bootstrap" (filter on, slow)
erf_trees           ExpertRuleFit, candidates="trees" (decision b, slow)

--k-start/--k-stop restrict the subsample indices (k = -1 is the full
training split), so GitHub Actions can split one dataset over many jobs;
the chunks' jsonl files are concatenated and summarised with
--summarize-only (.github/workflows/b1.yml).

Metrics over all pairs of perturbed fits
----------------------------------------
- Jaccard and Sørensen–Dice of rule sets (rule strings, positional prefix
  removed; linear terms included as "linear:<feature>");
- semantic coverage similarity: rules of the two fits are matched one to
  one (Hungarian algorithm) on the Jaccard similarity of their activation
  sets on the test rows; the matched similarity is summed and divided by
  the larger rule count. Threshold jitter (b <= 0.31 vs b <= 0.32) then
  counts as near-agreement instead of disagreement;
- prediction stability on the test rows: mean per-row SD of probabilities;
  ambiguity (share of test rows whose predicted class at 0.5 differs
  between at least two fits) and discrepancy (largest share of test rows
  on which one perturbed fit disagrees with the full-data fit), after Marx,
  Calmon and Ustun (2020);
- test AUC (mean, sd) and rule counts.

    docker run --rm --network none -v "$PWD":/work -w /work \\
        -e PYTHONPATH=/work erf-pinned python experiments/b1_stability.py --jobs 2
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
import itertools
import json
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE_SEED = 20261123
ARMS = ("rulefit_fixed_seed", "erf_default", "erf_no_filter", "erf_filter", "erf_trees")
_PREFIX = re.compile(r"^rule\[\d+\]:")


def _data(dataset, data_dir, k):
    from sklearn.model_selection import train_test_split
    from perturbation import Preprocessor, load_dataset

    df, y = load_dataset(dataset, data_dir)
    i_tr, i_te = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    if k >= 0:
        rng = np.random.RandomState(BASE_SEED + k)
        keep = []
        for c in np.unique(y[i_tr]):
            idx = i_tr[y[i_tr] == c]
            keep.append(rng.choice(idx, int(round(0.9 * len(idx))), replace=False))
        i_tr = np.sort(np.concatenate(keep))
    pre = Preprocessor(scale=True).fit(df.iloc[i_tr])
    return (
        pre.transform(df.iloc[i_tr]),
        pre.transform(df.iloc[i_te]),
        y[i_tr],
        y[i_te],
        pre.feature_names,
    )


def run(task):
    from sklearn.metrics import roc_auc_score

    from expertrulefit.expert_rulefit import eval_rule_on_data

    warnings.filterwarnings("ignore")
    arm, k = task["arm"], task["k"]
    X_tr, X_te, y_tr, y_te, names = _data(task["dataset"], task["data_dir"], k)
    t0 = time.perf_counter()
    if arm == "rulefit_fixed_seed":
        from imodels import RuleFitClassifier

        m = RuleFitClassifier(
            n_estimators=250,
            tree_size=4,
            max_rules=50,
            random_state=42,
            include_linear=True,
        )
        m.fit(X_tr, y_tr, feature_names=names)
        p = m.predict_proba(X_te)[:, 1]
        lin = [
            f"linear:{f}" for f, c in zip(names, m.coef[: len(names)]) if abs(c) > 1e-10
        ]
        # imodels keeps only rules with non-zero coefficient (its own L1 path)
        rules = [
            (str(rn), rw.rule)
            for rn, rw in zip(m.rules_, m.rules_without_feature_names_)
        ]
    else:
        from expertrulefit import ExpertRuleFit

        kw = {
            "erf_default": {},
            "erf_no_filter": {"selection": "none"},
            "erf_filter": {"selection": "bootstrap"},
            "erf_trees": {"candidates": "trees"},
        }[arm]
        m = ExpertRuleFit(**kw).fit(X_tr, y_tr, feature_names=names)
        p = m.predict_proba(X_te)[:, 1]
        stable = np.where(m.stable_mask_)[0]
        coefs = m._get_true_coefficients()
        # Model column order (canonical by name since 0.8.1): linear terms
        # and the X_<i> placeholders of generated rules index these columns.
        names = list(m.feature_names_)
        X_te = X_te[:, getattr(m, "column_order_", np.arange(X_te.shape[1]))]
        fingerprint = m.fingerprint()
        n_lin = len(names)
        lin, rules = [], []
        base_rules = m.base_rulefit_.rules_without_feature_names_
        for j, i in enumerate(stable):
            if abs(coefs[j]) <= 1e-10:
                continue
            if i < n_lin:
                lin.append(f"linear:{names[i]}")
            else:
                rw = base_rules[i - n_lin]
                rules.append((_PREFIX.sub("", m.rule_names_[i]), rw.rule))
    if arm == "rulefit_fixed_seed":
        fingerprint = None
    fit_s = time.perf_counter() - t0
    acts = [eval_rule_on_data(rw, X_te).astype(bool) for _, rw in rules]
    return {
        "dataset": task["dataset"],
        "arm": arm,
        "k": k,
        "n_train": int(len(y_tr)),
        "auc": float(roc_auc_score(y_te, p)),
        "proba": [round(float(v), 10) for v in p],
        "terms": sorted(lin + [n for n, _ in rules]),
        "n_rules": len(rules),
        "n_linear": len(lin),
        "activations": [np.packbits(a).tolist() for a in acts],
        "n_test": int(len(y_te)),
        "fit_seconds": round(fit_s, 1),
        "fingerprint": fingerprint,
        "erf_version": _erf_version(),
    }


def _erf_version():
    from expertrulefit._version import __version__

    return __version__


def _unpack(packed, n):
    return np.unpackbits(np.array(packed, dtype=np.uint8))[:n].astype(bool)


def semantic_similarity(A, B):
    """Hungarian-matched mean Jaccard of rule activation sets (A, B: lists of bool arrays)."""
    from scipy.optimize import linear_sum_assignment

    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    MA, MB = np.array(A, dtype=float), np.array(B, dtype=float)
    inter = MA @ MB.T
    union = MA.sum(1)[:, None] + MB.sum(1)[None, :] - inter
    S = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].sum() / max(len(A), len(B)))


def summarize(out, dataset):
    fits = [
        json.loads(l)
        for l in (out / f"{dataset}.jsonl").read_text().splitlines()
        if l.strip()
    ]
    rows = []
    for arm in ARMS:
        g = sorted((f for f in fits if f["arm"] == arm), key=lambda f: f["k"])
        if not g:
            continue
        ref = next((f for f in g if f["k"] == -1), None)
        pert = [f for f in g if f["k"] >= 0]
        if len(pert) < 2:
            continue
        sets = [set(f["terms"]) for f in pert]
        acts = [[_unpack(a, f["n_test"]) for a in f["activations"]] for f in pert]
        jac, dice, sem = [], [], []
        for i, j in itertools.combinations(range(len(pert)), 2):
            a, b = sets[i], sets[j]
            inter = len(a & b)
            jac.append(inter / len(a | b) if a | b else 1.0)
            dice.append(2 * inter / (len(a) + len(b)) if (a or b) else 1.0)
            sem.append(semantic_similarity(acts[i], acts[j]))
        P = np.array([f["proba"] for f in pert])
        cls = P >= 0.5
        ambiguity = float(np.mean(cls.min(axis=0) != cls.max(axis=0)))
        discrepancy = (
            float(max(np.mean(c != (np.array(ref["proba"]) >= 0.5)) for c in cls))
            if ref
            else None
        )
        aucs = [f["auc"] for f in pert]
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "n_fits": len(pert),
                "jaccard_mean": float(np.mean(jac)),
                "jaccard_sd": float(np.std(jac)),
                "dice_mean": float(np.mean(dice)),
                "semantic_mean": float(np.mean(sem)),
                "semantic_sd": float(np.std(sem)),
                "pred_sd_mean": float(P.std(axis=0).mean()),
                "ambiguity": ambiguity,
                "discrepancy": discrepancy,
                "auc_mean": float(np.mean(aucs)),
                "auc_sd": float(np.std(aucs, ddof=1)),
                "auc_full": ref["auc"] if ref else None,
                "n_rules_mean": float(np.mean([f["n_rules"] for f in pert])),
                "n_linear_mean": float(np.mean([f["n_linear"] for f in pert])),
                "fit_seconds_mean": float(np.mean([f["fit_seconds"] for f in pert])),
            }
        )
    (out / f"{dataset}_summary.json").write_text(json.dumps(rows, indent=1))
    md = [
        "| Arm | Fits | Jaccard (sd) | Dice | Semantic (sd) | Pred. SD | Ambiguity | Discrepancy "
        "| AUC mean (sd) | AUC full | Rules + linear | Fit s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        disc = "–" if r["discrepancy"] is None else "%.3f" % r["discrepancy"]
        full = "–" if r["auc_full"] is None else "%.3f" % r["auc_full"]
        md.append(
            f"| {r['arm']} | {r['n_fits']} | {r['jaccard_mean']:.3f} ({r['jaccard_sd']:.3f}) | "
            f"{r['dice_mean']:.3f} | {r['semantic_mean']:.3f} ({r['semantic_sd']:.3f}) | "
            f"{r['pred_sd_mean']:.4f} | {r['ambiguity']:.3f} | {disc} | "
            f"{r['auc_mean']:.3f} ({r['auc_sd']:.3f}) | {full} | "
            f"{r['n_rules_mean']:.1f} + {r['n_linear_mean']:.1f} | {r['fit_seconds_mean']:.0f} |"
        )
    (out / f"{dataset}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--arms", nargs="+", default=list(ARMS[:3]), choices=ARMS)
    ap.add_argument("--n-sub", type=int, default=25)
    ap.add_argument("--k-start", type=int, default=-1)
    ap.add_argument("--k-stop", type=int, default=None, help="exclusive; default n-sub")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "results" / "b1"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}.jsonl"
    if not args.summarize_only:
        done = set()
        if path.exists():
            done = {
                (f["arm"], f["k"])
                for f in map(json.loads, path.read_text().splitlines())
            }
        tasks = [
            {"dataset": args.dataset, "arm": a, "k": k, "data_dir": args.data_dir}
            for k in range(
                args.k_start, args.n_sub if args.k_stop is None else args.k_stop
            )
            for a in args.arms
            if (a, k) not in done
        ]
        print(f"{len(tasks)} fits to run", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.jobs) as ex:
            for fut in as_completed([ex.submit(run, t) for t in tasks]):
                r = fut.result()
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(
                    f"  {r['arm']} k={r['k']}: AUC {r['auc']:.3f}, {r['n_rules']}+{r['n_linear']} terms, "
                    f"{r['fit_seconds']}s [{time.perf_counter() - t0:.0f}s]",
                    flush=True,
                )
    summarize(out, args.dataset)


if __name__ == "__main__":
    main()
