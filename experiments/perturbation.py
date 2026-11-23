"""B1 — Stability under data perturbation, plus repeated-split performance.

Two parts, both reproducible from one command and fixed seeds:

1. **Perturbation stability.** One stratified 80/20 split (seed 42, as in the
   100-seed benchmark). The training set is perturbed ``--n-subsample`` times
   (90 % without replacement) and ``--n-bootstrap`` times (bootstrap). Each
   model is fitted on every perturbed set and scored on the fixed test set.
   Metrics: mean pairwise Jaccard of rule sets (exact rules and coarse rule
   signatures), per-rule selection frequency, rule count, mean rule length,
   and instance-level prediction standard deviation on the test set.

2. **Repeated splits.** ``--n-splits`` stratified 80/20 splits; AUC, Brier
   score and ECE with 95 % t-intervals over splits.

Models: ``rulefit`` (imodels RuleFit, fixed seed 42), ``erf`` (ExpertRuleFit,
benchmark settings) and ``logreg`` (L2 logistic regression on the
preprocessed features). Preprocessing (ordinal encoding, median imputation,
standardisation) is fitted on each training set only.

Seeds: every random draw uses ``BASE_SEED + offset`` (subsample k:
+k; bootstrap k: +1000+k; split s: +2000+s). Each fit runs single-threaded;
``--jobs`` parallelises across fits (separate processes), which does not
change any fit's output.

Fits are appended to ``<out>/<dataset>/fits.jsonl`` and skipped on rerun, so
an interrupted run resumes where it stopped. ``--summarize-only`` rebuilds
the tables and figures from existing fits.

Usage:
    python experiments/fetch_data.py
    python experiments/perturbation.py                      # full study
    python experiments/perturbation.py --datasets german --n-subsample 5 \\
        --n-bootstrap 5 --n-splits 3 --jobs 2               # smoke run
"""

import os

# Single-threaded numerics in this process and in workers (set before numpy).
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
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_SEED = 20261122
SPLIT_SEED_FIXED = 42  # same fixed split as expertrulefit_validation.py
TAIWAN_SUBSAMPLE = 5000  # benchmark used a 5,000-row subsample (seed 42)
MODELS = ("rulefit", "erf", "logreg")
DATASETS = ("german", "taiwan", "hmeq")
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_dataset(name, data_dir, taiwan_full=False):
    """Raw features (DataFrame) and binary target from a local CSV."""
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run experiments/fetch_data.py or copy the CSV "
            f"there (raw features + binary 'target' column)."
        )
    df = pd.read_csv(path)
    y = df.pop("target").to_numpy().astype(int)
    if name == "taiwan" and not taiwan_full and len(df) > TAIWAN_SUBSAMPLE:
        idx = np.random.RandomState(42).choice(len(df), TAIWAN_SUBSAMPLE, replace=False)
        df, y = df.iloc[idx].reset_index(drop=True), y[idx]
    return df, y


class Preprocessor:
    """Ordinal-encode categoricals, median-impute, standardise (fit on train).

    ``scale=False`` skips the standardisation (ExpertRuleFit >= 0.7
    standardises internally, and expert rules can then use raw units).
    """

    def __init__(self, scale=True):
        self.scale = scale

    def fit(self, df):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler

        self.cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num = [c for c in df.columns if c not in self.cat]
        self.imp_num = (
            SimpleImputer(strategy="median").fit(df[self.num]) if self.num else None
        )
        if self.cat:
            self.imp_cat = SimpleImputer(strategy="most_frequent").fit(df[self.cat])
            self.enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit(self.imp_cat.transform(df[self.cat]))
        self.scaler = StandardScaler().fit(self._raw(df))
        return self

    def _raw(self, df):
        parts = []
        if self.num:
            parts.append(self.imp_num.transform(df[self.num]))
        if self.cat:
            parts.append(self.enc.transform(self.imp_cat.transform(df[self.cat])))
        return np.column_stack(parts).astype(np.float64)

    def transform(self, df):
        X = self._raw(df)
        return self.scaler.transform(X) if self.scale else X

    def column(self, name):
        return self.feature_names.index(name)

    def code(self, name, category):
        """Ordinal code of a category label in an encoded column."""
        cats = list(self.enc.categories_[self.cat.index(name)])
        return float(cats.index(category))

    @property
    def feature_names(self):
        return [str(c) for c in self.num + self.cat]


# ---------------------------------------------------------------------------
# Models and rule extraction
# ---------------------------------------------------------------------------

_IDX_PREFIX = re.compile(r"^rule\[\d+\]:")


def normalise_rule(name):
    """Strip ERF's positional prefix so identical rules compare equal across fits."""
    return _IDX_PREFIX.sub("", name)


def rule_signature(rule):
    """Coarse identity: features and directions, thresholds dropped."""
    if rule.startswith("linear:"):
        return rule
    parts = []
    for cond in rule.split(" and "):
        m = re.match(r"\s*(.+?)\s*(<=|>=|<|>)\s*[-+0-9.eE]+\s*$", cond)
        parts.append(f"{m.group(1)}{m.group(2)}" if m else cond.strip())
    return " & ".join(sorted(parts))


def rule_length(rule):
    return 0 if rule.startswith("linear:") else rule.count(" and ") + 1


def fit_model(model, X, y, feature_names, erf_tol):
    from sklearn.linear_model import LogisticRegression

    if model == "rulefit":
        from imodels import RuleFitClassifier

        m = RuleFitClassifier(
            n_estimators=250,
            tree_size=4,
            max_rules=50,
            random_state=42,
            include_linear=True,
        )
        m.fit(X, y, feature_names=feature_names)
        rules_df = m._get_rules()
        rules = set()
        for _, row in rules_df.iterrows():
            if abs(row["coef"]) > 1e-10:
                r = str(row["rule"])
                rules.add(f"linear:{r}" if r in feature_names else r)
        return m, rules
    if model == "erf":
        from expertrulefit import ExpertRuleFit

        m = ExpertRuleFit(
            n_estimators=250,
            tree_size=4,
            max_rules=50,
            n_bootstrap=10,
            rule_threshold=0.8,
            tol=erf_tol,
        )
        m.fit(X, y, feature_names=feature_names)
        return m, {normalise_rule(r) for r in m.get_selected_rules()}
    if model == "logreg":
        m = LogisticRegression(C=1.0, max_iter=5000)
        m.fit(X, y)
        return m, {
            f"linear:{f}" for f, c in zip(feature_names, m.coef_[0]) if abs(c) > 1e-10
        }
    raise ValueError(model)


def ece_score(y, p, n_bins=10):
    """Expected calibration error, equal-width bins, weighted by bin size."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# One fit (runs in a worker process)
# ---------------------------------------------------------------------------


def run_fit(task):
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")
    ds, model, scheme, k = task["dataset"], task["model"], task["scheme"], task["k"]
    df, y = load_dataset(ds, task["data_dir"], task["taiwan_full"])

    split_seed = SPLIT_SEED_FIXED if scheme != "split" else BASE_SEED + 2000 + k
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=split_seed, stratify=y
    )
    seed = split_seed
    if scheme == "subsample":
        seed = BASE_SEED + k
        rng = np.random.RandomState(seed)
        idx_tr = np.sort(
            rng.choice(idx_tr, int(round(0.9 * len(idx_tr))), replace=False)
        )
    elif scheme == "bootstrap":
        seed = BASE_SEED + 1000 + k
        rng = np.random.RandomState(seed)
        idx_tr = rng.choice(idx_tr, len(idx_tr), replace=True)

    pre = Preprocessor().fit(df.iloc[idx_tr])
    X_tr, X_te = pre.transform(df.iloc[idx_tr]), pre.transform(df.iloc[idx_te])
    y_tr, y_te = y[idx_tr], y[idx_te]

    t0 = time.perf_counter()
    m, rules = fit_model(model, X_tr, y_tr, pre.feature_names, task["erf_tol"])
    fit_s = time.perf_counter() - t0
    p = m.predict_proba(X_te)[:, 1]

    non_linear = [r for r in rules if not r.startswith("linear:")]
    return {
        "dataset": ds,
        "model": model,
        "scheme": scheme,
        "k": k,
        "seed": int(seed),
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
        "rules": sorted(rules),
        "n_terms": len(rules),
        "n_rules": len(non_linear),
        "n_linear": len(rules) - len(non_linear),
        "mean_rule_len": float(np.mean([rule_length(r) for r in non_linear]))
        if non_linear
        else 0.0,
        "auc": float(roc_auc_score(y_te, p)),
        "brier": float(brier_score_loss(y_te, p)),
        "ece": ece_score(y_te, p),
        "fit_seconds": round(fit_s, 3),
        "test_proba": [round(float(v), 8) for v in p] if scheme != "split" else None,
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets):
    vals = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)


def t_ci(x):
    from scipy import stats

    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def load_fits(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(out_dir, datasets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stab_rows, perf_rows = [], []
    for ds in datasets:
        fits = load_fits(out_dir / ds / "fits.jsonl")
        if not fits:
            continue
        df = pd.DataFrame(fits)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        box_data, box_labels = [], []
        for (model, scheme), g in df[df.scheme != "split"].groupby(["model", "scheme"]):
            g = g.sort_values("k")
            sets = [set(r) for r in g.rules]
            sigs = [{rule_signature(r) for r in s} for s in sets]
            jm, js = mean_pairwise_jaccard(sets)
            sm, _ = mean_pairwise_jaccard(sigs)
            P = np.array(g.test_proba.tolist())
            pair_j = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
            box_data.append(pair_j)
            box_labels.append(f"{model}\n{scheme}")

            freq = pd.Series([r for s in sets for r in s]).value_counts() / len(sets)
            freq.rename("selection_frequency").rename_axis("rule").to_csv(
                out_dir / ds / f"selection_freq_{model}_{scheme}.csv"
            )
            if model != "logreg":
                axes[1].plot(
                    np.arange(1, len(freq) + 1),
                    freq.values,
                    label=f"{model} / {scheme}",
                )

            stab_rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "scheme": scheme,
                    "n_fits": len(g),
                    "jaccard_mean": jm,
                    "jaccard_sd": js,
                    "signature_jaccard_mean": sm,
                    "n_rules_mean": g.n_rules.mean(),
                    "n_rules_sd": g.n_rules.std(ddof=1),
                    "n_linear_mean": g.n_linear.mean(),
                    "mean_rule_len": g.mean_rule_len.mean(),
                    "n_distinct_terms": len(freq),
                    "share_terms_freq_ge_0.8": float((freq >= 0.8).mean()),
                    "pred_std_mean": float(P.std(axis=0).mean())
                    if len(P) > 1
                    else np.nan,
                    "pred_std_max": float(P.std(axis=0).max())
                    if len(P) > 1
                    else np.nan,
                    "auc_mean": g.auc.mean(),
                    "auc_sd": g.auc.std(ddof=1),
                    "fit_seconds_mean": g.fit_seconds.mean(),
                }
            )
        if box_data:
            axes[0].boxplot(box_data, tick_labels=box_labels, showfliers=False)
            axes[0].set_ylabel("pairwise Jaccard (rule sets)")
            axes[0].tick_params(axis="x", labelsize=7)
            axes[1].set_xlabel("rule rank")
            axes[1].set_ylabel("selection frequency")
            axes[1].axhline(0.8, ls=":", c="grey", lw=0.8)
            axes[1].legend(fontsize=7)
            fig.suptitle(f"{ds}: stability under perturbation")
            fig.tight_layout()
            fig.savefig(out_dir / ds / "stability.png", dpi=150)
        plt.close(fig)

        for model, g in df[df.scheme == "split"].groupby("model"):
            row = {"dataset": ds, "model": model, "n_splits": len(g)}
            for metric in ("auc", "brier", "ece"):
                m, lo, hi = t_ci(g[metric])
                row.update(
                    {f"{metric}_mean": m, f"{metric}_lo": lo, f"{metric}_hi": hi}
                )
            perf_rows.append(row)

    stab = pd.DataFrame(stab_rows)
    perf = pd.DataFrame(perf_rows)
    if not stab.empty:
        stab.to_csv(out_dir / "stability_summary.csv", index=False)
    if not perf.empty:
        perf.to_csv(out_dir / "performance_summary.csv", index=False)
    write_tables(out_dir, stab, perf)
    return stab, perf


def write_tables(out_dir, stab, perf):
    """Paper-ready Markdown and LaTeX tables."""
    md, tex = [], []
    if not stab.empty:
        md += [
            "| Dataset | Model | Scheme | Fits | Jaccard (exact) | Jaccard (signature) "
            "| Rules | Rule length | Pred. SD |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        tex += [
            r"\begin{tabular}{@{}lllrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Model & Scheme & Fits & $J$ exact & $J$ signature & Rules "
            r"& Length & Pred.\ SD\\",
            r"\midrule",
        ]
        for r in stab.itertuples():
            md.append(
                f"| {r.dataset} | {r.model} | {r.scheme} | {r.n_fits} | "
                f"{r.jaccard_mean:.3f} | {r.signature_jaccard_mean:.3f} | "
                f"{r.n_rules_mean:.1f} | {r.mean_rule_len:.2f} | {r.pred_std_mean:.4f} |"
            )
            tex.append(
                f"{r.dataset} & {r.model} & {r.scheme} & {r.n_fits} & "
                f"{r.jaccard_mean:.3f} & {r.signature_jaccard_mean:.3f} & "
                f"{r.n_rules_mean:.1f} & {r.mean_rule_len:.2f} & "
                f"{r.pred_std_mean:.4f}\\\\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", ""]
    if not perf.empty:
        md += [
            "",
            "| Dataset | Model | Splits | AUC [95% CI] | Brier [95% CI] | ECE [95% CI] |",
            "|---|---|---|---|---|---|",
        ]
        tex += [
            r"\begin{tabular}{@{}llrccc@{}}",
            r"\toprule",
            r"Dataset & Model & Splits & AUC & Brier & ECE\\",
            r"\midrule",
        ]
        for r in perf.itertuples():
            cells = [
                f"{getattr(r, m + '_mean'):.3f} [{getattr(r, m + '_lo'):.3f}, "
                f"{getattr(r, m + '_hi'):.3f}]"
                for m in ("auc", "brier", "ece")
            ]
            md.append(
                f"| {r.dataset} | {r.model} | {r.n_splits} | "
                + " | ".join(cells)
                + " |"
            )
            tex.append(
                f"{r.dataset} & {r.model} & {r.n_splits} & " + " & ".join(cells) + r"\\"
            )
        tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tables.md").write_text("\n".join(md) + "\n")
    (out_dir / "tables.tex").write_text("\n".join(tex) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="B1 perturbation stability study")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--n-subsample", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=100)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1, help="parallel fits (processes)")
    ap.add_argument(
        "--erf-tol",
        type=float,
        default=1e-6,
        help="ExpertRuleFit tol (benchmark used 1e-6)",
    )
    ap.add_argument(
        "--taiwan-full",
        action="store_true",
        help="use all 30,000 Taiwan rows instead of the 5,000-row subsample",
    )
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "results" / "perturbation"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not args.summarize_only:
        tasks = []
        for ds in args.datasets:
            (out_dir / ds).mkdir(parents=True, exist_ok=True)
            done = {
                (f["model"], f["scheme"], f["k"])
                for f in load_fits(out_dir / ds / "fits.jsonl")
            }
            for model in args.models:
                for scheme, n in (
                    ("subsample", args.n_subsample),
                    ("bootstrap", args.n_bootstrap),
                    ("split", args.n_splits),
                ):
                    for k in range(n):
                        if (model, scheme, k) not in done:
                            tasks.append(
                                {
                                    "dataset": ds,
                                    "model": model,
                                    "scheme": scheme,
                                    "k": k,
                                    "data_dir": args.data_dir,
                                    "taiwan_full": args.taiwan_full,
                                    "erf_tol": args.erf_tol,
                                }
                            )
        print(f"{len(tasks)} fits to run ({args.jobs} process(es))", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(run_fit, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with open(out_dir / rec["dataset"] / "fits.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if i % 10 == 0 or i == len(tasks):
                    print(
                        f"  {i}/{len(tasks)} done, {time.perf_counter() - t0:.0f}s",
                        flush=True,
                    )

    stab, perf = summarize(out_dir, args.datasets)
    print((out_dir / "tables.md").read_text())


if __name__ == "__main__":
    main()
