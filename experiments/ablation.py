"""C3 comparison and B2 ablation (with EBM baseline, C1) — German Credit first.

Same split and illustrative expert rules as ``sensitivity.py`` (one
confirmatory rule "checking account < 0 DM" and one optional rule
"duration > 24 months", both expected +1), preprocessing on the training
split without scaling, ExpertRuleFit library defaults (w_c = 1e-2,
w_o = 0.3, tol 1e-4) unless the variant changes them.

Variants
--------
erf_refit            full model, confirmatory_mode="refit" (current default)
erf_constrained      full model, one convex solve, sign bound from expected_sign
erf_constrained_ridge   same, ridge on the non-confirmatory block
erf_no_filter        selection="none" (no bootstrap frequency filter)
erf_no_weighting     w_c = w_o = 1 (no expert penalty weighting)
erf_no_enforcement   confirmatory_mode="none" (no refit, no constrained block)
erf_no_expert_rules  full model without expert rules (comparable to RuleFit)
rulefit_fixed_seed   imodels RuleFit with fixed internal seed 42 (key control)
ebm                  InterpretML EBM (baseline, already a dependency)

Each ExpertRuleFit variant runs ``--runs`` times (separate processes); the
script fails unless the runs give identical fingerprints (rules,
coefficients, test probabilities).

    docker run --rm --network none -v "$PWD":/work -w /work \\
        -e PYTHONPATH=/work erf-pinned python experiments/ablation.py --jobs 2
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
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

ERF_VARIANTS = {
    "erf_refit": dict(),
    "erf_constrained": dict(confirmatory_mode="constrained"),
    "erf_constrained_ridge": dict(confirmatory_mode="constrained", constrained_penalty="ridge"),
    "erf_no_filter": dict(selection="none"),
    "erf_no_weighting": dict(confirmatory_penalty=1.0, optional_penalty=1.0),
    "erf_no_enforcement": dict(confirmatory_mode="none"),
    "erf_no_expert_rules": dict(),
}
OTHER = ("rulefit_fixed_seed", "ebm")


def _split(dataset, data_dir):
    from sklearn.model_selection import train_test_split
    from perturbation import Preprocessor, load_dataset

    df, y = load_dataset(dataset, data_dir)
    i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    pre = Preprocessor(scale=False).fit(df.iloc[i_tr])
    return pre, pre.transform(df.iloc[i_tr]), pre.transform(df.iloc[i_te]), y[i_tr], y[i_te]


def _rule_len(name):
    body = re.sub(r"^(rule\[\d+\]:|confirmatory:|optional:)", "", name)
    return body.count(" and ") + 1


def run(task):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from sensitivity import EXPERT_RULES, make_rule

    ds, variant = task["dataset"], task["variant"]
    pre, X_tr, X_te, y_tr, y_te = _split(ds, task["data_dir"])
    names = pre.feature_names
    rec = {"dataset": ds, "variant": variant, "run": task["run"]}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if variant.startswith("erf"):
            from expertrulefit import ExpertRuleFit

            conf = [] if variant == "erf_no_expert_rules" else [make_rule(pre, EXPERT_RULES[ds]["confirmatory"])]
            opt = [] if variant == "erf_no_expert_rules" else [make_rule(pre, EXPERT_RULES[ds]["optional"])]
            m = ExpertRuleFit(**ERF_VARIANTS[variant]).fit(
                X_tr, y_tr, feature_names=names, confirmatory_rules=conf, optional_rules=opt)
            p = m.predict_proba(X_te)[:, 1]
            stable = np.where(m.stable_mask_)[0]
            coefs = m._get_true_coefficients()
            terms = [(m.rule_names_[i], c) for i, c in zip(stable, coefs) if abs(c) > 1e-10]
            Z = m._build_stable_raw_matrix(X_te)
            rule_cols = [j for j, i in enumerate(stable)
                         if abs(coefs[j]) > 1e-10 and not m.rule_names_[i].startswith("linear:")]
            per_case = (np.abs(Z[:, rule_cols]) > 0).sum(axis=1) if rule_cols else np.zeros(len(y_te))
            st = m.confirmatory_status_[0] if m.confirmatory_status_ else {}
            rec.update({
                "timings": m.timings_, "fingerprint": m.fingerprint(X_te),
                "n_stable": int(m.n_stable_rules_),
                "confirmatory_coef": st.get("coefficient"),
                "confirmatory_sign_ok": st.get("sign_ok"),
                "confirmatory_bound_active": st.get("bound_active"),
                "confirmatory_included": st.get("active"),
                "optional_active": any(n.startswith("optional:") for n, _ in terms),
                "constrained_info": m.constrained_info_,
            })
        elif variant == "rulefit_fixed_seed":
            from imodels import RuleFitClassifier
            from expertrulefit.expert_rulefit import eval_rule_on_data

            m = RuleFitClassifier(n_estimators=250, tree_size=4, max_rules=50,
                                  random_state=42, include_linear=True)
            m.fit(X_tr, y_tr, feature_names=names)
            p = m.predict_proba(X_te)[:, 1]
            rules = m._get_rules()
            rules = rules[rules.coef.abs() > 1e-10]
            terms = [((f"linear:{r}" if r in names else r), c) for r, c in zip(rules.rule, rules.coef)]
            idx = {n: i for i, n in enumerate(names)}
            pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b")
            acts = [eval_rule_on_data(pat.sub(lambda mo: f"X_{idx[mo.group(1)]}", n), X_te)
                    for n, _ in terms if not n.startswith("linear:")]
            per_case = np.sum(acts, axis=0) if acts else np.zeros(len(y_te))
        else:  # ebm
            import pandas as pd
            from interpret.glassbox import ExplainableBoostingClassifier

            m = ExplainableBoostingClassifier(random_state=42, n_jobs=1)
            m.fit(pd.DataFrame(X_tr, columns=names), y_tr)
            p = m.predict_proba(pd.DataFrame(X_te, columns=names))[:, 1]
            terms = [(t, 1.0) for t in m.term_names_]
            per_case = np.full(len(y_te), len(terms))
    rec["fit_seconds"] = round(time.perf_counter() - t0, 1)
    rec["convergence_warnings"] = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    rule_terms = [n for n, _ in terms if not n.startswith("linear:")] if variant != "ebm" else []
    rec.update({
        "auc": float(roc_auc_score(y_te, p)), "brier": float(brier_score_loss(y_te, p)),
        "n_terms": len(terms), "n_rules": len(rule_terms),
        "n_linear": sum(1 for n, _ in terms if n.startswith("linear:")),
        "mean_rule_len": float(np.mean([_rule_len(n) for n in rule_terms])) if rule_terms else None,
        "active_rules_per_case_mean": float(np.mean(per_case)),
        "active_rules_per_case_max": int(np.max(per_case)),
        "terms": sorted(n for n, _ in terms),
    })
    return rec


def summarize(out, dataset):
    fits = [json.loads(l) for l in (out / f"{dataset}.jsonl").read_text().splitlines() if l.strip()]
    order = list(ERF_VARIANTS) + list(OTHER)
    rows, problems = [], []
    for v in order:
        g = sorted((f for f in fits if f["variant"] == v), key=lambda f: f["run"])
        if not g:
            continue
        if v.startswith("erf"):
            fps = {json.dumps(f["fingerprint"], sort_keys=True) for f in g}
            identical = len(fps) == 1
            if not identical:
                problems.append(v)
        else:
            identical = None
        f = g[0]
        t = f.get("timings") or {}
        rows.append({
            "variant": v, "runs": len(g), "identical": identical, "auc": f["auc"], "brier": f["brier"],
            "n_terms": f["n_terms"], "n_rules": f["n_rules"], "n_linear": f["n_linear"],
            "mean_rule_len": f["mean_rule_len"],
            "active_rules_per_case_mean": f["active_rules_per_case_mean"],
            "active_rules_per_case_max": f["active_rules_per_case_max"],
            "confirmatory_coef": f.get("confirmatory_coef"),
            "confirmatory_sign_ok": f.get("confirmatory_sign_ok"),
            "confirmatory_bound_active": f.get("confirmatory_bound_active"),
            "confirmatory_included": f.get("confirmatory_included"),
            "optional_active": f.get("optional_active"),
            "convergence_warnings": f["convergence_warnings"],
            "fit_seconds": [x["fit_seconds"] for x in g],
            "timings": t, "constrained_info": f.get("constrained_info"),
        })
    (out / f"{dataset}_summary.json").write_text(json.dumps(rows, indent=1))

    md = ["| Variant | Runs identical | AUC | Brier | Terms (rules + linear) | Mean rule length | Active rules per case (mean / max) "
          "| Confirmatory coef | Sign / bound | Conv. warnings | Fit s | rule gen / bootstrap / final CV / refit-constr. (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["timings"]
        stage = (" / ".join(f"{t.get(k, 0):.0f}" for k in ("rule_generation", "bootstrap", "final_cv", "refit_or_constrained"))
                 if t else "–")
        cc = r["confirmatory_coef"]
        sign = ("bound active" if r["confirmatory_bound_active"] else
                ("OK" if r["confirmatory_sign_ok"] else "contradicted") if r["confirmatory_sign_ok"] is not None else "–")
        ident = "–" if r["identical"] is None else (f"{r['runs']}/{r['runs']}" if r["identical"] else "NO")
        rl = "–" if r["mean_rule_len"] is None else "%.2f" % r["mean_rule_len"]
        cc_txt = "–" if cc is None else "%+.3f" % cc
        md.append(
            f"| {r['variant']} | {ident} | "
            f"{r['auc']:.3f} | {r['brier']:.4f} | {r['n_terms']} ({r['n_rules']} + {r['n_linear']}) | "
            f"{rl} | {r['active_rules_per_case_mean']:.1f} / {r['active_rules_per_case_max']} | "
            f"{cc_txt} | {sign} | {r['convergence_warnings']} | "
            f"{min(r['fit_seconds']):.0f} | {stage} |")
    (out / f"{dataset}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if problems:
        raise AssertionError(f"runs not identical: {problems}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--variants", nargs="+", default=list(ERF_VARIANTS) + list(OTHER))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "results" / "ablation"))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}.jsonl"
    if not args.summarize_only:
        done = set()
        if path.exists():
            done = {(f["variant"], f["run"]) for f in map(json.loads, path.read_text().splitlines())}
        tasks = [{"dataset": args.dataset, "variant": v, "run": r, "data_dir": args.data_dir}
                 for v in args.variants for r in range(args.runs if v.startswith("erf") else 1)]
        tasks = [t for t in tasks if (t["variant"], t["run"]) not in done]
        print(f"{len(tasks)} fits to run", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.jobs) as ex:
            for fut in as_completed([ex.submit(run, t) for t in tasks]):
                r = fut.result()
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  {r['variant']} run {r['run']}: AUC {r['auc']:.3f}, {r['n_terms']} terms, "
                      f"{r['fit_seconds']}s [{time.perf_counter() - t0:.0f}s]", flush=True)
    summarize(out, args.dataset)


if __name__ == "__main__":
    main()
