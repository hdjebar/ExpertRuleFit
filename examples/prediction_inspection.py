"""
Prediction Inspection: Understanding What Drives Predictions
=============================================================
Demonstrates how to inspect and explain predictions:
    1. Rule importance (sorted by absolute coefficient)
    2. Per-sample active rules (which rules fire for a given sample)
    3. Bootstrap stability of each rule
    4. DualModel per-sample explanation (EBM + ERF + meta-weights)

Usage: python examples/prediction_inspection.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from expertrulefit import ExpertRuleFit, DualModel


def generate_credit_data(n_samples=1500, random_state=42):
    """Generate synthetic credit scoring data with interactions."""
    rng = np.random.RandomState(random_state)
    n = n_samples
    X = np.column_stack([
        rng.uniform(0, 5, n),       # country_risk
        rng.exponential(30, n),      # payment_delay
        rng.lognormal(8, 1.5, n),   # tx_volume
        rng.beta(2, 5, n),          # cash_ratio
        rng.beta(1.5, 8, n),        # night_tx_ratio
        rng.uniform(0, 20, n),      # tenure
    ])
    feature_names = ["country_risk", "payment_delay", "tx_volume",
                     "cash_ratio", "night_tx_ratio", "tenure"]

    logit = (
        0.8 * (X[:, 3] > 0.4)
        + 0.5 * (X[:, 4] > 0.15)
        + 0.3 * (X[:, 3] > 0.3) * (X[:, 4] > 0.1)
        + 0.2 * np.log1p(X[:, 1]) / 4
        + 0.1 * (X[:, 0] > 2.5)
        - 0.15 * np.minimum(X[:, 5], 10) / 10
        + rng.normal(0, 0.3, n) - 0.8
    )
    y = (logit > 0).astype(int)
    return X, y, feature_names


def main():
    print("=" * 65)
    print("ExpertRuleFit — Prediction Inspection")
    print("=" * 65)

    X, y, fn = generate_credit_data(n_samples=1500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    confirmatory_rules = [
        {
            "name": "CSSF: High cash ratio",
            "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
        },
        {
            "name": "BCBS: Country risk threshold",
            "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 2.5).astype(float),
        },
    ]

    # ── Fit ExpertRuleFit ────────────────────────────────────────
    print("\n[Fitting ExpertRuleFit]")
    erf = ExpertRuleFit(max_rules=50, n_bootstrap=10)
    erf.fit(X_train, y_train, feature_names=fn,
            confirmatory_rules=confirmatory_rules)

    auc = roc_auc_score(y_test, erf.predict_proba(X_test)[:, 1])
    print(f"  AUC = {auc:.4f}")

    # ══════════════════════════════════════════════════════════════
    # 1. Rule importance
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("1. RULE IMPORTANCE (sorted by |coefficient|)")
    print(f"{'=' * 65}")

    importance = erf.get_rule_importance()
    print(f"\n  {'Rank':>4}  {'Coef':>10}  {'Boot%':>6}  {'Category':<14}  Name")
    print(f"  {'─' * 60}")
    for rank, r in enumerate(importance, 1):
        print(f"  {rank:>4}  {r['coefficient']:>+10.6f}  "
              f"{r['bootstrap_frequency']:>5.0%}  "
              f"{r['category']:<14}  {r['name']}")

    # ══════════════════════════════════════════════════════════════
    # 2. Per-sample active rules
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("2. PER-SAMPLE ACTIVE RULES")
    print(f"{'=' * 65}")

    stable_indices = np.where(erf.stable_mask_)[0]
    true_coefs = erf._get_true_coefficients()

    for sample_idx in [0, 1, 2]:
        x = X_test[sample_idx:sample_idx + 1]
        proba = erf.predict_proba(x)[0, 1]
        label = erf.predict(x)[0]

        print(f"\n  Sample {sample_idx}: P(y=1)={proba:.4f}, predicted={label}, "
              f"true={y_test[sample_idx]}")

        # Evaluate each stable rule on this sample
        from expertrulefit.expert_rulefit import build_rule_feature_matrix
        from scipy import sparse

        X_rules = build_rule_feature_matrix(erf.base_rulefit_, x)
        expert_cols = []
        for rule in erf.confirmatory_rules_:
            col = np.asarray(
                rule["evaluate"](x, erf.feature_names_), dtype=np.float64
            )
            expert_cols.append(sparse.csc_matrix(col).T)
        for rule in erf.optional_rules_:
            col = np.asarray(
                rule["evaluate"](x, erf.feature_names_), dtype=np.float64
            )
            expert_cols.append(sparse.csc_matrix(col).T)

        if expert_cols:
            X_all = sparse.hstack([X_rules] + expert_cols, format="csr")
        else:
            X_all = X_rules

        X_stable = X_all[:, erf.stable_mask_]
        if sparse.issparse(X_stable):
            values = np.asarray(X_stable.todense()).ravel()
        else:
            values = np.asarray(X_stable).ravel()

        print(f"  Active rules for this sample:")
        found_any = False
        for j, idx in enumerate(stable_indices):
            if j < len(true_coefs) and abs(true_coefs[j]) > 1e-10:
                val = values[j] if j < len(values) else 0
                if abs(val) > 1e-10:  # rule fires for this sample
                    contribution = true_coefs[j] * val
                    print(f"    {contribution:>+10.6f} | {erf.rule_names_[idx]}")
                    found_any = True
        if not found_any:
            print(f"    (no rules active)")

    # ══════════════════════════════════════════════════════════════
    # 3. Bootstrap stability
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("3. BOOTSTRAP STABILITY")
    print(f"{'=' * 65}")

    freqs = erf.bootstrap_frequencies_
    n_above_80 = (freqs >= 0.8).sum()
    n_above_50 = (freqs >= 0.5).sum()
    n_total = len(freqs)

    print(f"\n  Total candidate features:  {n_total}")
    print(f"  Selected (>=80% bootstrap): {n_above_80}")
    print(f"  Moderate (>=50% bootstrap): {n_above_50}")
    print(f"  Active (non-zero coef):     {len(erf.get_selected_rules())}")

    # Distribution of bootstrap frequencies
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    print(f"\n  Bootstrap frequency distribution:")
    for i in range(len(bins) - 1):
        count = ((freqs >= bins[i]) & (freqs < bins[i + 1])).sum()
        bar = "#" * count
        print(f"    {labels[i]:>8}: {count:3d} {bar}")

    # ══════════════════════════════════════════════════════════════
    # 4. DualModel per-sample explanation
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("4. DUALMODEL PER-SAMPLE EXPLANATION")
    print(f"{'=' * 65}")

    try:
        import interpret  # noqa: F401
    except ImportError:
        print("\n  Skipped: requires 'interpret' package.")
        print("  Install with: pip install interpret pandas")
        print()
        return

    print("\n  [Fitting DualModel: EBM + ExpertRuleFit + Meta-classifier]")
    dm = DualModel(
        ebm_params=dict(interactions=5, outer_bags=10),
        erf_params=dict(max_rules=50, n_bootstrap=10),
    )
    dm.fit(X_train, y_train, feature_names=fn,
           confirmatory_rules=confirmatory_rules)

    auc_dm = roc_auc_score(y_test, dm.predict_proba(X_test)[:, 1])
    print(f"  DualModel AUC = {auc_dm:.4f}")

    # Explain sample 0
    explanations = dm.explain(X_test[:1], top_n=5)
    exp = explanations[0]

    print(f"\n  Sample 0 explanation:")
    print(f"    Final P(y=1):  {exp['final_score']:.4f}")
    print(f"    EBM score:     {exp['ebm_score']:.4f}")
    print(f"    ERF score:     {exp['erf_score']:.4f}")

    w = exp["meta_weights"]
    print(f"\n    Meta-weights:")
    print(f"      EBM:       {w['ebm_weight']:+.4f}")
    print(f"      ERF:       {w['erf_weight']:+.4f}")
    print(f"      Intercept: {w['intercept']:+.4f}")

    print(f"\n    EBM top feature contributions:")
    for c in exp["ebm_top_contributions"]:
        print(f"      {c['contribution']:+.4f} | {c['feature']}")

    print(f"\n    ExpertRuleFit active rules:")
    if exp["erf_active_rules"]:
        for r in exp["erf_active_rules"]:
            print(f"      coef={r['coefficient']:+.6f} [{r['category']}] {r['rule']}")
    else:
        print(f"      (no rules active for this sample)")

    print(f"\n    Confirmatory rules preserved: {exp['confirmatory_all_active']}")
    print()


if __name__ == "__main__":
    main()
