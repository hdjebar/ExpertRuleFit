"""
Dual Model Demo: EBM + ExpertRuleFit Stacking
==============================================
Full pipeline from the BIL technical specification (§5):
    ① EBM (continuous shape functions) + ExpertRuleFit (discrete rules)
    ② Meta-classifier (logistic regression) combines both scores
    ③ Per-sample explanations: shape functions + active rules + meta-weights

Usage: python examples/dual_model_demo.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from expertrulefit import DualModel, discover_interaction_rules


def generate_credit_data(n_samples=1500, random_state=42):
    """Generate synthetic credit scoring data with non-linear effects + interactions."""
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

    # Non-linear effects (EBM should capture these)
    # + interaction (ExpertRuleFit confirmatory)
    # + discrete thresholds (ExpertRuleFit rules)
    logit = (
        0.8 * (X[:, 3] > 0.4)                              # discrete threshold
        + 0.5 * (X[:, 4] > 0.15)                            # discrete threshold
        + 0.3 * (X[:, 3] > 0.3) * (X[:, 4] > 0.1)          # interaction
        + 0.2 * np.log1p(X[:, 1]) / 4                       # non-linear (EBM captures)
        + 0.1 * (X[:, 0] > 2.5)                             # country risk
        - 0.15 * np.minimum(X[:, 5], 10) / 10               # tenure protective
        + rng.normal(0, 0.3, n) - 0.8
    )
    y = (logit > 0).astype(int)
    return X, y, feature_names


def main():
    print("=" * 65)
    print("Dual Model — EBM + ExpertRuleFit Stacking Demo")
    print("=" * 65)

    X, y, fn = generate_credit_data(n_samples=1500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nData: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{y.mean():.1%} positive rate")

    # ── Define regulatory confirmatory rules ────────────────────────
    confirmatory = [
        {
            "name": "CSSF: High cash ratio",
            "evaluate": lambda X, fn: (X[:, fn.index("cash_ratio")] > 0.4).astype(float),
        },
        {
            "name": "AML: Country risk threshold",
            "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 2.5).astype(float),
        },
    ]

    # ── Fit Dual Model ──────────────────────────────────────────────
    print("\n[Fitting] EBM + ExpertRuleFit + Meta-classifier...")
    dm = DualModel(
        ebm_params=dict(interactions=5, outer_bags=10),  # faster for demo
        erf_params=dict(max_rules=50, n_bootstrap=10),
        meta_cv=5,
    )
    dm.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)

    # ── Evaluate ────────────────────────────────────────────────────
    proba = dm.predict_proba(X_test)[:, 1]
    auc_dual = roc_auc_score(y_test, proba)

    # Compare with individual models
    from interpret.glassbox import ExplainableBoostingClassifier
    import pandas as pd
    df_train = pd.DataFrame(X_train, columns=fn)
    df_test = pd.DataFrame(X_test, columns=fn)

    ebm_solo = ExplainableBoostingClassifier(
        interactions=5, outer_bags=10, random_state=42, n_jobs=1
    )
    ebm_solo.fit(df_train, y_train)
    auc_ebm = roc_auc_score(y_test, ebm_solo.predict_proba(df_test)[:, 1])

    from expertrulefit import ExpertRuleFit
    erf_solo = ExpertRuleFit(max_rules=50, n_bootstrap=10)
    erf_solo.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)
    auc_erf = roc_auc_score(y_test, erf_solo.predict_proba(X_test)[:, 1])

    print(f"\n{'─' * 50}")
    print(f"  EBM alone:        AUC = {auc_ebm:.4f}")
    print(f"  ExpertRuleFit:    AUC = {auc_erf:.4f}")
    print(f"  Dual (stacking):  AUC = {auc_dual:.4f}")
    print(f"{'─' * 50}")

    # ── Model Summary ───────────────────────────────────────────────
    print()
    dm.summary()

    # ── Per-Sample Explanation ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("Per-Sample Explanation (sample 0)")
    print("=" * 65)
    explanations = dm.explain(X_test[:1], top_n=5)
    exp = explanations[0]

    print(f"\n  Final score:  {exp['final_score']:.4f}")
    print(f"  EBM score:    {exp['ebm_score']:.4f}")
    print(f"  ERF score:    {exp['erf_score']:.4f}")

    w = exp["meta_weights"]
    print(f"\n  Meta-weights: EBM={w['ebm_weight']:+.3f}, "
          f"ERF={w['erf_weight']:+.3f}, intercept={w['intercept']:+.3f}")

    print(f"\n  EBM top contributions:")
    for c in exp["ebm_top_contributions"]:
        print(f"    {c['contribution']:+.4f} | {c['feature']}")

    print(f"\n  ExpertRuleFit active rules:")
    if exp["erf_active_rules"]:
        for r in exp["erf_active_rules"]:
            print(f"    coef={r['coefficient']:+.6f} [{r['category']}] {r['rule']}")
    else:
        print(f"    (no rules active for this sample)")

    print(f"\n  Confirmatory rules preserved: {exp['confirmatory_all_active']}")
    print()


if __name__ == "__main__":
    main()
