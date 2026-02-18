"""
EBM → ExpertRuleFit Pipeline Demo
==================================
1. Fit EBM to discover feature interactions (GA2M)
2. Extract top interactions as confirmatory rules
3. Fit ExpertRuleFit with guaranteed rule preservation
4. Compare AUC and verify confirmatory rules survive

Usage: python examples/ebm_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from expertrulefit import ExpertRuleFit, discover_interaction_rules


def generate_credit_data(n_samples=1000, random_state=42):
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

    # Ground truth includes an INTERACTION: cash_ratio * night_tx_ratio
    logit = (0.8 * (X[:, 3] > 0.4)
             + 0.5 * (X[:, 4] > 0.15)
             + 0.3 * (X[:, 3] > 0.3) * (X[:, 4] > 0.1)  # interaction!
             + 0.1 * (X[:, 0] > 2.5)
             + rng.normal(0, 0.3, n) - 0.8)
    y = (logit > 0).astype(int)
    return X, y, feature_names


def main():
    print("=" * 65)
    print("EBM → ExpertRuleFit Pipeline")
    print("=" * 65)

    X, y, fn = generate_credit_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ── Step 1: Discover interactions with EBM ──────────────────────
    print("\n[Step 1] Fitting EBM to discover interactions...")
    rules, ebm = discover_interaction_rules(
        X_train, y_train, fn,
        top_k=3,
        rule_type="confirmatory",
        max_interactions=10,
    )

    from expertrulefit.ebm_bridge import summarize_ebm_interactions
    summarize_ebm_interactions(ebm)

    print(f"\nExtracted {len(rules)} interaction rules:")
    for r in rules:
        print(f"  → {r['name']}")

    # ── Step 2: ExpertRuleFit WITHOUT confirmatory rules ────────────
    print("\n[Step 2] ExpertRuleFit (auto rules only)...")
    erf_auto = ExpertRuleFit(max_rules=50, n_bootstrap=10)
    erf_auto.fit(X_train, y_train, feature_names=fn)
    proba_auto = erf_auto.predict_proba(X_test)[:, 1]
    auc_auto = roc_auc_score(y_test, proba_auto)
    n_rules_auto = len(erf_auto.get_selected_rules())
    print(f"  AUC = {auc_auto:.4f}, Active rules = {n_rules_auto}")

    # ── Step 3: ExpertRuleFit WITH EBM confirmatory rules ───────────
    print("\n[Step 3] ExpertRuleFit + EBM confirmatory rules...")
    erf_ebm = ExpertRuleFit(max_rules=50, n_bootstrap=10)
    erf_ebm.fit(X_train, y_train, feature_names=fn, confirmatory_rules=rules)
    proba_ebm = erf_ebm.predict_proba(X_test)[:, 1]
    auc_ebm = roc_auc_score(y_test, proba_ebm)
    n_rules_ebm = len(erf_ebm.get_selected_rules())
    print(f"  AUC = {auc_ebm:.4f}, Active rules = {n_rules_ebm}")
    print(f"  Confirmatory preserved: {erf_ebm.confirmatory_all_active_}")

    # ── Step 4: Verify reproducibility ──────────────────────────────
    print("\n[Step 4] Reproducibility check (5 seeds)...")
    rule_sets = []
    for seed in range(5):
        erf = ExpertRuleFit(max_rules=50, n_bootstrap=10, random_state=seed)
        erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=rules)
        rule_sets.append(frozenset(erf.get_selected_rules()))
        assert erf.confirmatory_all_active_, f"Seed {seed}: confirmatory rule lost!"
    unique = len(set(rule_sets))
    print(f"  {unique} unique rule set(s) across 5 seeds")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Auto-only:    AUC={auc_auto:.4f}, {n_rules_auto} rules")
    print(f"  EBM+Confirm:  AUC={auc_ebm:.4f}, {n_rules_ebm} rules")
    delta = auc_ebm - auc_auto
    print(f"  AUC delta:    {delta:+.4f}")
    print(f"  Reproducible: {unique}/5 unique rule sets")
    print(f"  Confirmatory: ALL PRESERVED")
    print()

    # Full model summary
    erf_ebm.summary()


if __name__ == "__main__":
    main()
