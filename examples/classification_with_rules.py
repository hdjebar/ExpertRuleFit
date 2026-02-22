"""
Classification with Expert Rules
=================================
Demonstrates how confirmatory and optional rules affect classification:
    1. Fit ExpertRuleFit *without* expert rules (auto-only)
    2. Fit with confirmatory rules (regulatory — guaranteed to survive)
    3. Fit with optional rules (analyst-suggested — may be eliminated)
    4. Compare predictions, AUC, and active rule sets

Usage: python examples/classification_with_rules.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from expertrulefit import ExpertRuleFit


def generate_credit_data(n_samples=1000, random_state=42):
    """Generate synthetic credit scoring data."""
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

    logit = (0.8 * (X[:, 3] > 0.4) + 0.5 * (X[:, 4] > 0.15)
             + 0.3 * (X[:, 3] > 0.3) * (X[:, 4] > 0.1)
             + 0.1 * (X[:, 0] > 2.5) + rng.normal(0, 0.3, n) - 0.8)
    y = (logit > 0).astype(int)
    return X, y, feature_names


def main():
    print("=" * 65)
    print("ExpertRuleFit — Classification with Expert Rules")
    print("=" * 65)

    X, y, fn = generate_credit_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nData: {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{y.mean():.1%} positive rate\n")

    # ── Define expert rules ──────────────────────────────────────
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

    optional_rules = [
        {
            "name": "Analyst: Night tx ratio",
            "evaluate": lambda X, fn: (X[:, fn.index("night_tx_ratio")] > 0.15).astype(float),
        },
    ]

    # ══════════════════════════════════════════════════════════════
    # Scenario 1: Auto rules only
    # ══════════════════════════════════════════════════════════════
    print("[Scenario 1] Auto rules only (no expert knowledge)")
    erf_auto = ExpertRuleFit(max_rules=50, n_bootstrap=10)
    erf_auto.fit(X_train, y_train, feature_names=fn)

    proba_auto = erf_auto.predict_proba(X_test)[:, 1]
    auc_auto = roc_auc_score(y_test, proba_auto)
    acc_auto = accuracy_score(y_test, erf_auto.predict(X_test))
    rules_auto = erf_auto.get_selected_rules()
    print(f"  AUC={auc_auto:.4f}  Acc={acc_auto:.4f}  Active rules={len(rules_auto)}")

    # ══════════════════════════════════════════════════════════════
    # Scenario 2: With confirmatory rules (regulatory)
    # ══════════════════════════════════════════════════════════════
    print("\n[Scenario 2] With confirmatory rules (regulatory — guaranteed)")
    erf_conf = ExpertRuleFit(max_rules=50, n_bootstrap=10)
    erf_conf.fit(X_train, y_train, feature_names=fn,
                 confirmatory_rules=confirmatory_rules)

    proba_conf = erf_conf.predict_proba(X_test)[:, 1]
    auc_conf = roc_auc_score(y_test, proba_conf)
    acc_conf = accuracy_score(y_test, erf_conf.predict(X_test))
    rules_conf = erf_conf.get_selected_rules()
    print(f"  AUC={auc_conf:.4f}  Acc={acc_conf:.4f}  Active rules={len(rules_conf)}")
    print(f"  Confirmatory all active: {erf_conf.confirmatory_all_active_}")

    # Show confirmatory rule status
    for s in erf_conf.confirmatory_status_:
        status = "ACTIVE" if s["active"] else "INACTIVE"
        print(f"    [{status}] {s['name']}")

    # ══════════════════════════════════════════════════════════════
    # Scenario 3: With confirmatory + optional rules
    # ══════════════════════════════════════════════════════════════
    print("\n[Scenario 3] Confirmatory + optional rules")
    erf_full = ExpertRuleFit(max_rules=50, n_bootstrap=10)
    erf_full.fit(X_train, y_train, feature_names=fn,
                 confirmatory_rules=confirmatory_rules,
                 optional_rules=optional_rules)

    proba_full = erf_full.predict_proba(X_test)[:, 1]
    auc_full = roc_auc_score(y_test, proba_full)
    acc_full = accuracy_score(y_test, erf_full.predict(X_test))
    rules_full = erf_full.get_selected_rules()
    print(f"  AUC={auc_full:.4f}  Acc={acc_full:.4f}  Active rules={len(rules_full)}")
    print(f"  Confirmatory all active: {erf_full.confirmatory_all_active_}")

    # Check if optional rule survived
    optional_active = any("optional:" in r for r in rules_full)
    print(f"  Optional rule survived: {optional_active}")

    # ══════════════════════════════════════════════════════════════
    # Comparison summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Scenario':<35} {'AUC':>8} {'Acc':>8} {'Rules':>7}")
    print(f"  {'─' * 60}")
    print(f"  {'Auto rules only':<35} {auc_auto:>8.4f} {acc_auto:>8.4f} {len(rules_auto):>7}")
    print(f"  {'+ confirmatory':<35} {auc_conf:>8.4f} {acc_conf:>8.4f} {len(rules_conf):>7}")
    print(f"  {'+ confirmatory + optional':<35} {auc_full:>8.4f} {acc_full:>8.4f} {len(rules_full):>7}")

    # ══════════════════════════════════════════════════════════════
    # Prediction differences between scenarios
    # ══════════════════════════════════════════════════════════════
    diff = np.abs(proba_auto - proba_conf)
    print(f"\n  P(y=1) difference (auto vs confirmatory):")
    print(f"    mean={diff.mean():.4f}  max={diff.max():.4f}")

    diff2 = np.abs(proba_conf - proba_full)
    print(f"  P(y=1) difference (confirmatory vs full):")
    print(f"    mean={diff2.mean():.4f}  max={diff2.max():.4f}")

    # ══════════════════════════════════════════════════════════════
    # Full model summary (scenario 3)
    # ══════════════════════════════════════════════════════════════
    print()
    erf_full.summary()


if __name__ == "__main__":
    main()
