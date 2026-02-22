"""
Quick Demo: ExpertRuleFit Reproducibility
==========================================
Shows that ExpertRuleFit produces identical rules across 10 random seeds,
while standard RuleFit produces different rules each time.

Usage: python examples/quick_demo.py
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from expertrulefit import ExpertRuleFit
from imodels import RuleFitClassifier


def generate_credit_data(n_samples=1000, random_state=42):
    """Generate synthetic credit scoring data."""
    rng = np.random.RandomState(random_state)
    n = n_samples
    X = np.column_stack([
        rng.uniform(0, 5, n),
        rng.exponential(30, n),
        rng.lognormal(8, 1.5, n),
        rng.beta(2, 5, n),
        rng.beta(1.5, 8, n),
        rng.uniform(0, 20, n),
    ])
    feature_names = ["country_risk", "payment_delay", "tx_volume",
                     "cash_ratio", "night_tx_ratio", "tenure"]

    logit = (0.8 * (X[:, 3] > 0.4) + 0.5 * (X[:, 4] > 0.15)
             + 0.1 * (X[:, 0] > 2.5) + rng.normal(0, 0.3, n) - 0.8)
    y = (logit > 0).astype(int)
    return X, y, feature_names


def extract_rulefit_rules(model):
    """Extract rules with non-zero coefficients from RuleFit."""
    try:
        rules_df = model._get_rules()
        rules = set()
        for _, row in rules_df.iterrows():
            if abs(row["coef"]) > 1e-10:
                rules.add(str(row["rule"])[:80])
        return rules
    except Exception:
        return set()


def main():
    N_SEEDS = 10
    print("=" * 60)
    print("ExpertRuleFit — Quick Reproducibility Demo")
    print("=" * 60)

    X, y, fn = generate_credit_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- Standard RuleFit ---
    print(f"\nRuleFit Standard ({N_SEEDS} seeds):")
    rf_hashes = []
    rf_aucs = []
    for seed in range(N_SEEDS):
        rf = RuleFitClassifier(max_rules=50, random_state=seed)
        rf.fit(X_train, y_train, feature_names=fn)
        rules = extract_rulefit_rules(rf)
        h = hashlib.md5(str(sorted(rules)).encode()).hexdigest()[:8]
        rf_hashes.append(h)
        try:
            from expertrulefit.expert_rulefit import build_rule_feature_matrix
            X_r = build_rule_feature_matrix(rf, X_test)
            # Use _get_rules to get predictions
            preds = rf.predict(X_test)
            auc = roc_auc_score(y_test, preds)
            rf_aucs.append(auc)
        except Exception:
            pass
        print(f"  seed={seed:2d}  rules={len(rules):2d}  hash={h}")

    unique_rf = len(set(rf_hashes))
    print(f"  → {unique_rf} unique rule sets out of {N_SEEDS}")

    # --- ExpertRuleFit ---
    print(f"\nExpertRuleFit ({N_SEEDS} seeds):")
    erf_hashes = []
    erf_aucs = []
    for seed in range(N_SEEDS):
        erf = ExpertRuleFit(max_rules=50, n_bootstrap=10, random_state=seed)
        erf.fit(X_train, y_train, feature_names=fn)
        rules = erf.get_selected_rules()
        h = hashlib.md5(str(sorted(rules)).encode()).hexdigest()[:8]
        erf_hashes.append(h)
        try:
            proba = erf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            erf_aucs.append(auc)
        except Exception:
            pass
        print(f"  seed={seed:2d}  rules={len(rules):2d}  hash={h}")

    unique_erf = len(set(erf_hashes))
    print(f"  → {unique_erf} unique rule set(s) out of {N_SEEDS}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  RuleFit Standard:  {unique_rf:2d}/{N_SEEDS} unique rule sets (unstable)")
    print(f"  ExpertRuleFit:     {unique_erf:2d}/{N_SEEDS} unique rule set(s) (stable)")
    if unique_erf == 1:
        print(f"\n  ExpertRuleFit achieves PERFECT reproducibility!")
    print()


if __name__ == "__main__":
    main()
