"""
Validation: ExpertRuleFit vs Standard Lasso — 100 Random Seeds

This script demonstrates the key property:
    - Weighted Lasso (ExpertRuleFit) preserves confirmatory rules ~100% of seeds
    - Standard Lasso eliminates weak confirmatory rules ~50% of seeds

Uses synthetic banking data where confirmatory rules have WEAK but real signal,
which is the realistic scenario: regulatory rules may not be the strongest
predictors, but they MUST be preserved for compliance.

References:
    - Zou (2006) "The Adaptive LASSO and Its Oracle Properties", JASA
    - Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles"
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def generate_banking_data(n_samples=2000, random_state=42):
    """Generate synthetic banking data for KYC/AML scoring.

    Key design: confirmatory rules have WEAK signal (coef ~0.1) while
    other features are STRONG (coef ~0.8). This creates the realistic
    scenario where standard Lasso drops the weak regulatory rules.
    """
    rng = np.random.RandomState(random_state)
    n = n_samples

    data = {
        "country_risk": rng.uniform(0, 5, n),
        "payment_delay": rng.exponential(30, n),
        "tx_volume": rng.lognormal(8, 1.5, n),
        "cash_ratio": rng.beta(2, 5, n),
        "night_tx_ratio": rng.beta(1.5, 8, n),
        "tenure": rng.uniform(0, 20, n),
        "income_declared": rng.lognormal(10.5, 0.8, n),
        "n_accounts": rng.poisson(2, n) + 1,
    }
    X = pd.DataFrame(data)

    # True risk function:
    # - Confirmatory rules: VERY WEAK signal — Lasso WILL drop these
    # - Other features: STRONG signal — Lasso keeps these
    # This simulates the realistic banking scenario: regulatory rules
    # capture edge cases that are statistically rare but legally required.
    logit = (
        0.03 * (X["country_risk"] > 2.5).astype(float)    # VERY WEAK regulatory
        + 0.02 * (X["payment_delay"] > 60).astype(float)   # VERY WEAK regulatory
        + 1.20 * (X["cash_ratio"] > 0.4).astype(float)     # STRONG data signal
        + 0.90 * (X["night_tx_ratio"] > 0.15).astype(float) # STRONG
        + 0.50 * np.log1p(X["tx_volume"]) / 10             # moderate continuous
        - 0.40 * (X["tenure"] > 10).astype(float)           # moderate
        + 0.30 * (X["n_accounts"] > 3).astype(float)        # moderate
        + rng.normal(0, 0.5, n)                              # noise
        - 1.5                                                # base rate
    )
    y = (logit > 0).astype(int)

    return X.values, y, list(X.columns)


def build_expert_features(X, feature_names, confirmatory_rules, optional_rules):
    """Build augmented feature matrix with expert rule columns."""
    rule_cols = []
    for rule in confirmatory_rules + optional_rules:
        col = rule["evaluate"](X, feature_names)
        rule_cols.append(col.reshape(-1, 1))
    if rule_cols:
        return np.hstack([X] + rule_cols)
    return X


def weighted_lasso_fit(X_aug, y, n_original, n_confirmatory, n_optional,
                       conf_penalty=1e-6, opt_penalty=0.3, auto_penalty=1.0,
                       C=1.0, random_state=None):
    """Fit logistic regression with per-feature penalty weighting.

    Trick: first standardize, THEN scale by 1/w_j. This ensures the
    weighting isn't undone by standardization.
    """
    n_total = X_aug.shape[1]

    # Build penalty weight vector
    weights = np.ones(n_total)
    weights[:n_original] = auto_penalty
    weights[n_original:n_original + n_confirmatory] = conf_penalty
    weights[n_original + n_confirmatory:n_original + n_confirmatory + n_optional] = opt_penalty

    # Step 1: Standardize first
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_aug)

    # Step 2: THEN apply weighted scaling (1/sqrt(w) so effective penalty = lambda * w)
    inv_sqrt_w = 1.0 / np.sqrt(np.maximum(weights, 1e-12))
    X_weighted = X_std * inv_sqrt_w[np.newaxis, :]

    model = LogisticRegression(
        solver="saga", l1_ratio=1.0, C=C,
        max_iter=10000, random_state=random_state,
    )
    model.fit(X_weighted, y)

    # Transform coefficients back to original space:
    # X_weighted = X_std * inv_sqrt_w = (X - mu) / sigma * inv_sqrt_w
    # logit = coef_w @ X_weighted + intercept_w
    #       = coef_w * inv_sqrt_w / sigma @ X + (intercept_w - coef_w * inv_sqrt_w * mu / sigma)
    coefs_original = model.coef_[0] * inv_sqrt_w / scaler.scale_
    intercept_original = model.intercept_[0] - np.sum(model.coef_[0] * inv_sqrt_w * scaler.mean_ / scaler.scale_)

    return coefs_original, intercept_original, model


def run_validation(n_seeds=100, verbose=True):
    """Run the 100-seed validation experiment."""
    if verbose:
        print("=" * 70)
        print("ExpertRuleFit Validation — 100 Random Seeds")
        print("=" * 70)
        print()

    # Confirmatory rules (CSSF regulatory — MUST be preserved)
    confirmatory_rules = [
        {
            "name": "CSSF: Country risk > 2.5 (sanctions proxy)",
            "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 2.5).astype(float),
        },
        {
            "name": "CSSF: Payment delay > 60 days (severe delinquency)",
            "evaluate": lambda X, fn: (X[:, fn.index("payment_delay")] > 60).astype(float),
        },
    ]

    # Optional rules (analyst-suggested)
    optional_rules = [
        {
            "name": "Analyst: Night TX ratio > 0.15 AND cash ratio > 0.3",
            "evaluate": lambda X, fn: (
                (X[:, fn.index("night_tx_ratio")] > 0.15)
                & (X[:, fn.index("cash_ratio")] > 0.3)
            ).astype(float),
        },
    ]

    n_conf = len(confirmatory_rules)
    n_opt = len(optional_rules)

    weighted_preserved = 0
    standard_preserved = 0
    weighted_aucs = []
    standard_aucs = []

    for seed in range(n_seeds):
        X, y, feature_names = generate_banking_data(n_samples=2000, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        n_orig = X_train.shape[1]

        # Build augmented matrices
        X_train_aug = build_expert_features(X_train, feature_names, confirmatory_rules, optional_rules)
        X_test_aug = build_expert_features(X_test, feature_names, confirmatory_rules, optional_rules)

        # --- WEIGHTED Lasso (ExpertRuleFit) ---
        try:
            coefs_w, intercept_w, _ = weighted_lasso_fit(
                X_train_aug, y_train, n_orig, n_conf, n_opt,
                conf_penalty=1e-6, opt_penalty=0.3, auto_penalty=1.0,
                C=0.1, random_state=seed,
            )
            logits_w = X_test_aug @ coefs_w + intercept_w
            proba_w = 1.0 / (1.0 + np.exp(-logits_w))
            auc_w = roc_auc_score(y_test, proba_w)
            weighted_aucs.append(auc_w)

            conf_coefs_w = coefs_w[n_orig:n_orig + n_conf]
            if all(abs(c) > 1e-10 for c in conf_coefs_w):
                weighted_preserved += 1
        except Exception as e:
            if verbose:
                print(f"  Seed {seed}: Weighted error — {e}")

        # --- STANDARD Lasso (uniform penalties) ---
        try:
            scaler_s = StandardScaler()
            X_train_std = scaler_s.fit_transform(X_train_aug)
            X_test_std = scaler_s.transform(X_test_aug)

            model_s = LogisticRegression(
                solver="saga", l1_ratio=1.0, C=0.1,
                max_iter=10000, random_state=seed,
            )
            model_s.fit(X_train_std, y_train)
            proba_s = model_s.predict_proba(X_test_std)[:, 1]
            auc_s = roc_auc_score(y_test, proba_s)
            standard_aucs.append(auc_s)

            # Check confirmatory coefs in original scale
            coefs_s = model_s.coef_[0] / scaler_s.scale_
            conf_coefs_s = coefs_s[n_orig:n_orig + n_conf]
            if all(abs(c) > 1e-10 for c in conf_coefs_s):
                standard_preserved += 1
        except Exception as e:
            if verbose:
                print(f"  Seed {seed}: Standard error — {e}")

        if verbose and (seed + 1) % 20 == 0:
            print(f"  Seeds {seed + 1}/{n_seeds} complete...")

    # Results
    avg_auc_w = np.mean(weighted_aucs) if weighted_aucs else 0
    avg_auc_s = np.mean(standard_aucs) if standard_aucs else 0
    std_auc_w = np.std(weighted_aucs) if weighted_aucs else 0
    std_auc_s = np.std(standard_aucs) if standard_aucs else 0

    if verbose:
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"{'Metric':<45} {'Weighted':>14} {'Standard':>14}")
        print("-" * 73)
        print(f"{'Confirmatory rules preserved':<45} {weighted_preserved:>10}/100 {standard_preserved:>10}/100")
        print(f"{'Preservation rate':<45} {weighted_preserved:>13}% {standard_preserved:>13}%")
        print(f"{'Mean AUC-ROC':<45} {avg_auc_w:>14.4f} {avg_auc_s:>14.4f}")
        print(f"{'Std AUC-ROC':<45} {std_auc_w:>14.4f} {std_auc_s:>14.4f}")
        print(f"{'AUC delta':<45} {avg_auc_w - avg_auc_s:>+14.4f}")
        print()
        if weighted_preserved >= 95:
            print(f"PASS: Weighted Lasso preserves confirmatory rules {weighted_preserved}% of the time.")
        else:
            print(f"NOTE: Weighted preserved {weighted_preserved}% — tune conf_penalty if needed.")
        if standard_preserved < 80:
            print(f"PASS: Standard Lasso drops confirmatory rules {100 - standard_preserved}% of the time.")
        else:
            print(f"NOTE: Standard preserved {standard_preserved}% — confirmatory signal may be too strong.")
        print()

    return {
        "weighted_preserved": weighted_preserved,
        "standard_preserved": standard_preserved,
        "weighted_auc_mean": avg_auc_w,
        "standard_auc_mean": avg_auc_s,
    }


if __name__ == "__main__":
    results = run_validation(n_seeds=100, verbose=True)
