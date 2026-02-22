"""
Prediction Basics: predict() and predict_proba()
=================================================
Demonstrates the core prediction pipeline of ExpertRuleFit:
    1. Fit the model on binary classification data
    2. predict()       -> hard class labels {0, 1}
    3. predict_proba() -> calibrated probabilities [P(y=0), P(y=1)]
    4. Evaluate with standard sklearn metrics (AUC, accuracy, confusion matrix)

Usage: python examples/prediction_basics.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

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
             + 0.1 * (X[:, 0] > 2.5) + rng.normal(0, 0.3, n) - 0.8)
    y = (logit > 0).astype(int)
    return X, y, feature_names


def main():
    print("=" * 60)
    print("ExpertRuleFit — Prediction Basics")
    print("=" * 60)

    # ── Generate data and split ──────────────────────────────────
    X, y, fn = generate_credit_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nData: {X_train.shape[0]} train / {X_test.shape[0]} test, "
          f"{y.mean():.1%} positive rate")

    # ── Fit ──────────────────────────────────────────────────────
    print("\n[1] Fitting ExpertRuleFit...")
    erf = ExpertRuleFit(max_rules=50, n_bootstrap=10, rule_threshold=0.8)
    erf.fit(X_train, y_train, feature_names=fn)
    print(f"    Stable features: {erf.n_stable_rules_}")

    # ── predict(): hard labels ───────────────────────────────────
    print("\n[2] predict() -> hard class labels {0, 1}")
    y_pred = erf.predict(X_test)
    print(f"    Shape:   {y_pred.shape}")
    print(f"    Unique:  {np.unique(y_pred)}")
    print(f"    Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # ── predict_proba(): calibrated probabilities ────────────────
    print("\n[3] predict_proba() -> calibrated probabilities")
    proba = erf.predict_proba(X_test)
    print(f"    Shape:   {proba.shape}  (n_samples, 2)")
    print(f"    Columns: P(y=0), P(y=1)")
    print(f"    Row sums to 1: {np.allclose(proba.sum(axis=1), 1.0)}")
    print(f"    P(y=1) range: [{proba[:, 1].min():.4f}, {proba[:, 1].max():.4f}]")

    auc = roc_auc_score(y_test, proba[:, 1])
    print(f"    AUC:     {auc:.4f}")

    # ── Sample predictions ───────────────────────────────────────
    print("\n[4] Sample predictions (first 5 test samples)")
    print(f"    {'True':>6}  {'Pred':>6}  {'P(y=1)':>8}")
    print(f"    {'─' * 24}")
    for i in range(5):
        print(f"    {int(y_test[i]):>6d}  {int(y_pred[i]):>6d}  {proba[i, 1]:>8.4f}")

    # ── Confusion matrix ─────────────────────────────────────────
    print("\n[5] Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
    print(f"    FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")

    # ── Classification report ────────────────────────────────────
    print("\n[6] Classification report")
    print(classification_report(y_test, y_pred, target_names=["class 0", "class 1"]))

    # ── Custom threshold ─────────────────────────────────────────
    print("[7] Custom decision threshold")
    for threshold in [0.3, 0.5, 0.7]:
        y_custom = (proba[:, 1] >= threshold).astype(int)
        acc = accuracy_score(y_test, y_custom)
        cm_t = confusion_matrix(y_test, y_custom)
        print(f"    threshold={threshold:.1f}  acc={acc:.4f}  "
              f"TP={cm_t[1,1]:3d}  FP={cm_t[0,1]:3d}  "
              f"FN={cm_t[1,0]:3d}  TN={cm_t[0,0]:3d}")

    print()


if __name__ == "__main__":
    main()
