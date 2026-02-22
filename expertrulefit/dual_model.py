"""Dual Architecture: EBM + ExpertRuleFit + Logistic Regression Stacking.

Implements the architecture from the BIL technical specification (§5):
    ① Features → EBM scoring (p_ebm) + ExpertRuleFit scoring (p_erf)
    ② [p_ebm, p_erf] → Meta-classifier (Logistic Regression) → Score final
    ③ Score final + explanation (shape functions + active rules + meta-weights)

The meta-classifier uses cross-validated out-of-fold predictions to avoid
overfitting the stacking layer. Both base models are refitted on full data
for production inference.

References:
    - Lou et al. (2013) "Accurate Intelligible Models with Pairwise Interactions"
    - Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles"
    - Wolpert (1992) "Stacked Generalization"
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from .expert_rulefit import ExpertRuleFit


def _get_interpret_version():
    """Return the installed InterpretML version string, or None."""
    try:
        from importlib.metadata import version as _meta_version
        return _meta_version("interpret")
    except Exception:
        return None


def _extract_ebm_local(ebm_local, sample_idx):
    """Extract feature names and scores from an EBM local explanation.

    Compatibility layer that isolates InterpretML private API access so
    the rest of the codebase never touches ``_internal_obj`` directly.

    Supported interpret versions:
        - **>= 0.3**: uses the public ``.data(sample_idx)`` API
        - **< 0.3** (legacy): falls back to the private
          ``_internal_obj["specific"][sample_idx]`` dict

    Parameters
    ----------
    ebm_local : EBMExplanation
        Result of ``ebm.explain_local(X)``.
    sample_idx : int
        Index of the sample within the explanation object.

    Returns
    -------
    names : list of str
    scores : ndarray
    """
    # ---- Public API path (interpret >= 0.3) ----
    if hasattr(ebm_local, "data") and callable(ebm_local.data):
        try:
            data = ebm_local.data(sample_idx)
            if isinstance(data, dict) and "names" in data and "scores" in data:
                return data["names"], np.asarray(data["scores"], dtype=np.float64)
        except Exception:
            pass  # fall through to legacy path

    # ---- Legacy path (interpret < 0.3 or non-standard structure) ----
    try:
        obj = ebm_local._internal_obj["specific"][sample_idx]
        return obj["names"], np.asarray(obj["scores"], dtype=np.float64)
    except (AttributeError, KeyError, TypeError, IndexError) as exc:
        interpret_ver = _get_interpret_version() or "unknown"
        raise RuntimeError(
            f"Cannot extract local EBM explanation for sample {sample_idx}. "
            f"Installed InterpretML version: {interpret_ver}. "
            f"Supported versions: >= 0.3 (public .data() API) and "
            f"< 0.3 (legacy _internal_obj). If you are on a newer version "
            f"whose API has changed, please open an issue. "
            f"Underlying error: {exc}"
        ) from exc


class DualModel(BaseEstimator, ClassifierMixin):
    """EBM + ExpertRuleFit stacking for regulated banking environments.

    Combines two interpretable models via a logistic regression meta-classifier:
    - **EBM** captures smooth non-linear effects and pairwise interactions
    - **ExpertRuleFit** captures discrete rules + confirmatory regulatory rules
    - **Meta-classifier** weights their contributions (2 coefficients + intercept)

    Every layer is interpretable — no post-hoc approximations.

    Parameters
    ----------
    ebm_params : dict, optional
        Parameters for ExplainableBoostingClassifier.
        Default: max_bins=256, interactions=10, learning_rate=0.01,
                 outer_bags=25, min_samples_leaf=2, n_jobs=1.

    erf_params : dict, optional
        Parameters for ExpertRuleFit.
        Default: max_rules=50, n_bootstrap=10, rule_threshold=0.8.

    meta_cv : int, default=5
        Number of folds for cross-validated stacking predictions.

    random_state : int, default=42
        Random state for reproducibility.

    Examples
    --------
    >>> from expertrulefit import DualModel
    >>> dm = DualModel()
    >>> dm.fit(X_train, y_train, feature_names=fn, confirmatory_rules=rules)
    >>> proba = dm.predict_proba(X_test)[:, 1]
    >>> dm.explain(X_test[:1])  # per-sample explanation
    """

    def __init__(
        self,
        ebm_params=None,
        erf_params=None,
        meta_cv=5,
        random_state=42,
    ):
        self.ebm_params = ebm_params
        self.erf_params = erf_params
        self.meta_cv = meta_cv
        self.random_state = random_state

    def fit(self, X, y, feature_names=None, confirmatory_rules=None, optional_rules=None):
        """Fit the dual architecture with cross-validated stacking.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
        confirmatory_rules : list of dict, optional
            Regulatory rules for ExpertRuleFit (never eliminated).
        optional_rules : list of dict, optional
            Analyst rules for ExpertRuleFit (reduced penalty).

        Returns
        -------
        self
        """
        try:
            import pandas as pd
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError:
            raise ImportError(
                "interpret and pandas required: pip install interpret pandas"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = list(feature_names)

        confirmatory_rules = confirmatory_rules or []
        optional_rules = optional_rules or []

        # Default parameters matching the spec (§3.2)
        ebm_defaults = dict(
            max_bins=256, interactions=10, learning_rate=0.01,
            outer_bags=25, min_samples_leaf=2, n_jobs=1,
            random_state=self.random_state,
        )
        if self.ebm_params:
            ebm_defaults.update(self.ebm_params)

        erf_defaults = dict(max_rules=50, n_bootstrap=10, rule_threshold=0.8)
        if self.erf_params:
            erf_defaults.update(self.erf_params)

        # Step 1: Cross-validated out-of-fold predictions for stacking
        oof_ebm = np.zeros(len(X))
        oof_erf = np.zeros(len(X))
        skf = StratifiedKFold(
            n_splits=self.meta_cv, shuffle=True, random_state=self.random_state
        )

        df = pd.DataFrame(X, columns=feature_names)

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            df_tr = pd.DataFrame(X_tr, columns=feature_names)
            df_val = pd.DataFrame(X_val, columns=feature_names)

            # EBM fold
            ebm_fold = ExplainableBoostingClassifier(**ebm_defaults)
            ebm_fold.fit(df_tr, y_tr)
            oof_ebm[val_idx] = ebm_fold.predict_proba(df_val)[:, 1]

            # ExpertRuleFit fold
            erf_fold = ExpertRuleFit(**erf_defaults)
            erf_fold.fit(
                X_tr, y_tr, feature_names=feature_names,
                confirmatory_rules=confirmatory_rules,
                optional_rules=optional_rules,
            )
            oof_erf[val_idx] = erf_fold.predict_proba(X_val)[:, 1]

        # Step 2: Fit meta-classifier on OOF predictions
        meta_X = np.column_stack([oof_ebm, oof_erf])
        self.meta_model_ = LogisticRegressionCV(
            cv=5, random_state=self.random_state, max_iter=1000,
        )
        self.meta_model_.fit(meta_X, y)

        # Step 3: Refit both models on full data for production
        self.ebm_ = ExplainableBoostingClassifier(**ebm_defaults)
        self.ebm_.fit(df, y)

        self.erf_ = ExpertRuleFit(**erf_defaults)
        self.erf_.fit(
            X, y, feature_names=feature_names,
            confirmatory_rules=confirmatory_rules,
            optional_rules=optional_rules,
        )

        # Store stacking metadata
        self.meta_weights_ = {
            "ebm_weight": float(self.meta_model_.coef_[0, 0]),
            "erf_weight": float(self.meta_model_.coef_[0, 1]),
            "intercept": float(self.meta_model_.intercept_[0]),
        }
        self.confirmatory_all_active_ = self.erf_.confirmatory_all_active_

        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        check_is_fitted(self, ["ebm_", "erf_", "meta_model_"])
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities via stacking.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, 2)
        """
        check_is_fitted(self, ["ebm_", "erf_", "meta_model_"])
        import pandas as pd

        X = np.asarray(X, dtype=np.float64, copy=False)
        df = pd.DataFrame(X, columns=self.feature_names_)

        p_ebm = self.ebm_.predict_proba(df)[:, 1]
        p_erf = self.erf_.predict_proba(X)[:, 1]

        meta_X = np.column_stack([p_ebm, p_erf])
        return self.meta_model_.predict_proba(meta_X)

    def explain(self, X, top_n=5):
        """Generate per-sample explanations from both models.

        Returns a list of explanation dicts (one per sample), each containing:
        - ebm_score, erf_score, final_score
        - meta_weights (how EBM vs ERF are combined)
        - ebm_contributions: top feature contributions from EBM shape functions
        - erf_active_rules: active rules with coefficients from ExpertRuleFit
        - confirmatory_status: which confirmatory rules are active

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        top_n : int, default=5
            Number of top EBM feature contributions to include.

        Returns
        -------
        explanations : list of dict
        """
        check_is_fitted(self, ["ebm_", "erf_", "meta_model_"])
        import pandas as pd

        X = np.asarray(X, dtype=np.float64, copy=False)
        df = pd.DataFrame(X, columns=self.feature_names_)

        p_ebm = self.ebm_.predict_proba(df)[:, 1]
        p_erf = self.erf_.predict_proba(X)[:, 1]
        meta_X = np.column_stack([p_ebm, p_erf])
        p_final = self.meta_model_.predict_proba(meta_X)[:, 1]

        # EBM local explanations
        ebm_local = self.ebm_.explain_local(df)

        # ExpertRuleFit rule importance (global — same for all samples)
        erf_importance = self.erf_.get_rule_importance()

        # Precompute ERF matrix once for all samples (vectorized)
        stable_indices = np.where(self.erf_.stable_mask_)[0]
        true_coefs = self.erf_._get_true_coefficients()
        X_pred_all = self.erf_._build_predict_matrix(X)

        # Precompute active coefficient mask
        active_coef_mask = np.zeros(len(true_coefs), dtype=bool)
        for j in range(len(true_coefs)):
            active_coef_mask[j] = abs(true_coefs[j]) > 1e-10

        explanations = []
        for i in range(X.shape[0]):
            # EBM: extract per-feature contributions for this sample
            # Prefer the public .data() API (interpret >= 0.3);
            # fall back to the legacy _internal_obj if unavailable.
            ebm_names, ebm_scores = _extract_ebm_local(ebm_local, i)

            # Sort by absolute contribution, take top_n
            order = np.argsort(np.abs(ebm_scores))[::-1]
            ebm_contribs = [
                {"feature": ebm_names[j], "contribution": float(ebm_scores[j])}
                for j in order[:top_n]
            ]

            # ERF: find which rules are active for this sample (no recomputation)
            erf_active = []
            for j, idx in enumerate(stable_indices):
                if j < len(true_coefs) and active_coef_mask[j]:
                    if abs(X_pred_all[i, j]) > 1e-10:
                        name = self.erf_.rule_names_[idx]
                        category = "confirmatory" if name.startswith("confirmatory:") else \
                                   "optional" if name.startswith("optional:") else "auto"
                        erf_active.append({
                            "rule": name,
                            "coefficient": float(true_coefs[j]),
                            "category": category,
                        })

            explanations.append({
                "sample_index": i,
                "ebm_score": float(p_ebm[i]),
                "erf_score": float(p_erf[i]),
                "final_score": float(p_final[i]),
                "meta_weights": self.meta_weights_,
                "ebm_top_contributions": ebm_contribs,
                "erf_active_rules": erf_active,
                "confirmatory_all_active": self.confirmatory_all_active_,
            })

        return explanations

    def summary(self):
        """Print a summary of the dual model."""
        check_is_fitted(self, ["ebm_", "erf_", "meta_model_"])

        print("=" * 65)
        print("DualModel — EBM + ExpertRuleFit Stacking Summary")
        print("=" * 65)

        # Meta-classifier weights
        w = self.meta_weights_
        total = abs(w["ebm_weight"]) + abs(w["erf_weight"])
        ebm_pct = abs(w["ebm_weight"]) / total * 100 if total > 0 else 50
        erf_pct = abs(w["erf_weight"]) / total * 100 if total > 0 else 50
        print(f"\nMeta-classifier weights:")
        print(f"  EBM:           {w['ebm_weight']:+.4f} ({ebm_pct:.0f}%)")
        print(f"  ExpertRuleFit: {w['erf_weight']:+.4f} ({erf_pct:.0f}%)")
        print(f"  Intercept:     {w['intercept']:+.4f}")

        # EBM summary
        importances = self.ebm_.term_importances()
        n_main = sum(1 for tf in self.ebm_.term_features_ if len(tf) == 1)
        n_inter = sum(1 for tf in self.ebm_.term_features_ if len(tf) == 2)
        print(f"\nEBM component:")
        print(f"  Main effects: {n_main}, Interactions: {n_inter}")
        # Top features
        pairs = [(self.ebm_.term_names_[i], importances[i])
                 for i in range(len(importances))]
        pairs.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top features:")
        for name, imp in pairs[:5]:
            print(f"    {imp:.4f} | {name}")

        # ExpertRuleFit summary
        print(f"\nExpertRuleFit component:")
        self.erf_.summary()

        print("=" * 65)
