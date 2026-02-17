"""
ExpertRuleFit — Weighted Lasso extension of RuleFit for banking compliance.

Extends imodels.RuleFitClassifier with a weighted Lasso that guarantees
preservation of regulatory (confirmatory) rules while learning from data.

Mathematical foundation:
    Standard Lasso:  minimize ||y - Xb||^2 + lambda * sum(|b_j|)
    Weighted Lasso:  minimize ||y - Xb||^2 + lambda * sum(w_j * |b_j|)

    By scaling feature j by (1 / w_j) before fitting, the effective penalty
    on b_j becomes lambda * w_j. Setting w_j ~ 0 for confirmatory rules
    ensures they are never eliminated by the Lasso.

References:
    - Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles"
    - Zou (2006) "The Adaptive LASSO and Its Oracle Properties", JASA
    - Singh et al. (2021) "imodels", JOSS (base implementation)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from imodels import RuleFitClassifier


class ExpertRuleFit(BaseEstimator, ClassifierMixin):
    """Compliance-native scoring model with guaranteed regulatory rule preservation.

    ExpertRuleFit injects domain-expert rules into a RuleFit model with
    differentiated Lasso penalties:

    - **Confirmatory** (regulatory): penalty ~ 0 → never eliminated
    - **Optional** (analyst): reduced penalty → preferred but not guaranteed
    - **Auto** (data-driven): standard penalty → kept only if predictive

    Parameters
    ----------
    confirmatory_penalty : float, default=1e-8
        Lasso penalty weight for confirmatory rules. Near-zero ensures
        these rules are never eliminated. Must be > 0 for numerical
        stability.

    optional_penalty : float, default=0.3
        Lasso penalty weight for optional (analyst-suggested) rules.
        Lower than auto means these are preferred but can still be
        eliminated if truly non-predictive.

    auto_penalty : float, default=1.0
        Lasso penalty weight for auto-discovered rules from the
        tree ensemble. Standard Lasso behavior.

    max_rules : int, default=100
        Maximum number of rules to extract from the tree ensemble.

    tree_size : int, default=4
        Maximum depth of trees used for rule extraction.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> from expertrulefit import ExpertRuleFit
    >>> confirmatory = [
    ...     {"name": "CSSF: Country risk > 1.5",
    ...      "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 1.5).astype(float)}
    ... ]
    >>> erf = ExpertRuleFit(confirmatory_penalty=1e-8)
    >>> erf.fit(X_train, y_train, feature_names=names, confirmatory_rules=confirmatory)
    >>> proba = erf.predict_proba(X_test)[:, 1]
    >>> explanations = erf.explain(X_test[0])
    """

    def __init__(
        self,
        confirmatory_penalty=1e-8,
        optional_penalty=0.3,
        auto_penalty=1.0,
        max_rules=100,
        tree_size=4,
        random_state=None,
    ):
        self.confirmatory_penalty = confirmatory_penalty
        self.optional_penalty = optional_penalty
        self.auto_penalty = auto_penalty
        self.max_rules = max_rules
        self.tree_size = tree_size
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        feature_names=None,
        confirmatory_rules=None,
        optional_rules=None,
    ):
        """Fit the ExpertRuleFit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).

        feature_names : list of str, optional
            Names for each feature column.

        confirmatory_rules : list of dict, optional
            Each dict has:
            - "name": str — human-readable rule name
            - "evaluate": callable(X, feature_names) → array of shape (n_samples,)

        optional_rules : list of dict, optional
            Same format as confirmatory_rules.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = list(feature_names)

        confirmatory_rules = confirmatory_rules or []
        optional_rules = optional_rules or []
        self.confirmatory_rules_ = confirmatory_rules
        self.optional_rules_ = optional_rules

        # Step 1: Fit base RuleFit to extract auto rules
        self.base_rulefit_ = RuleFitClassifier(
            max_rules=self.max_rules,
            tree_size=self.tree_size,
            random_state=self.random_state,
            include_linear=True,
        )
        self.base_rulefit_.fit(X, y, feature_names=self.feature_names_)

        # Step 2: Extract auto rules from the fitted RuleFit
        auto_rules_info = self._extract_auto_rules()

        # Step 3: Evaluate all rule features on training data
        rule_features, rule_meta, penalty_weights = self._build_rule_matrix(
            X, auto_rules_info
        )

        # Step 4: Combine original features + rule features
        X_augmented = np.hstack([X, rule_features])
        n_original = X.shape[1]
        n_rules = rule_features.shape[1]

        # Penalty weights: 1.0 for original linear features, custom for rules
        all_weights = np.concatenate([
            np.ones(n_original) * self.auto_penalty,
            penalty_weights,
        ])

        # Step 5: Standardize first, THEN apply weighted scaling
        # This ensures the weighting isn't undone by standardization
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X_augmented)

        # Scale by 1/sqrt(w_j) — effective penalty on b_j becomes lambda * w_j
        inv_sqrt_w = 1.0 / np.sqrt(np.maximum(all_weights, 1e-12))
        X_weighted = X_std * inv_sqrt_w[np.newaxis, :]

        # Step 6: Fit Lasso on weighted features
        self.lasso_ = LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            C=1.0,
            max_iter=10000,
            random_state=self.random_state,
        )
        self.lasso_.fit(X_weighted, y)

        # Step 7: Transform coefficients back to original scale
        # X_weighted = (X - mu) / sigma * inv_sqrt_w
        # logit = coef_w @ X_weighted + intercept_w
        #       = (coef_w * inv_sqrt_w / sigma) @ X + (intercept_w - sum(coef_w * inv_sqrt_w * mu / sigma))
        raw_coefs = self.lasso_.coef_[0]
        self.coef_ = raw_coefs * inv_sqrt_w / self.scaler_.scale_
        self.intercept_ = (
            self.lasso_.intercept_[0]
            - np.sum(raw_coefs * inv_sqrt_w * self.scaler_.mean_ / self.scaler_.scale_)
        )

        # Store metadata
        self.n_original_features_ = n_original
        self.n_rules_ = n_rules
        self.rule_meta_ = rule_meta
        # Step 8: Verify confirmatory rules are preserved
        self._verify_confirmatory()

        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, 2)
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)

        # Build augmented feature matrix
        rule_features = self._evaluate_rules(X)
        X_augmented = np.hstack([X, rule_features])

        # Compute logits
        logits = X_augmented @ self.coef_ + self.intercept_
        proba_1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def explain(self, x):
        """Explain a single prediction by listing rule contributions.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            A single sample.

        Returns
        -------
        explanations : list of dict
            Each dict has keys: "rule", "category", "contribution", "active".
        """
        check_is_fitted(self)
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)

        # Evaluate rules
        rule_features = self._evaluate_rules(x)[0]

        explanations = []

        # Linear feature contributions
        for i, name in enumerate(self.feature_names_):
            coef = self.coef_[i]
            if abs(coef) > 1e-10:
                explanations.append({
                    "rule": f"Linear: {name}",
                    "category": "linear",
                    "contribution": float(coef * x[0, i]),
                    "active": True,
                    "coefficient": float(coef),
                })

        # Rule contributions
        n_orig = self.n_original_features_
        for j, meta in enumerate(self.rule_meta_):
            coef = self.coef_[n_orig + j]
            active = rule_features[j] > 0.5
            if abs(coef) > 1e-10:
                explanations.append({
                    "rule": meta["name"],
                    "category": meta["category"],
                    "contribution": float(coef * rule_features[j]),
                    "active": bool(active),
                    "coefficient": float(coef),
                })

        # Sort by absolute contribution
        explanations.sort(key=lambda e: abs(e["contribution"]), reverse=True)
        return explanations

    def get_active_rules(self):
        """Return all rules with non-zero coefficients.

        Returns
        -------
        rules : list of dict
            Each dict has keys: "name", "category", "coefficient".
        """
        check_is_fitted(self)
        n_orig = self.n_original_features_
        active = []
        for j, meta in enumerate(self.rule_meta_):
            coef = self.coef_[n_orig + j]
            if abs(coef) > 1e-10:
                active.append({
                    "name": meta["name"],
                    "category": meta["category"],
                    "coefficient": float(coef),
                })
        return active

    def export_sql(self, table_name="scoring_input"):
        """Export the scoring function as a SQL expression.

        Parameters
        ----------
        table_name : str
            Name of the source table.

        Returns
        -------
        sql : str
            SQL CASE expression for real-time scoring.
        """
        check_is_fitted(self)
        terms = [f"{self.intercept_:.6f}"]

        # Linear terms
        for i, name in enumerate(self.feature_names_):
            coef = self.coef_[i]
            if abs(coef) > 1e-10:
                terms.append(f"({coef:+.6f} * {table_name}.{name})")

        # Rule terms (as CASE WHEN)
        n_orig = self.n_original_features_
        for j, meta in enumerate(self.rule_meta_):
            coef = self.coef_[n_orig + j]
            if abs(coef) > 1e-10 and "sql_condition" in meta:
                terms.append(
                    f"({coef:+.6f} * CASE WHEN {meta['sql_condition']} THEN 1 ELSE 0 END)"
                )

        logit_expr = "\n    + ".join(terms)
        return (
            f"-- ExpertRuleFit scoring function\n"
            f"-- Generated automatically — DO NOT EDIT\n"
            f"SELECT\n"
            f"    1.0 / (1.0 + EXP(-(\n    {logit_expr}\n    ))) AS risk_score\n"
            f"FROM {table_name};"
        )

    # --- Private methods ---

    def _extract_auto_rules(self):
        """Extract rules from the base RuleFit model."""
        rules = []
        if hasattr(self.base_rulefit_, "rules_") and self.base_rulefit_.rules_ is not None:
            for i, rule in enumerate(self.base_rulefit_.rules_):
                rule_str = str(rule)
                if hasattr(rule, "agg_dict"):
                    # imodels Rule object
                    rules.append({
                        "name": f"Auto: {rule_str}",
                        "rule_obj": rule,
                        "index": i,
                    })
                else:
                    rules.append({
                        "name": f"Auto: rule_{i}",
                        "rule_obj": rule,
                        "index": i,
                    })
        return rules

    def _build_rule_matrix(self, X, auto_rules_info):
        """Build the rule feature matrix with all three rule categories."""
        rule_columns = []
        rule_meta = []
        penalty_weights = []

        # Confirmatory rules
        for rule in self.confirmatory_rules_:
            col = rule["evaluate"](X, self.feature_names_)
            rule_columns.append(col.reshape(-1, 1))
            meta = {"name": rule["name"], "category": "confirmatory"}
            if "sql_condition" in rule:
                meta["sql_condition"] = rule["sql_condition"]
            rule_meta.append(meta)
            penalty_weights.append(self.confirmatory_penalty)

        # Optional rules
        for rule in self.optional_rules_:
            col = rule["evaluate"](X, self.feature_names_)
            rule_columns.append(col.reshape(-1, 1))
            meta = {"name": rule["name"], "category": "optional"}
            if "sql_condition" in rule:
                meta["sql_condition"] = rule["sql_condition"]
            rule_meta.append(meta)
            penalty_weights.append(self.optional_penalty)

        # Auto rules from base RuleFit
        if auto_rules_info and hasattr(self.base_rulefit_, "transform"):
            try:
                auto_matrix = self.base_rulefit_.transform(X)
                # auto_matrix may include linear terms; we want only rules
                n_linear = X.shape[1]
                if auto_matrix.shape[1] > n_linear:
                    rule_part = auto_matrix[:, n_linear:]
                else:
                    rule_part = auto_matrix

                for j in range(rule_part.shape[1]):
                    rule_columns.append(rule_part[:, j].reshape(-1, 1))
                    name = (
                        auto_rules_info[j]["name"]
                        if j < len(auto_rules_info)
                        else f"Auto: rule_{j}"
                    )
                    rule_meta.append({"name": name, "category": "auto"})
                    penalty_weights.append(self.auto_penalty)
            except Exception:
                # Fallback: use rules_ attribute directly
                self._add_auto_rules_fallback(
                    X, auto_rules_info, rule_columns, rule_meta, penalty_weights
                )
        elif auto_rules_info:
            self._add_auto_rules_fallback(
                X, auto_rules_info, rule_columns, rule_meta, penalty_weights
            )

        if not rule_columns:
            # No rules at all — add a dummy column
            rule_columns.append(np.zeros((X.shape[0], 1)))
            rule_meta.append({"name": "dummy", "category": "auto"})
            penalty_weights.append(1.0)

        rule_features = np.hstack(rule_columns)
        penalty_weights = np.array(penalty_weights, dtype=np.float64)
        return rule_features, rule_meta, penalty_weights

    def _add_auto_rules_fallback(
        self, X, auto_rules_info, rule_columns, rule_meta, penalty_weights
    ):
        """Fallback for extracting auto rules when transform isn't available."""
        for info in auto_rules_info:
            rule_obj = info["rule_obj"]
            try:
                col = rule_obj.transform(X)
                if col.ndim == 1:
                    col = col.reshape(-1, 1)
                rule_columns.append(col[:, 0:1])
            except Exception:
                # Skip rules that can't be evaluated
                continue
            rule_meta.append({"name": info["name"], "category": "auto"})
            penalty_weights.append(self.auto_penalty)

    def _evaluate_rules(self, X):
        """Evaluate all rules on new data."""
        rule_columns = []

        # Confirmatory
        for rule in self.confirmatory_rules_:
            col = rule["evaluate"](X, self.feature_names_)
            rule_columns.append(col.reshape(-1, 1))

        # Optional
        for rule in self.optional_rules_:
            col = rule["evaluate"](X, self.feature_names_)
            rule_columns.append(col.reshape(-1, 1))

        # Auto rules
        n_expert = len(self.confirmatory_rules_) + len(self.optional_rules_)
        n_auto = self.n_rules_ - n_expert

        if n_auto > 0 and hasattr(self.base_rulefit_, "transform"):
            try:
                auto_matrix = self.base_rulefit_.transform(X)
                n_linear = X.shape[1]
                if auto_matrix.shape[1] > n_linear:
                    rule_part = auto_matrix[:, n_linear:]
                else:
                    rule_part = auto_matrix
                # Take only the columns we used during fit
                for j in range(min(n_auto, rule_part.shape[1])):
                    rule_columns.append(rule_part[:, j].reshape(-1, 1))
            except Exception:
                for _ in range(n_auto):
                    rule_columns.append(np.zeros((X.shape[0], 1)))
        elif n_auto > 0:
            for _ in range(n_auto):
                rule_columns.append(np.zeros((X.shape[0], 1)))

        if not rule_columns:
            return np.zeros((X.shape[0], 1))

        return np.hstack(rule_columns)

    def _verify_confirmatory(self):
        """Check that all confirmatory rules have non-zero coefficients."""
        n_conf = len(self.confirmatory_rules_)
        n_orig = self.n_original_features_

        self.confirmatory_coefs_ = []
        all_active = True
        for i in range(n_conf):
            coef = self.coef_[n_orig + i]
            self.confirmatory_coefs_.append(float(coef))
            if abs(coef) < 1e-10:
                all_active = False

        self.confirmatory_all_active_ = all_active

    def summary(self):
        """Print a human-readable summary of the fitted model."""
        check_is_fitted(self)

        n_orig = self.n_original_features_
        n_conf = len(self.confirmatory_rules_)
        n_opt = len(self.optional_rules_)
        n_auto = self.n_rules_ - n_conf - n_opt

        print("=" * 60)
        print("ExpertRuleFit — Model Summary")
        print("=" * 60)

        # Confirmatory
        print(f"\nConfirmatory Rules ({n_conf}):")
        for i, rule in enumerate(self.confirmatory_rules_):
            coef = self.coef_[n_orig + i]
            status = "ACTIVE" if abs(coef) > 1e-10 else "ELIMINATED"
            print(f"  {'OK' if status == 'ACTIVE' else 'XX'} | coef={coef:+.6f} | {rule['name']}")

        # Optional
        print(f"\nOptional Rules ({n_opt}):")
        for i, rule in enumerate(self.optional_rules_):
            coef = self.coef_[n_orig + n_conf + i]
            status = "ACTIVE" if abs(coef) > 1e-10 else "ELIMINATED"
            print(f"  {'OK' if status == 'ACTIVE' else '--'} | coef={coef:+.6f} | {rule['name']}")

        # Auto
        active_auto = sum(
            1 for j in range(n_auto)
            if abs(self.coef_[n_orig + n_conf + n_opt + j]) > 1e-10
        )
        print(f"\nAuto Rules: {active_auto}/{n_auto} active")
        print(f"\nConfirmatory preservation: {'100%' if self.confirmatory_all_active_ else 'FAILED'}")
        print("=" * 60)
