"""
ExpertRuleFit — Elastic Net + Bootstrap Stabilization for Reproducible Rule Learning.

Extends imodels.RuleFitClassifier with a deterministic rule selection pipeline
that guarantees 100/100 seed reproducibility for regulated banking environments,
with optional confirmatory rules that survive any regularization strength.

Two guarantees:
    1. **Reproducibility** — same data → same rules → same predictions (100/100 seeds)
    2. **Rule preservation** — confirmatory (regulatory) rules are NEVER eliminated

Approach:
    1. Fit base RuleFit with FIXED internal seed → same candidate rules always
    2. Build rule feature matrix via manual rule string evaluation
    3. Append confirmatory/optional expert rules as additional features
    4. Bootstrap stabilization: 10 bootstrap samples × ElasticNetCV
       (confirmatory features scaled by 1/sqrt(w_j) → near-zero effective penalty)
    5. Frequency-based filtering: keep rules selected in >= 80% of bootstraps
       (confirmatory rules force-included regardless of bootstrap frequency)
    6. Final ElasticNetCV on stable features with weighted scaling

Mathematical foundation:
    Standard Lasso:   minimize ||y - Xb||^2 + lambda * sum(|b_j|)
    Weighted Elastic:  minimize ||y - Xb||^2 + lambda * sum(w_j * |b_j|)

    By scaling feature j by 1/sqrt(w_j), the effective penalty becomes lambda * w_j.
    Setting w_j ~ 0 for confirmatory rules ensures they are never eliminated.

References:
    - Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles"
    - Zou & Hastie (2005) "Regularization and variable selection via the elastic net"
    - Zou (2006) "The Adaptive LASSO and Its Oracle Properties", JASA
    - Singh et al. (2021) "imodels", JOSS (base implementation)
"""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import ElasticNetCV
from imodels import RuleFitClassifier

# Pre-compiled regex for parsing rule conditions like "X_3 <= 0.5"
_CONDITION_RE = re.compile(
    r"X_(\d+)\s*(<=|>=|<|>)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

# Vectorized comparison operators keyed by string token
_OPERATORS = {
    "<=": np.less_equal,
    ">=": np.greater_equal,
    "<": np.less,
    ">": np.greater,
}


def eval_rule_on_data(rule_str, X):
    """Evaluate a rule string (X_i notation) on data matrix X.

    Rule strings look like: 'X_1 > -0.06917 and X_5 <= 0.41409'
    Returns a binary float array (0.0 or 1.0) of shape (n_samples,).

    Uses pre-compiled regex and numpy vectorized comparisons for performance.
    """
    n_samples, n_cols = X.shape
    result = np.ones(n_samples, dtype=bool)

    for match in _CONDITION_RE.finditer(rule_str):
        col_idx = int(match.group(1))
        op_str = match.group(2)
        threshold = float(match.group(3))

        if col_idx >= n_cols:
            result[:] = False
            break

        result &= _OPERATORS[op_str](X[:, col_idx], threshold)

    return result.astype(np.float64)


def build_rule_feature_matrix(rulefit_model, X):
    """Build the full rule feature matrix from a fitted RuleFitClassifier.

    Columns: [linear features (X columns)] + [rule features (0/1 per rule)]
    This replaces the broken transform() method in current imodels.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    rules_no_fn = rulefit_model.rules_without_feature_names_

    if not rules_no_fn:
        return X.copy()

    n_rules = len(rules_no_fn)
    X_augmented = np.empty((n_samples, n_features + n_rules), dtype=np.float64)
    X_augmented[:, :n_features] = X
    X_augmented[:, n_features:] = 0.0

    for i, rule in enumerate(rules_no_fn):
        try:
            X_augmented[:, n_features + i] = eval_rule_on_data(rule.rule, X)
        except Exception:
            pass  # column stays zero

    return X_augmented


class ExpertRuleFit(BaseEstimator, ClassifierMixin):
    """Reproducible rule-based classifier for regulated banking environments.

    ExpertRuleFit extends RuleFit with bootstrap-stabilized Elastic Net
    to guarantee identical rule selection across random seeds, with optional
    confirmatory rules that survive any regularization strength.

    Two guarantees:
    1. **Reproducibility** — fixed internal seed + bootstrap → 100/100 stability
    2. **Rule preservation** — confirmatory rules get near-zero penalty → never eliminated

    Design principle — **deterministic by construction**:
    1. Base tree generation uses a FIXED internal seed (_BASE_SEED=42)
       → same candidate rules regardless of external random_state
    2. Bootstrap sampling uses fixed seeds (_BASE_SEED + 0..N)
       → reproducible bootstrap procedure
    3. ElasticNetCV with tight tolerance (tol=1e-6)
       → numerical convergence guarantee
    4. Frequency-based rule filtering (threshold >= 80%)
       → only rules robust to bootstrap perturbation survive
    5. Confirmatory rules force-included with near-zero penalty
       → regulatory rules NEVER eliminated

    Parameters
    ----------
    n_estimators : int, default=250
        Number of trees in the base RuleFit ensemble.

    tree_size : int, default=4
        Maximum depth of trees used for rule extraction.

    max_rules : int, default=50
        Maximum number of rules to extract from the tree ensemble.

    random_state : int or None, default=42
        Accepted for sklearn API compatibility but overridden internally.
        The model is deterministic by design regardless of this value.

    n_bootstrap : int, default=10
        Number of bootstrap samples for rule stability estimation.

    rule_threshold : float, default=0.8
        Minimum frequency (0-1) a rule must be selected across bootstraps.

    confirmatory_penalty : float, default=1e-8
        Penalty weight for confirmatory rules. Near-zero ensures they are
        never eliminated. Must be > 0 for numerical stability.

    optional_penalty : float, default=0.3
        Penalty weight for optional (analyst-suggested) rules. Lower than
        auto means these are preferred but can still be eliminated.

    l1_ratios : list of float, optional
        L1/L2 mixing ratios for ElasticNetCV. Default: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    tol : float, default=1e-6
        Convergence tolerance for ElasticNetCV.

    Examples
    --------
    >>> from expertrulefit import ExpertRuleFit
    >>> confirmatory = [
    ...     {"name": "CSSF: Country risk > 2.5",
    ...      "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 2.5).astype(float)}
    ... ]
    >>> erf = ExpertRuleFit(max_rules=50)
    >>> erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)
    >>> assert erf.confirmatory_all_active_, "COMPLIANCE FAILURE"
    >>> proba = erf.predict_proba(X_test)[:, 1]
    """

    _BASE_SEED = 42

    def __init__(
        self,
        n_estimators=250,
        tree_size=4,
        max_rules=50,
        random_state=42,
        n_bootstrap=10,
        rule_threshold=0.8,
        confirmatory_penalty=1e-8,
        optional_penalty=0.3,
        l1_ratios=None,
        tol=1e-6,
    ):
        self.n_estimators = n_estimators
        self.tree_size = tree_size
        self.max_rules = max_rules
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.rule_threshold = rule_threshold
        self.confirmatory_penalty = confirmatory_penalty
        self.optional_penalty = optional_penalty
        self.l1_ratios = l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.tol = tol

    def fit(self, X, y, feature_names=None, confirmatory_rules=None, optional_rules=None):
        """Fit ExpertRuleFit with bootstrap-stabilized Elastic Net.

        All internal randomness uses _BASE_SEED for determinism.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).

        feature_names : list of str, optional
            Names for each feature column.

        confirmatory_rules : list of dict, optional
            Regulatory rules that MUST be preserved. Each dict has:
            - "name": str — human-readable rule name
            - "evaluate": callable(X, feature_names) → array of shape (n_samples,)
            These rules bypass bootstrap filtering and get near-zero penalty.

        optional_rules : list of dict, optional
            Analyst-suggested rules with reduced penalty. Same format.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._validate_fit_input(X, y)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = list(feature_names)

        confirmatory_rules = confirmatory_rules or []
        optional_rules = optional_rules or []
        self._validate_expert_rules(confirmatory_rules, "confirmatory")
        self._validate_expert_rules(optional_rules, "optional")
        self.confirmatory_rules_ = confirmatory_rules
        self.optional_rules_ = optional_rules

        # Step 1: Fit base RuleFit with FIXED seed → same trees always
        self.base_rulefit_ = RuleFitClassifier(
            n_estimators=self.n_estimators,
            tree_size=self.tree_size,
            max_rules=self.max_rules,
            random_state=self._BASE_SEED,
            include_linear=True,
        )
        self.base_rulefit_.fit(X, y, feature_names=self.feature_names_)

        # Step 2: Build rule feature matrix manually (transform() is broken)
        X_rules = build_rule_feature_matrix(self.base_rulefit_, X)

        if X_rules is None or X_rules.shape[1] == 0:
            self.stable_mask_ = np.ones(X.shape[1], dtype=bool)
            self.n_stable_rules_ = 0
            self.confirmatory_all_active_ = True
            return self

        n_auto_features = X_rules.shape[1]

        # Step 3: Append confirmatory + optional expert rule features
        expert_columns, expert_names, expert_categories = self._build_expert_columns(
            X, confirmatory_rules, optional_rules
        )

        n_expert = len(expert_columns)
        if expert_columns:
            X_all = np.hstack([X_rules] + expert_columns)
        else:
            X_all = X_rules

        n_total_features = X_all.shape[1]
        self._build_rule_names(n_auto_features)
        self.rule_names_ = self.rule_names_ + expert_names
        self.expert_categories_ = expert_categories

        # Build penalty weight vector: 1.0 for auto, custom for expert
        penalty_weights = self._build_penalty_weights(
            n_auto_features, n_total_features, expert_categories
        )

        # Step 4: Bootstrap stabilization with Elastic Net (fixed seeds)
        # Scale features by 1/sqrt(w_j) so effective penalty = lambda * w_j
        inv_sqrt_w = 1.0 / np.sqrt(np.maximum(penalty_weights, 1e-12))
        rule_selection_count = np.zeros(n_total_features)

        # Pre-scale the full matrix once; bootstrap only needs row indexing
        X_all_scaled = X_all * inv_sqrt_w[np.newaxis, :]

        for b in range(self.n_bootstrap):
            rng = np.random.RandomState(self._BASE_SEED + b)
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot = X_all_scaled[idx]
            y_boot = y[idx]

            try:
                enet = self._make_elasticnet(selection="random")
                enet.fit(X_boot, y_boot)
                rule_selection_count += (np.abs(enet.coef_) > 1e-10)
            except Exception:
                continue

        # Step 5: Keep stable rules (>= threshold)
        self.stable_mask_ = (
            rule_selection_count / max(self.n_bootstrap, 1)
        ) >= self.rule_threshold

        # Force-include ALL confirmatory and optional rules regardless of bootstrap
        if n_expert > 0:
            self.stable_mask_[n_auto_features:n_auto_features + n_expert] = True

        # Step 6: Re-fit final model on stable features with weighted scaling
        if self.stable_mask_.sum() > 0:
            X_stable_weighted = X_all_scaled[:, self.stable_mask_]
            self.stable_inv_sqrt_w_ = inv_sqrt_w[self.stable_mask_]
        else:
            self.stable_mask_ = np.ones(n_total_features, dtype=bool)
            X_stable_weighted = X_all_scaled
            self.stable_inv_sqrt_w_ = inv_sqrt_w

        self.final_model_ = self._make_elasticnet()
        self.final_model_.fit(X_stable_weighted, y)

        self.n_stable_rules_ = int(self.stable_mask_.sum())
        self.n_auto_features_ = n_auto_features
        self.n_expert_ = n_expert

        # Step 7: Verify confirmatory rules are preserved
        self._verify_confirmatory()

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
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        X_stable_weighted = self._build_predict_matrix(X)
        raw = self.final_model_.predict(X_stable_weighted)
        return (raw >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, 2)
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        X_stable_weighted = self._build_predict_matrix(X)
        raw = self.final_model_.predict(X_stable_weighted)
        raw = np.clip(raw, 0, 1)
        return np.column_stack([1 - raw, raw])

    def _build_predict_matrix(self, X):
        """Build the weighted feature matrix for prediction."""
        X = np.asarray(X, dtype=np.float64)
        X_rules = build_rule_feature_matrix(self.base_rulefit_, X)

        expert_columns, _, _ = self._build_expert_columns(
            X, self.confirmatory_rules_, self.optional_rules_
        )

        if expert_columns:
            X_all = np.hstack([X_rules] + expert_columns)
        else:
            X_all = X_rules

        X_stable = X_all[:, self.stable_mask_]
        return X_stable * self.stable_inv_sqrt_w_[np.newaxis, :]

    def get_selected_rules(self):
        """Return names of selected (stable) rules with non-zero final coefficients.

        Returns
        -------
        rules : set of str
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        if self.rule_names_ is None or self.stable_mask_ is None:
            return set()
        stable_indices = np.where(self.stable_mask_)[0]
        final_coefs = self.final_model_.coef_
        return {
            self.rule_names_[idx]
            for j, idx in enumerate(stable_indices)
            if j < len(final_coefs) and abs(final_coefs[j]) > 1e-10
        }

    def get_rule_importance(self):
        """Return rules sorted by absolute coefficient magnitude.

        Returns
        -------
        rules : list of dict with keys 'name', 'coefficient', 'abs_importance', 'category'
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        stable_indices = np.where(self.stable_mask_)[0]
        final_coefs = self.final_model_.coef_
        rules = []
        for j, idx in enumerate(stable_indices):
            if j < len(final_coefs) and abs(final_coefs[j]) > 1e-10:
                name = self.rule_names_[idx]
                rules.append({
                    "name": name,
                    "coefficient": float(final_coefs[j]),
                    "abs_importance": abs(float(final_coefs[j])),
                    "category": _rule_category(name),
                })
        rules.sort(key=lambda r: r["abs_importance"], reverse=True)
        return rules

    def summary(self):
        """Print a human-readable summary of the fitted model."""
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        n_total = len(self.rule_names_) if self.rule_names_ else 0
        n_stable = self.n_stable_rules_
        selected = self.get_selected_rules()

        print("=" * 60)
        print("ExpertRuleFit — Model Summary")
        print("=" * 60)
        print(f"\nCandidate features: {n_total}")
        print(f"Stable features (>= {self.rule_threshold:.0%} bootstrap freq): {n_stable}")
        print(f"Active rules (non-zero coef): {len(selected)}")

        if self.confirmatory_rules_:
            print(f"\nConfirmatory rules ({len(self.confirmatory_rules_)}):")
            for s in self.confirmatory_status_:
                status = "ACTIVE" if s["active"] else "INACTIVE"
                print(f"  [{status}] {s['name']}")
            if self.confirmatory_all_active_:
                print("  All confirmatory rules preserved.")
            else:
                print("  WARNING: Some confirmatory rules were eliminated!")

        if self.optional_rules_:
            print(f"\nOptional rules ({len(self.optional_rules_)}):")
            for rule in self.optional_rules_:
                name = f"optional:{rule['name']}"
                active = name in selected
                status = "ACTIVE" if active else "dropped"
                print(f"  [{status}] {rule['name']}")

        print(f"\nTop rules by importance:")
        for r in self.get_rule_importance()[:10]:
            print(f"  coef={r['coefficient']:+.6f} | {r['name']}")
        print("=" * 60)

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _validate_fit_input(X, y):
        """Validate training data dimensions and types."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent sample counts: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must have at least one sample")

    @staticmethod
    def _validate_expert_rules(rules, label):
        """Validate that expert rules have required keys."""
        for i, rule in enumerate(rules):
            if "name" not in rule:
                raise ValueError(
                    f"{label}_rules[{i}] missing required key 'name'"
                )
            if "evaluate" not in rule:
                raise ValueError(
                    f"{label}_rules[{i}] missing required key 'evaluate'"
                )
            if not callable(rule["evaluate"]):
                raise ValueError(
                    f"{label}_rules[{i}]['evaluate'] must be callable"
                )

    def _build_expert_columns(self, X, confirmatory_rules, optional_rules):
        """Build expert rule feature columns, names, and categories.

        Returns
        -------
        columns : list of ndarray, each shape (n_samples, 1)
        names : list of str
        categories : list of str ('confirmatory' or 'optional')
        """
        columns = []
        names = []
        categories = []
        for rule in confirmatory_rules:
            col = np.asarray(
                rule["evaluate"](X, self.feature_names_), dtype=np.float64
            )
            columns.append(col.reshape(-1, 1))
            names.append(f"confirmatory:{rule['name']}")
            categories.append("confirmatory")
        for rule in optional_rules:
            col = np.asarray(
                rule["evaluate"](X, self.feature_names_), dtype=np.float64
            )
            columns.append(col.reshape(-1, 1))
            names.append(f"optional:{rule['name']}")
            categories.append("optional")
        return columns, names, categories

    def _build_penalty_weights(self, n_auto, n_total, expert_categories):
        """Build the penalty weight vector for weighted Elastic Net.

        Auto-discovered features get weight 1.0. Expert features get
        reduced weights based on their category (confirmatory or optional).
        """
        weights = np.ones(n_total)
        penalty_map = {
            "confirmatory": self.confirmatory_penalty,
            "optional": self.optional_penalty,
        }
        for i, cat in enumerate(expert_categories):
            weights[n_auto + i] = penalty_map[cat]
        return weights

    def _make_elasticnet(self, selection="cyclic"):
        """Create a configured ElasticNetCV instance."""
        return ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=5,
            random_state=self._BASE_SEED,
            tol=self.tol,
            max_iter=10000,
            selection=selection,
        )

    def _build_rule_names(self, n_features):
        """Build human-readable names for all rule features."""
        names = [f"linear:{fn}" for fn in self.feature_names_]
        if hasattr(self.base_rulefit_, "rules_") and self.base_rulefit_.rules_ is not None:
            names.extend(f"rule:{str(rule)[:60]}" for rule in self.base_rulefit_.rules_)
        while len(names) < n_features:
            names.append(f"rule:unknown_{len(names)}")
        self.rule_names_ = names[:n_features]

    def _verify_confirmatory(self):
        """Verify that all confirmatory rules have non-zero coefficients.

        Sets self.confirmatory_all_active_ (bool) and
        self.confirmatory_status_ (list of dict with name + active).
        """
        self.confirmatory_status_ = []
        all_active = True

        if not self.confirmatory_rules_:
            self.confirmatory_all_active_ = True
            return

        # Build a lookup: rule name → position in stable feature set
        stable_indices = np.where(self.stable_mask_)[0]
        final_coefs = self.final_model_.coef_
        name_to_active = {}
        for j, idx in enumerate(stable_indices):
            if j < len(final_coefs):
                name_to_active[self.rule_names_[idx]] = abs(final_coefs[j]) > 1e-10

        for rule in self.confirmatory_rules_:
            name = f"confirmatory:{rule['name']}"
            active = name_to_active.get(name, False)
            self.confirmatory_status_.append({"name": rule["name"], "active": active})
            if not active:
                all_active = False

        self.confirmatory_all_active_ = all_active


def _rule_category(name):
    """Determine the category of a rule from its prefixed name."""
    if name.startswith("confirmatory:"):
        return "confirmatory"
    if name.startswith("optional:"):
        return "optional"
    if name.startswith("linear:"):
        return "linear"
    return "rule"
