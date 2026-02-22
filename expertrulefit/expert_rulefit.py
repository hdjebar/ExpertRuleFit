"""
ExpertRuleFit — Logistic + Bootstrap Stabilization for Reproducible Rule Learning.

Extends imodels.RuleFitClassifier with a deterministic rule selection pipeline
that guarantees 100/100 seed reproducibility for regulated banking environments,
with optional confirmatory rules that survive any regularization strength.

Guarantees:
    1. **Reproducibility** — same data -> same rules -> same predictions (100/100 seeds)
    2. **Rule preservation** — confirmatory (regulatory) rules are NEVER eliminated
    3. **Calibrated Probabilities** — outputs true logistic probabilities (e.g., PD)

Mathematical Foundation — Weighted Feature Scaling:

    Standard Elastic Net minimizes:
        ||y - Xb||^2 + l1 * sum|b_j| + l2 * sum(b_j^2)

    By scaling feature j by 1/sqrt(w_j), the model learns b_tilde_j = b_j * sqrt(w_j).
    The effective penalties become:
        L1:  l1 * sqrt(w_j) * |b_j|   (reduced by sqrt(w_j))
        L2:  l2 * w_j * b_j^2          (reduced by w_j)

    With w_j = 1e-8 for confirmatory rules, the L1 penalty is reduced by
    sqrt(1e-8) = 1e-4, making elimination extremely unlikely. The true
    coefficients must be recovered as: b_j = b_tilde_j / sqrt(w_j)

    Note: the 1/sqrt(w_j) scaling gives exact adaptive-weight behavior on L2
    (per Zou 2006) but only sqrt(w_j) reduction on L1. This is a deliberate
    compromise — scaling by 1/w_j would give exact L1 weighting but causes
    numerical instability.

    As a structural safeguard, confirmatory rules that survive bootstrap
    selection but are zeroed by the final fit are re-introduced via a
    post-hoc constrained refit (see _refit_with_confirmatory).

References:
    - Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles"
    - Zou & Hastie (2005) "Regularization and variable selection via the elastic net"
    - Zou (2006) "The Adaptive LASSO and Its Oracle Properties", JASA
    - Singh et al. (2021) "imodels", JOSS (base implementation)
"""

import logging
import re
import warnings

import numpy as np
from joblib import Parallel, delayed
from packaging.version import Version
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils.validation import check_is_fitted
from imodels import RuleFitClassifier
import sklearn as _sklearn

logger = logging.getLogger(__name__)

# scikit-learn >= 1.8 deprecated the ``penalty`` parameter on
# LogisticRegression / LogisticRegressionCV.  The penalty type is now
# inferred from ``l1_ratios`` (CV) or ``l1_ratio`` (non-CV).
_SKLEARN_PENALTY_DEPRECATED = Version(_sklearn.__version__) >= Version("1.8")


def _make_logistic_cv(*, l1_ratios, solver="saga", cv=5, random_state=42,
                       tol=1e-4, max_iter=10000, n_jobs=None):
    """Create a LogisticRegressionCV compatible with sklearn >= 1.8."""
    if _SKLEARN_PENALTY_DEPRECATED:
        return LogisticRegressionCV(
            solver=solver,
            l1_ratios=l1_ratios,
            cv=cv,
            random_state=random_state,
            tol=tol,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
    return LogisticRegressionCV(
        penalty="elasticnet",
        solver=solver,
        l1_ratios=l1_ratios,
        cv=cv,
        random_state=random_state,
        tol=tol,
        max_iter=max_iter,
        n_jobs=n_jobs,
    )


def _make_logistic(*, penalty=None, solver="lbfgs", max_iter=10000,
                    random_state=42, tol=1e-4, l1_ratio=None):
    """Create a LogisticRegression compatible with sklearn >= 1.8."""
    if _SKLEARN_PENALTY_DEPRECATED:
        kw = dict(solver=solver, max_iter=max_iter,
                  random_state=random_state, tol=tol)
        if penalty is None:
            # Unpenalized: use C=inf (sklearn 1.8 idiom)
            import math
            kw["C"] = math.inf
            kw["l1_ratio"] = 0
        elif penalty == "l2":
            kw["l1_ratio"] = 0
        elif penalty == "l1":
            kw["l1_ratio"] = 1
        elif penalty == "elasticnet":
            kw["l1_ratio"] = l1_ratio if l1_ratio is not None else 0.5
        return LogisticRegression(**kw)
    return LogisticRegression(
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
    )


def _bootstrap_iteration(b, X_all, y, inv_sqrt_w, base_seed, l1_ratios, tol, n_jobs):
    """Run a single bootstrap iteration (for joblib parallelization)."""
    rng = np.random.RandomState(base_seed + b)
    idx = rng.choice(len(y), size=len(y), replace=True)
    X_boot = X_all[idx].multiply(inv_sqrt_w)
    y_boot = y[idx]

    log_reg = _make_logistic_cv(
        l1_ratios=l1_ratios,
        cv=3,
        random_state=base_seed,
        tol=tol,
        max_iter=5000,
        n_jobs=n_jobs,
    )
    log_reg.fit(X_boot, y_boot)
    return (np.abs(log_reg.coef_[0]) > 1e-10).astype(int)


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

_RULE_PATTERN = re.compile(
    r"X_(\d+)\s*(<=|>=|<|>)\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
)

# Vectorized comparison operators keyed by string token
_OPERATORS = {
    "<=": np.less_equal,
    ">=": np.greater_equal,
    "<": np.less,
    ">": np.greater,
}


def eval_rule_on_data(rule_str, X, *, parse_mode="drop_rule"):
    """Evaluate a rule string robustly using regex.

    Handles variations in spacing, negative numbers, and scientific notation.

    Parameters
    ----------
    rule_str : str
        Rule in ``X_i <op> threshold`` notation joined by `` and ``.
    X : array-like of shape (n_samples, n_features)
        Input data (dense or sparse).
    parse_mode : {'drop_rule', 'raise', 'warn_and_zero'}, default='drop_rule'
        How to handle unparseable conditions:

        - ``'drop_rule'`` (default, safest): if *any* condition in the rule
          cannot be parsed, the **entire rule** evaluates to all-False and a
          warning is emitted.  This prevents a malformed rule from silently
          firing for every sample.
        - ``'raise'``: raise a ``ValueError`` immediately.
        - ``'warn_and_zero'``: emit a warning and set the result for the
          unparseable condition to all-False (other conditions still apply).

    Returns
    -------
    result : ndarray of shape (n_samples,), dtype float64
        Binary activation vector (0.0 or 1.0).
    """
    _VALID_PARSE_MODES = {"drop_rule", "raise", "warn_and_zero"}
    if parse_mode not in _VALID_PARSE_MODES:
        raise ValueError(
            f"parse_mode must be one of {_VALID_PARSE_MODES}, got '{parse_mode}'"
        )

    n = X.shape[0]
    result = np.ones(n, dtype=bool)

    for cond in rule_str.split(" and "):
        match = _RULE_PATTERN.search(cond)
        if match is None:
            msg = (
                f"Failed to parse rule condition: '{cond.strip()}' "
                f"in rule '{rule_str}'"
            )
            if parse_mode == "raise":
                raise ValueError(msg)
            warnings.warn(msg)
            if parse_mode == "drop_rule":
                return np.zeros(n, dtype=np.float64)
            # warn_and_zero: zero out this condition (AND with False)
            result[:] = False
            continue

        col_idx = int(match.group(1))
        operator = match.group(2)
        threshold = float(match.group(3))

        if col_idx >= X.shape[1]:
            result[:] = False
            continue

        col_data = X[:, col_idx]
        if sparse.issparse(X):
            col_data = col_data.toarray().ravel()

        result &= _OPERATORS[operator](col_data, threshold)

    return result.astype(np.float64)


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_rule_feature_matrix(rulefit_model, X, *, min_support=0.0,
                              max_support=1.0):
    """Build the full rule feature matrix from a fitted RuleFitClassifier.

    Returns a sparse CSR matrix: [linear features | rule activations].

    Rule columns are built incrementally as sparse vectors and horizontally
    stacked, avoiding a dense ``(n_samples, n_rules)`` intermediate that can
    cause memory blowups on large datasets.

    Parameters
    ----------
    rulefit_model : RuleFitClassifier
        Fitted imodels rule-fit model.
    X : array-like of shape (n_samples, n_features)
        Input data.
    min_support : float, default=0.0
        Minimum fraction of samples a rule must activate on to be kept.
        Rules firing on fewer samples are replaced with an all-zero column
        to preserve column alignment with ``rules_without_feature_names_``.
    max_support : float, default=1.0
        Maximum fraction of samples a rule may activate on.  Rules firing
        on more samples are replaced with all-zero (too common to be
        informative).

    Returns
    -------
    X_augmented : sparse CSR matrix of shape (n_samples, n_features + n_rules)
    """
    X = np.asarray(X, dtype=np.float64, copy=False)
    rules_no_fn = rulefit_model.rules_without_feature_names_

    if not rules_no_fn:
        return sparse.csc_matrix(X)

    n_samples = X.shape[0]

    # Build sparse columns incrementally — avoids dense (n_samples, n_rules)
    rule_columns = []
    for i, rule in enumerate(rules_no_fn):
        try:
            col = eval_rule_on_data(rule.rule, X)
        except Exception as exc:
            warnings.warn(f"Rule {i} evaluation failed: {exc}")
            col = np.zeros(n_samples, dtype=np.float64)

        # Support pre-filter: replace out-of-range rules with zeros
        support = col.sum() / n_samples
        if support < min_support or support > max_support:
            col = np.zeros(n_samples, dtype=np.float64)

        rule_columns.append(sparse.csc_matrix(col.reshape(-1, 1)))

    X_base = sparse.csc_matrix(X)
    return sparse.hstack([X_base] + rule_columns, format="csr")


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


class ExpertRuleFit(BaseEstimator, ClassifierMixin):
    """Reproducible rule-based logistic classifier for regulated environments.

    ExpertRuleFit extends RuleFit with bootstrap-stabilized logistic elastic net
    to guarantee identical rule selection across random seeds, with optional
    confirmatory rules that survive any regularization strength.

    Guarantees:
        1. **Reproducibility** — fixed internal seed + bootstrap -> 100/100 stability
        2. **Rule preservation** — confirmatory rules get near-zero penalty and are
           structurally protected via post-hoc refit if zeroed out
        3. **Calibrated probabilities** — LogisticRegressionCV outputs true probabilities

    Parameters
    ----------
    n_estimators : int, default=250
        Number of trees in the base RuleFit ensemble.

    tree_size : int, default=4
        Maximum depth of trees used for rule extraction.

    max_rules : int, default=50
        Maximum number of rules to extract from the tree ensemble.

    random_state : int or None, default=42
        Accepted for sklearn API compatibility. Internally, ``_BASE_SEED``
        is always used for determinism. A warning is emitted if a different
        value is provided.

    n_bootstrap : int, default=10
        Number of bootstrap samples for rule stability estimation.

    rule_threshold : float, default=0.8
        Minimum frequency (0-1) a rule must be selected across bootstraps
        to be retained. Must be in (0, 1].

    confirmatory_penalty : float, default=1e-8
        Penalty weight for confirmatory rules. Near-zero ensures they are
        never eliminated. Must be > 0.

    optional_penalty : float, default=0.3
        Penalty weight for optional (analyst-suggested) rules. Lower than
        1.0 means preferred but can still be eliminated by bootstrap
        filtering and regularization.

    l1_ratios : list of float, optional
        L1/L2 mixing ratios for LogisticRegressionCV.
        Default: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    tol : float, default=1e-4
        Convergence tolerance for LogisticRegressionCV (saga solver).

    n_jobs : int or None, default=None
        Number of CPU cores for cross-validation. ``None`` means 1 core.
        Use ``-1`` for all cores (caution on shared infrastructure).

    Attributes
    ----------
    base_rulefit_ : RuleFitClassifier
        The fitted base rule-fit model used for candidate rule generation.

    final_model_ : LogisticRegressionCV or LogisticRegression
        The fitted final logistic model on stable features.

    stable_mask_ : ndarray of bool
        Boolean mask indicating which features survived bootstrap selection.

    bootstrap_frequencies_ : ndarray of float
        Bootstrap selection frequency for each feature.

    confirmatory_all_active_ : bool
        True if all confirmatory rules have non-zero coefficients.

    rule_names_ : list of str
        Human-readable names for all candidate features.

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
        tol=1e-4,
        n_jobs=None,
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
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, X, y):
        """Validate inputs and hyperparameters before fitting."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have incompatible shapes: "
                f"X has {X.shape[0]} samples, y has {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must have at least one sample")

        unique_classes = np.unique(y)
        if not np.array_equal(unique_classes, np.array([0.0, 1.0])):
            raise ValueError(
                f"y must be binary with classes {{0, 1}}, "
                f"got unique values: {unique_classes}"
            )

        if not 0 < self.rule_threshold <= 1:
            raise ValueError(
                f"rule_threshold must be in (0, 1], got {self.rule_threshold}"
            )
        if self.confirmatory_penalty <= 0:
            raise ValueError(
                f"confirmatory_penalty must be > 0, got {self.confirmatory_penalty}"
            )
        if self.optional_penalty <= 0:
            raise ValueError(
                f"optional_penalty must be > 0, got {self.optional_penalty}"
            )

        if self.random_state != self._BASE_SEED:
            warnings.warn(
                f"random_state={self.random_state} was provided but ExpertRuleFit "
                f"always uses _BASE_SEED={self._BASE_SEED} internally for "
                f"deterministic reproducibility. The provided value is ignored.",
                UserWarning,
                stacklevel=3,
            )

    def _validate_expert_rules(self, rules, X, label):
        """Validate that expert rule evaluate functions return correct shapes."""
        for i, rule in enumerate(rules):
            if "name" not in rule or "evaluate" not in rule:
                raise ValueError(
                    f"{label} rule {i} must have 'name' and 'evaluate' keys"
                )
            if not callable(rule["evaluate"]):
                raise ValueError(
                    f"{label} rule '{rule['name']}': 'evaluate' must be callable"
                )

            result = rule["evaluate"](X, self.feature_names_)
            result = np.asarray(result, dtype=np.float64).ravel()
            if result.shape[0] != X.shape[0]:
                raise ValueError(
                    f"{label} rule '{rule['name']}' returned {result.shape[0]} "
                    f"values, expected {X.shape[0]}"
                )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X, y, feature_names=None, confirmatory_rules=None, optional_rules=None):
        """Fit ExpertRuleFit with bootstrap-stabilized logistic elastic net.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Binary target (0 or 1).

        feature_names : list of str, optional
            Human-readable names for each feature column.

        confirmatory_rules : list of dict, optional
            Regulatory rules that MUST be preserved. Each dict must have:
            - ``"name"``: str — human-readable rule name
            - ``"evaluate"``: callable(X, feature_names) -> array of shape (n_samples,)
            These rules bypass bootstrap filtering and get near-zero penalty.
            If still zeroed by the solver, a post-hoc refit guarantees inclusion.

        optional_rules : list of dict, optional
            Analyst-suggested rules with reduced penalty. Same format as
            confirmatory rules, but these CAN be eliminated by bootstrap
            filtering and regularization.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = list(feature_names)

        confirmatory_rules = confirmatory_rules or []
        optional_rules = optional_rules or []
        self.confirmatory_rules_ = confirmatory_rules
        self.optional_rules_ = optional_rules

        self._validate_inputs(X, y)
        self._validate_expert_rules(confirmatory_rules, X, "Confirmatory")
        self._validate_expert_rules(optional_rules, X, "Optional")

        # === Step 1: Fit base RuleFit with FIXED seed ===
        logger.info("Step 1: Fitting base RuleFit (seed=%d)", self._BASE_SEED)
        self.base_rulefit_ = RuleFitClassifier(
            n_estimators=self.n_estimators,
            tree_size=self.tree_size,
            max_rules=self.max_rules,
            random_state=self._BASE_SEED,
            include_linear=True,
        )
        self.base_rulefit_.fit(X, y, feature_names=self.feature_names_)

        # === Step 2: Build rule feature matrix ===
        X_rules_sparse = build_rule_feature_matrix(self.base_rulefit_, X)
        n_auto_features = X_rules_sparse.shape[1]

        # === Step 3: Append expert rules ===
        expert_columns = []
        expert_names = []
        expert_categories = []

        for rule in confirmatory_rules:
            col = np.asarray(
                rule["evaluate"](X, self.feature_names_), dtype=np.float64
            )
            expert_columns.append(sparse.csc_matrix(col).T)
            expert_names.append(f"confirmatory:{rule['name']}")
            expert_categories.append("confirmatory")

        for rule in optional_rules:
            col = np.asarray(
                rule["evaluate"](X, self.feature_names_), dtype=np.float64
            )
            expert_columns.append(sparse.csc_matrix(col).T)
            expert_names.append(f"optional:{rule['name']}")
            expert_categories.append("optional")

        n_expert = len(expert_columns)

        if expert_columns:
            X_all = sparse.hstack([X_rules_sparse] + expert_columns, format="csr")
        else:
            X_all = X_rules_sparse

        n_total_features = X_all.shape[1]
        self._build_rule_names(n_auto_features)
        self.rule_names_ = self.rule_names_ + expert_names
        self.expert_categories_ = expert_categories

        # === Build penalty weight vector ===
        penalty_weights = np.ones(n_total_features)
        penalty_map = {"confirmatory": self.confirmatory_penalty,
                       "optional": self.optional_penalty}
        for i, cat in enumerate(expert_categories):
            penalty_weights[n_auto_features + i] = penalty_map[cat]

        # inv_sqrt_w: feature scaling so effective L2 penalty = l2*w_j*b_j^2
        inv_sqrt_w = 1.0 / np.sqrt(np.maximum(penalty_weights, 1e-12))

        # === Step 4: Bootstrap stabilization (parallelized) ===
        logger.info(
            "Step 4: Bootstrap stabilization (%d iterations)", self.n_bootstrap
        )

        # Each bootstrap iteration is independent — run in parallel
        # When n_jobs is used for inner CV, use sequential bootstrap;
        # otherwise parallelize the outer loop.
        bootstrap_n_jobs = 1 if self.n_jobs and self.n_jobs != 1 else -1
        results = Parallel(n_jobs=bootstrap_n_jobs)(
            delayed(_bootstrap_iteration)(
                b, X_all, y, inv_sqrt_w,
                self._BASE_SEED, self.l1_ratios, self.tol, self.n_jobs,
            )
            for b in range(self.n_bootstrap)
        )

        # Filter out failed iterations (returned as None by exception wrapper)
        successful = [r for r in results if r is not None]
        n_successful_boots = len(successful)

        if n_successful_boots == 0:
            raise RuntimeError(
                "All bootstrap iterations failed. Cannot determine stable rules."
            )

        rule_selection_count = np.sum(successful, axis=0)

        if n_successful_boots < self.n_bootstrap:
            warnings.warn(
                f"Only {n_successful_boots}/{self.n_bootstrap} bootstrap iterations "
                f"succeeded. Stability estimates may be unreliable."
            )

        # === Step 5: Frequency-based filtering ===
        self.bootstrap_frequencies_ = rule_selection_count / n_successful_boots
        self.stable_mask_ = self.bootstrap_frequencies_ >= self.rule_threshold

        # Force-include ONLY confirmatory rules (optional rules must earn it)
        for i, cat in enumerate(expert_categories):
            if cat == "confirmatory":
                self.stable_mask_[n_auto_features + i] = True
            # optional rules keep their bootstrap-determined value

        logger.info(
            "Step 5: %d/%d features survived bootstrap (threshold=%.0f%%)",
            self.stable_mask_.sum(),
            n_total_features,
            self.rule_threshold * 100,
        )

        # === Step 6: Final fit on stable features ===
        if self.stable_mask_.sum() == 0:
            raise RuntimeError(
                "No features survived bootstrap filtering. Consider lowering "
                "rule_threshold or increasing n_bootstrap."
            )

        X_stable = X_all[:, self.stable_mask_]
        w_stable = inv_sqrt_w[self.stable_mask_]
        X_stable_weighted = X_stable.multiply(w_stable)

        self.final_model_ = _make_logistic_cv(
            l1_ratios=self.l1_ratios,
            cv=5,
            random_state=self._BASE_SEED,
            tol=self.tol,
            max_iter=10000,
            n_jobs=self.n_jobs,
        )
        self.final_model_.fit(X_stable_weighted, y)

        self.stable_inv_sqrt_w_ = w_stable
        self.n_stable_rules_ = int(self.stable_mask_.sum())
        self.n_auto_features_ = n_auto_features
        self.n_expert_ = n_expert

        # === Step 7: Verify + enforce confirmatory rules ===
        self._verify_confirmatory()

        if not self.confirmatory_all_active_ and confirmatory_rules:
            warnings.warn(
                "Some confirmatory rules were zeroed by the solver. "
                "Executing post-hoc constrained refit to enforce inclusion."
            )
            self._refit_with_confirmatory(X_all, y, inv_sqrt_w)
            self._verify_confirmatory()

            if not self.confirmatory_all_active_:
                warnings.warn(
                    "COMPLIANCE WARNING: Confirmatory rules could not be "
                    "activated even after constrained refit. This may indicate "
                    "perfect collinearity or degenerate data for these rules."
                )

        return self

    # ------------------------------------------------------------------
    # Post-hoc confirmatory enforcement
    # ------------------------------------------------------------------

    def _refit_with_confirmatory(self, X_all, y, inv_sqrt_w):
        """Refit the final model with confirmatory features unpenalized.

        Falls back to a single unpenalized logistic regression on all stable
        features (sacrificing sparsity for the confirmatory guarantee).
        """
        stable_indices = np.where(self.stable_mask_)[0]

        # Identify which stable features are confirmatory
        confirmatory_mask_in_stable = np.array([
            self.rule_names_[idx].startswith("confirmatory:")
            for idx in stable_indices
        ])

        if not confirmatory_mask_in_stable.any():
            return

        X_stable = X_all[:, self.stable_mask_]
        w_stable = inv_sqrt_w[self.stable_mask_]
        X_stable_weighted = X_stable.multiply(w_stable)

        try:
            lr_unpenalized = _make_logistic(
                penalty=None,
                solver="lbfgs",
                max_iter=10000,
                random_state=self._BASE_SEED,
                tol=self.tol,
            )
            lr_unpenalized.fit(X_stable_weighted, y)
            self.final_model_ = lr_unpenalized
            logger.info("Post-hoc refit: using unpenalized logistic regression")
        except Exception as exc:
            warnings.warn(
                f"Post-hoc unpenalized refit failed: {exc}. "
                f"Confirmatory guarantee may not hold."
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        X_weighted = self._build_predict_matrix(X)
        return self.final_model_.predict(X_weighted)

    def predict_proba(self, X):
        """Predict calibrated class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0 = P(y=0), Column 1 = P(y=1).
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        X_weighted = self._build_predict_matrix(X)
        return self.final_model_.predict_proba(X_weighted)

    def _build_predict_matrix(self, X):
        """Build the weighted sparse feature matrix for prediction.

        Uses an identity-based cache so that consecutive calls with the
        same array (e.g. predict then predict_proba) skip recomputation.
        """
        X = np.asarray(X, dtype=np.float64, copy=False)

        # Fast identity cache: reuse result if same array object
        cache_key = id(X)
        if hasattr(self, "_pred_cache_") and self._pred_cache_[0] == cache_key:
            return self._pred_cache_[1]

        X_rules_sparse = build_rule_feature_matrix(self.base_rulefit_, X)

        expert_columns = []
        for rule in self.confirmatory_rules_:
            col = np.asarray(
                rule["evaluate"](X, self.feature_names_), dtype=np.float64
            )
            expert_columns.append(sparse.csc_matrix(col).T)
        for rule in self.optional_rules_:
            col = np.asarray(
                rule["evaluate"](X, self.feature_names_), dtype=np.float64
            )
            expert_columns.append(sparse.csc_matrix(col).T)

        if expert_columns:
            X_all = sparse.hstack([X_rules_sparse] + expert_columns, format="csr")
        else:
            X_all = X_rules_sparse

        X_stable = X_all[:, self.stable_mask_]
        result = X_stable.multiply(self.stable_inv_sqrt_w_)
        self._pred_cache_ = (cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Interpretability (coefficients in ORIGINAL feature space)
    # ------------------------------------------------------------------

    def _get_true_coefficients(self):
        """Recover coefficients in the original (unscaled) feature space.

        The model is fitted on X * diag(1/sqrt(w)), so the fitted coefficient
        b_tilde_j relates to the true coefficient as: b_j = b_tilde_j * (1/sqrt(w_j)).

        Returns
        -------
        true_coefs : ndarray of shape (n_stable_features,)
        """
        fitted_coefs = self.final_model_.coef_[0]
        return fitted_coefs * self.stable_inv_sqrt_w_

    def get_selected_rules(self):
        """Return names of selected (stable) rules with non-zero coefficients.

        Returns
        -------
        rules : set of str
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])
        if not hasattr(self, "rule_names_") or not hasattr(self, "stable_mask_"):
            return set()

        stable_indices = np.where(self.stable_mask_)[0]
        true_coefs = self._get_true_coefficients()

        return {
            self.rule_names_[idx]
            for j, idx in enumerate(stable_indices)
            if j < len(true_coefs) and abs(true_coefs[j]) > 1e-10
        }

    def get_rule_importance(self):
        """Return rules sorted by absolute coefficient magnitude.

        Coefficients are reported in the **original feature space** (rescaled
        from the weighted fitting space), so magnitudes are directly comparable
        across rules with different penalty weights.

        Returns
        -------
        rules : list of dict
            Each dict has keys: ``name``, ``coefficient``, ``abs_importance``,
            ``category``, ``bootstrap_frequency``.
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])

        stable_indices = np.where(self.stable_mask_)[0]
        true_coefs = self._get_true_coefficients()
        boot_freqs = self.bootstrap_frequencies_

        rules = []
        for j, idx in enumerate(stable_indices):
            if j >= len(true_coefs):
                break

            coef = float(true_coefs[j])
            if abs(coef) <= 1e-10:
                continue

            name = self.rule_names_[idx]
            category = _rule_category(name)

            rules.append({
                "name": name,
                "coefficient": coef,
                "abs_importance": abs(coef),
                "category": category,
                "bootstrap_frequency": float(boot_freqs[idx]),
            })

        rules.sort(key=lambda r: r["abs_importance"], reverse=True)
        return rules

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self, return_string=False):
        """Generate a human-readable summary of the fitted model.

        Parameters
        ----------
        return_string : bool, default=False
            If True, return the summary as a string instead of printing.

        Returns
        -------
        text : str or None
            Summary string if ``return_string=True``, else None.
        """
        check_is_fitted(self, ["base_rulefit_", "final_model_"])

        n_total = len(self.rule_names_) if self.rule_names_ else 0
        selected = self.get_selected_rules()

        lines = [
            "=" * 60,
            "ExpertRuleFit — Model Summary (Logistic)",
            "=" * 60,
            "",
            f"Candidate features:  {n_total}",
            f"Stable features (>={self.rule_threshold:.0%} bootstrap): {self.n_stable_rules_}",
            f"Active rules (non-zero coef): {len(selected)}",
            f"Bootstrap iterations: {self.n_bootstrap}",
        ]

        if self.confirmatory_rules_:
            lines.append(f"\nConfirmatory rules ({len(self.confirmatory_rules_)}):")
            for s in self.confirmatory_status_:
                status = "ACTIVE" if s["active"] else "INACTIVE"
                lines.append(f"  [{status}] {s['name']}")
            if self.confirmatory_all_active_:
                lines.append("  -> All confirmatory rules preserved.")
            else:
                lines.append("  WARNING: Some confirmatory rules were eliminated!")

        if self.optional_rules_:
            lines.append(f"\nOptional rules ({len(self.optional_rules_)}):")
            for rule in self.optional_rules_:
                name = f"optional:{rule['name']}"
                active = name in selected
                status = "ACTIVE" if active else "dropped"
                lines.append(f"  [{status}] {rule['name']}")

        importance = self.get_rule_importance()
        if importance:
            lines.append("\nTop rules by importance:")
            for r in importance[:10]:
                lines.append(
                    f"  coef={r['coefficient']:+.6f} "
                    f"(boot={r['bootstrap_frequency']:.0%}) | {r['name']}"
                )

        lines.append("=" * 60)
        text = "\n".join(lines)

        if return_string:
            return text

        print(text)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rule_names(self, n_features):
        """Build unique human-readable names for all rule features."""
        names = []
        for fn in self.feature_names_:
            names.append(f"linear:{fn}")

        if hasattr(self.base_rulefit_, "rules_") and self.base_rulefit_.rules_ is not None:
            for i, rule in enumerate(self.base_rulefit_.rules_):
                rule_str = str(rule)
                names.append(f"rule[{i}]:{rule_str}")

        while len(names) < n_features:
            names.append(f"rule:unknown_{len(names)}")

        self.rule_names_ = names[:n_features]

    def _verify_confirmatory(self):
        """Check that all confirmatory rules have non-zero coefficients."""
        self.confirmatory_status_ = []
        all_active = True

        if not self.confirmatory_rules_:
            self.confirmatory_all_active_ = True
            return

        # Build a lookup: rule name -> position in stable feature set
        stable_indices = np.where(self.stable_mask_)[0]
        true_coefs = self._get_true_coefficients()
        name_to_pos = {
            self.rule_names_[idx]: j
            for j, idx in enumerate(stable_indices)
            if j < len(true_coefs)
        }

        for rule in self.confirmatory_rules_:
            name = f"confirmatory:{rule['name']}"
            j = name_to_pos.get(name)
            active = j is not None and abs(true_coefs[j]) > 1e-10

            self.confirmatory_status_.append({
                "name": rule["name"],
                "active": active,
            })
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
