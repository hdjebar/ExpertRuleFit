"""
EBM + ExpertRuleFit — Dual Glass-Box Architecture for Regulated Banking.

Combines two complementary glass-box models:
    - **EBM** (InterpretML): Discovers smooth shape functions + pairwise interactions
    - **ExpertRuleFit**: Stabilizes discovered rules via bootstrap + preserves regulatory rules

Architecture (Option B — EBM Discovery -> ExpertRuleFit Stabilization):
    1. Fit EBM on training data -> learn shape functions f(x_j) and interactions f(x_i, x_j)
    2. Extract threshold-based rules from EBM shape functions (inflection points, sign changes)
    3. Extract interaction rules from EBM pairwise terms (high-importance regions)
    4. Inject EBM-discovered rules as optional_rules into ExpertRuleFit
    5. Regulatory rules enter as confirmatory_rules (NEVER eliminated)
    6. ExpertRuleFit bootstrap-stabilizes everything -> 100/100 reproducibility

Why this combination:
    - RuleFit alone only generates axis-aligned tree splits (X_3 > 0.4)
    - EBM discovers smooth nonlinearities and pairwise interactions via FAST algorithm
    - ExpertRuleFit provides what EBM lacks: reproducibility guarantee + regulatory rule
      preservation
    - Together: EBM finds the signal, ExpertRuleFit locks it down for production

References:
    - Lou et al. (2012) "Intelligible Models for Classification and Regression", KDD
    - Lou et al. (2013) "Accurate Intelligible Models with Pairwise Interactions", KDD
    - Nori et al. (2019) "InterpretML: A Unified Framework for ML Interpretability"
    - Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles"
    - Zou (2006) "The Adaptive LASSO and Its Oracle Properties", JASA
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for extracted rules
# ---------------------------------------------------------------------------


@dataclass
class ExtractedRule:
    """A rule extracted from an EBM shape function or interaction."""

    name: str
    description: str
    source: str  # 'ebm_shape', 'ebm_interaction', 'confirmatory', 'optional'
    feature_indices: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    ebm_importance: float
    evaluate: Callable  # callable(X, feature_names) -> array(n_samples,)

    def to_expertrulefit_dict(self):
        """Convert to ExpertRuleFit rule dict format."""
        return {
            "name": self.name,
            "evaluate": self.evaluate,
            "_description": self.description,
            "_source": self.source,
            "_ebm_importance": self.ebm_importance,
        }


@dataclass
class EBMExtractionReport:
    """Summary of rules extracted from an EBM."""

    n_shape_rules: int = 0
    n_interaction_rules: int = 0
    shape_rules: List[ExtractedRule] = field(default_factory=list)
    interaction_rules: List[ExtractedRule] = field(default_factory=list)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    top_interactions: List[Tuple[str, str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EBM Rule Extractor
# ---------------------------------------------------------------------------


class EBMRuleExtractor:
    """Extract threshold-based rules from a fitted EBM.

    Analyzes EBM shape functions to find meaningful thresholds:
    - **Sign changes**: where a feature's contribution crosses zero
    - **Large jumps**: where the score changes sharply (step-like behavior)

    For interactions, extracts rules for high-importance 2D regions.

    Parameters
    ----------
    sign_change_min_effect : float, default=0.1
        Minimum absolute score difference around a sign change to extract a rule.

    jump_percentile : float, default=90
        Percentile threshold for identifying "large" score jumps between bins.

    extreme_percentile : float, default=90
        Percentile for identifying extreme score regions.

    min_importance_rank : float, default=0.3
        Only extract rules from terms with importance >= this fraction of the max.

    interaction_top_k : int, default=5
        Maximum number of interaction rules to extract.
    """

    def __init__(
        self,
        sign_change_min_effect=0.1,
        jump_percentile=90,
        extreme_percentile=90,
        min_importance_rank=0.3,
        interaction_top_k=5,
    ):
        self.sign_change_min_effect = sign_change_min_effect
        self.jump_percentile = jump_percentile
        self.extreme_percentile = extreme_percentile
        self.min_importance_rank = min_importance_rank
        self.interaction_top_k = interaction_top_k

    def extract(self, ebm, feature_names=None):
        """Extract rules from a fitted EBM.

        Parameters
        ----------
        ebm : ExplainableBoostingClassifier or ExplainableBoostingRegressor
            A fitted EBM model.

        feature_names : list of str, optional
            Feature names. If None, uses ebm.feature_names_in_.

        Returns
        -------
        report : EBMExtractionReport
        """
        if feature_names is None:
            feature_names = list(ebm.feature_names_in_)

        importances = ebm.term_importances()
        max_importance = max(importances) if len(importances) > 0 else 1.0
        importance_threshold = self.min_importance_rank * max_importance

        report = EBMExtractionReport()
        report.feature_importances = {
            ebm.term_names_[i]: float(importances[i])
            for i in range(len(importances))
        }

        for term_idx in range(len(ebm.term_features_)):
            term_features = ebm.term_features_[term_idx]
            term_importance = importances[term_idx]

            if term_importance < importance_threshold:
                continue

            if len(term_features) == 1:
                rules = self._extract_shape_rules(
                    ebm, term_idx, term_features[0], feature_names, term_importance
                )
                report.shape_rules.extend(rules)

            elif len(term_features) == 2:
                rules = self._extract_interaction_rules(
                    ebm, term_idx, term_features, feature_names, term_importance
                )
                report.interaction_rules.extend(rules)

                f1 = feature_names[term_features[0]]
                f2 = feature_names[term_features[1]]
                report.top_interactions.append((f1, f2, float(term_importance)))

        # Sort interactions by importance and limit
        report.interaction_rules.sort(key=lambda r: r.ebm_importance, reverse=True)
        report.interaction_rules = report.interaction_rules[:self.interaction_top_k]

        report.n_shape_rules = len(report.shape_rules)
        report.n_interaction_rules = len(report.interaction_rules)

        logger.info(
            "Extracted %d shape rules + %d interaction rules from EBM",
            report.n_shape_rules,
            report.n_interaction_rules,
        )

        return report

    def _extract_shape_rules(
        self, ebm, term_idx, feat_idx, feature_names, importance
    ):
        """Extract threshold rules from a single EBM shape function.

        Looks for:
        1. Sign changes (score crosses zero)
        2. Large score jumps (step-like thresholds)
        """
        scores = ebm.term_scores_[term_idx]
        feat_name = feature_names[feat_idx]

        # Get bin edges for this feature
        bins_for_feat = ebm.bins_[feat_idx]
        if isinstance(bins_for_feat, list) and len(bins_for_feat) > 0:
            if isinstance(bins_for_feat[0], np.ndarray):
                bin_edges = bins_for_feat[0]
            else:
                bin_edges = np.array(bins_for_feat)
        elif isinstance(bins_for_feat, np.ndarray):
            bin_edges = bins_for_feat
        else:
            return []

        if len(bin_edges) == 0:
            return []

        # scores has len(bin_edges) + 2 entries typically:
        # [missing_bin, bin_0, bin_1, ..., bin_n]
        if len(scores) == len(bin_edges) + 2:
            score_values = scores[1:]  # skip missing bin
        elif len(scores) == len(bin_edges) + 1:
            score_values = scores
        else:
            score_values = scores

        rules = []
        rule_names_seen = set()

        # --- Strategy 1: Sign changes ---
        for i in range(len(score_values) - 1):
            if score_values[i] * score_values[i + 1] < 0:
                effect_size = abs(score_values[i + 1] - score_values[i])
                if effect_size < self.sign_change_min_effect:
                    continue

                if i < len(bin_edges):
                    threshold = float(bin_edges[i])
                    direction = ">" if score_values[i + 1] > score_values[i] else "<="

                    rule_name = f"EBM:{feat_name} {direction} {threshold:.4f}"

                    # Capture loop variables via default args
                    _thresh = threshold
                    _fidx = feat_idx

                    if direction == ">":
                        def _eval(X, fn, t=_thresh, fi=_fidx):
                            return (X[:, fi] > t).astype(np.float64)
                    else:
                        def _eval(X, fn, t=_thresh, fi=_fidx):
                            return (X[:, fi] <= t).astype(np.float64)

                    rule_names_seen.add(rule_name)
                    rules.append(ExtractedRule(
                        name=rule_name,
                        description=(
                            f"EBM shape function for {feat_name} crosses zero "
                            f"at {threshold:.4f} (effect={effect_size:.3f})"
                        ),
                        source="ebm_shape",
                        feature_indices=(feat_idx,),
                        feature_names=(feat_name,),
                        ebm_importance=importance,
                        evaluate=_eval,
                    ))

        # --- Strategy 2: Largest score jumps ---
        if len(score_values) > 2:
            diffs = np.abs(np.diff(score_values))
            jump_threshold = np.percentile(diffs, self.jump_percentile)

            for i in range(len(diffs)):
                if diffs[i] >= jump_threshold and i < len(bin_edges):
                    threshold = float(bin_edges[i])
                    direction = ">" if score_values[i + 1] > score_values[i] else "<="

                    # Avoid duplicating sign-change rules (O(1) set lookup)
                    candidate_name = f"EBM:{feat_name} {direction} {threshold:.4f}"
                    if candidate_name in rule_names_seen:
                        continue
                    rule_names_seen.add(candidate_name)

                    _thresh = threshold
                    _fidx = feat_idx
                    _effect = float(diffs[i])

                    if direction == ">":
                        def _eval(X, fn, t=_thresh, fi=_fidx):
                            return (X[:, fi] > t).astype(np.float64)
                    else:
                        def _eval(X, fn, t=_thresh, fi=_fidx):
                            return (X[:, fi] <= t).astype(np.float64)

                    rules.append(ExtractedRule(
                        name=candidate_name,
                        description=(
                            f"EBM large score jump for {feat_name} "
                            f"at {threshold:.4f} (delta_score={_effect:.3f})"
                        ),
                        source="ebm_shape",
                        feature_indices=(feat_idx,),
                        feature_names=(feat_name,),
                        ebm_importance=importance,
                        evaluate=_eval,
                    ))

        return rules

    def _extract_interaction_rules(
        self, ebm, term_idx, term_features, feature_names, importance
    ):
        """Extract rules from EBM pairwise interaction terms.

        Finds the high-score region of the 2D interaction surface and
        creates a joint threshold rule.
        """
        scores_2d = ebm.term_scores_[term_idx]
        feat_idx_1, feat_idx_2 = term_features
        fname1 = feature_names[feat_idx_1]
        fname2 = feature_names[feat_idx_2]

        bins1 = ebm.bins_[feat_idx_1]
        bins2 = ebm.bins_[feat_idx_2]

        # Resolve multi-level bins
        if isinstance(bins1, list) and len(bins1) > 0:
            edges1 = bins1[0] if isinstance(bins1[0], np.ndarray) else np.array(bins1)
        else:
            edges1 = np.asarray(bins1)

        if isinstance(bins2, list) and len(bins2) > 0:
            edges2 = bins2[0] if isinstance(bins2[0], np.ndarray) else np.array(bins2)
        else:
            edges2 = np.asarray(bins2)

        if len(edges1) == 0 or len(edges2) == 0:
            return []

        rules = []

        try:
            working_scores = scores_2d
            if working_scores.shape[0] == len(edges1) + 2:
                working_scores = working_scores[1:, :]
            if working_scores.shape[1] == len(edges2) + 2:
                working_scores = working_scores[:, 1:]

            # Find the cell with max absolute score
            flat_idx = np.argmax(np.abs(working_scores))
            i, j = np.unravel_index(flat_idx, working_scores.shape)
            peak_score = float(working_scores[i, j])

            if abs(peak_score) < self.sign_change_min_effect:
                return []

            thresh1 = float(edges1[min(i, len(edges1) - 1)])
            thresh2 = float(edges2[min(j, len(edges2) - 1)])
            dir1 = ">" if peak_score > 0 else "<="
            dir2 = ">" if peak_score > 0 else "<="

            rule_name = (
                f"EBM:{fname1} {dir1} {thresh1:.4f} AND "
                f"{fname2} {dir2} {thresh2:.4f}"
            )

            _t1, _t2 = thresh1, thresh2
            _fi1, _fi2 = feat_idx_1, feat_idx_2
            _d1, _d2 = dir1, dir2

            def _eval_interaction(
                X, fn, t1=_t1, t2=_t2, fi1=_fi1, fi2=_fi2, d1=_d1, d2=_d2
            ):
                c1 = X[:, fi1] > t1 if d1 == ">" else X[:, fi1] <= t1
                c2 = X[:, fi2] > t2 if d2 == ">" else X[:, fi2] <= t2
                return (c1 & c2).astype(np.float64)

            rules.append(ExtractedRule(
                name=rule_name,
                description=(
                    f"EBM interaction: {fname1} x {fname2} peak region "
                    f"(score={peak_score:.3f})"
                ),
                source="ebm_interaction",
                feature_indices=(feat_idx_1, feat_idx_2),
                feature_names=(fname1, fname2),
                ebm_importance=importance,
                evaluate=_eval_interaction,
            ))

        except Exception as exc:
            warnings.warn(
                f"Failed to extract interaction rule for "
                f"{fname1} x {fname2}: {exc}"
            )

        return rules


# ---------------------------------------------------------------------------
# Dual Glass-Box Orchestrator
# ---------------------------------------------------------------------------


class DualGlassBox(BaseEstimator, ClassifierMixin):
    """Dual glass-box architecture: EBM discovery + ExpertRuleFit stabilization.

    Fits an EBM to discover shape functions and pairwise interactions, extracts
    threshold-based rules from the learned functions, then feeds them into
    ExpertRuleFit for bootstrap-stabilized selection with regulatory rule
    preservation.

    The result is a model that combines:
    - EBM's ability to discover smooth nonlinearities and interactions
    - ExpertRuleFit's reproducibility guarantee (100/100 seeds)
    - ExpertRuleFit's confirmatory rule preservation (NEVER eliminated)

    Two models are accessible after fitting:
    - ``self.ebm_`` — the fitted EBM (for shape function visualization, PD curves)
    - ``self.erf_`` — the fitted ExpertRuleFit (for stable rule-based predictions)

    Parameters
    ----------
    ebm_params : dict, optional
        Parameters for ExplainableBoostingClassifier.

    erf_params : dict, optional
        Parameters for ExpertRuleFit.

    extractor_params : dict, optional
        Parameters for EBMRuleExtractor.

    inject_ebm_rules_as : str, default='optional'
        How to inject EBM-discovered rules into ExpertRuleFit:
        - 'optional': reduced penalty, can be eliminated by bootstrap
        - 'confirmatory': near-zero penalty, never eliminated

    Examples
    --------
    >>> from expertrulefit import DualGlassBox
    >>> confirmatory = [
    ...     {"name": "CSSF: Country risk > 2.5",
    ...      "evaluate": lambda X, fn: (X[:, fn.index("country_risk")] > 2.5).astype(float)}
    ... ]
    >>> dgb = DualGlassBox()
    >>> dgb.fit(X_train, y_train, feature_names=fn, confirmatory_rules=confirmatory)
    >>> proba = dgb.predict_proba(X_test)[:, 1]
    """

    def __init__(
        self,
        ebm_params=None,
        erf_params=None,
        extractor_params=None,
        inject_ebm_rules_as="optional",
    ):
        self.ebm_params = ebm_params
        self.erf_params = erf_params
        self.extractor_params = extractor_params
        self.inject_ebm_rules_as = inject_ebm_rules_as

    def fit(
        self,
        X,
        y,
        feature_names=None,
        confirmatory_rules=None,
        optional_rules=None,
    ):
        """Fit the dual glass-box pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
        confirmatory_rules : list of dict, optional
            Regulatory rules for ExpertRuleFit (NEVER eliminated).
        optional_rules : list of dict, optional
            Analyst-suggested rules for ExpertRuleFit.

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

        from .expert_rulefit import ExpertRuleFit

        X = np.asarray(X, dtype=np.float64, copy=False)
        y = np.asarray(y, dtype=np.float64).ravel()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = list(feature_names)

        confirmatory_rules = confirmatory_rules or []
        optional_rules = list(optional_rules or [])

        # ============================================================
        # Stage 1: Fit EBM — discover shape functions + interactions
        # ============================================================
        ebm_defaults = {
            "max_bins": 256,
            "interactions": 10,
            "outer_bags": 25,
            "inner_bags": 25,
            "learning_rate": 0.01,
            "random_state": 42,
            "n_jobs": -1,
        }
        ebm_kw = {**ebm_defaults, **(self.ebm_params or {})}

        logger.info("Stage 1: Fitting EBM")
        self.ebm_ = ExplainableBoostingClassifier(**ebm_kw)
        # Pass DataFrame so EBM stores feature names correctly
        df = pd.DataFrame(X, columns=feature_names)
        self.ebm_.fit(df, y)

        # ============================================================
        # Stage 2: Extract rules from EBM
        # ============================================================
        extractor_kw = self.extractor_params or {}
        extractor = EBMRuleExtractor(**extractor_kw)
        self.extraction_report_ = extractor.extract(self.ebm_, feature_names)

        # Convert extracted rules to ExpertRuleFit format
        ebm_rules = []
        for rule in (
            self.extraction_report_.shape_rules
            + self.extraction_report_.interaction_rules
        ):
            ebm_rules.append(rule.to_expertrulefit_dict())

        logger.info(
            "Stage 2: Extracted %d rules from EBM (%d shape + %d interaction)",
            len(ebm_rules),
            self.extraction_report_.n_shape_rules,
            self.extraction_report_.n_interaction_rules,
        )

        # ============================================================
        # Stage 3: Inject EBM rules into ExpertRuleFit
        # ============================================================
        if self.inject_ebm_rules_as == "confirmatory":
            inject_confirmatory = confirmatory_rules + ebm_rules
            inject_optional = optional_rules
        else:
            inject_confirmatory = confirmatory_rules
            inject_optional = optional_rules + ebm_rules

        erf_defaults = {
            "n_estimators": 250,
            "tree_size": 4,
            "max_rules": 50,
            "n_bootstrap": 10,
            "rule_threshold": 0.8,
        }
        erf_kw = {**erf_defaults, **(self.erf_params or {})}

        logger.info(
            "Stage 3: Fitting ExpertRuleFit with %d confirmatory + %d optional rules",
            len(inject_confirmatory),
            len(inject_optional),
        )

        self.erf_ = ExpertRuleFit(**erf_kw)
        self.erf_.fit(
            X,
            y,
            feature_names=feature_names,
            confirmatory_rules=inject_confirmatory,
            optional_rules=inject_optional,
        )

        # ============================================================
        # Store comparison metrics
        # ============================================================
        self._compute_comparison_metrics(X, y)

        return self

    def predict(self, X):
        """Predict using ExpertRuleFit (stable rule-based model).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["erf_"])
        return self.erf_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using ExpertRuleFit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        check_is_fitted(self, ["erf_"])
        return self.erf_.predict_proba(X)

    def predict_proba_ebm(self, X):
        """Predict probabilities using EBM (smooth shape functions).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        check_is_fitted(self, ["ebm_"])
        import pandas as pd

        X = np.asarray(X, dtype=np.float64, copy=False)
        df = pd.DataFrame(X, columns=self.feature_names_)
        return self.ebm_.predict_proba(df)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def explain_rules(self):
        """Get ExpertRuleFit rule importance (stable, interpretable rules).

        Returns
        -------
        rules : list of dict
        """
        check_is_fitted(self, ["erf_"])
        return self.erf_.get_rule_importance()

    def explain_ebm_global(self):
        """Get EBM global explanation object (for InterpretML show()).

        Returns
        -------
        explanation : EBMExplanation
        """
        check_is_fitted(self, ["ebm_"])
        return self.ebm_.explain_global()

    def explain_ebm_local(self, X, y=None):
        """Get EBM local explanation for individual predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like, optional

        Returns
        -------
        explanation : EBMExplanation
        """
        check_is_fitted(self, ["ebm_"])
        import pandas as pd

        X = np.asarray(X, dtype=np.float64, copy=False)
        df = pd.DataFrame(X, columns=self.feature_names_)
        return self.ebm_.explain_local(df, y)

    def get_ebm_rules_in_model(self):
        """Show which EBM-discovered rules survived ExpertRuleFit selection.

        Returns
        -------
        report : list of dict
        """
        check_is_fitted(self, ["erf_", "extraction_report_"])

        selected = self.erf_.get_selected_rules()
        importance_map = {
            r["name"]: r["coefficient"]
            for r in self.erf_.get_rule_importance()
        }

        report = []
        for rule in (
            self.extraction_report_.shape_rules
            + self.extraction_report_.interaction_rules
        ):
            prefix = (
                "confirmatory"
                if self.inject_ebm_rules_as == "confirmatory"
                else "optional"
            )
            full_name = f"{prefix}:{rule.name}"
            survived = full_name in selected
            coef = importance_map.get(full_name, 0.0)

            report.append({
                "name": rule.name,
                "full_name": full_name,
                "source": rule.source,
                "description": rule.description,
                "ebm_importance": rule.ebm_importance,
                "survived_bootstrap": survived,
                "erf_coefficient": coef,
            })

        report.sort(key=lambda r: abs(r["erf_coefficient"]), reverse=True)
        return report

    # ------------------------------------------------------------------
    # Comparison / diagnostics
    # ------------------------------------------------------------------

    def _compute_comparison_metrics(self, X, y):
        """Compute AUC for both models on training data (diagnostic only)."""
        import pandas as pd

        try:
            df = pd.DataFrame(X, columns=self.feature_names_)
            ebm_proba = self.ebm_.predict_proba(df)[:, 1]
            erf_proba = self.erf_.predict_proba(X)[:, 1]
            self.train_auc_ebm_ = roc_auc_score(y, ebm_proba)
            self.train_auc_erf_ = roc_auc_score(y, erf_proba)
        except Exception:
            self.train_auc_ebm_ = None
            self.train_auc_erf_ = None

    def comparison_summary(self, X_test=None, y_test=None, return_string=False):
        """Print a comparative summary of both models.

        Parameters
        ----------
        X_test : array-like, optional
            Test data for out-of-sample AUC comparison.
        y_test : array-like, optional
            Test labels.
        return_string : bool, default=False
            If True, return summary as string instead of printing.

        Returns
        -------
        text : str or None
        """
        check_is_fitted(self, ["ebm_", "erf_"])

        lines = [
            "=" * 70,
            "Dual Glass-Box — EBM + ExpertRuleFit Summary",
            "=" * 70,
            "",
            "EBM (InterpretML ExplainableBoostingClassifier)",
            f"  Terms: {len(self.ebm_.term_names_)}",
            f"  Interactions: {sum(1 for t in self.ebm_.term_features_ if len(t) == 2)}",
        ]

        if self.train_auc_ebm_ is not None:
            lines.append(f"  Train AUC: {self.train_auc_ebm_:.4f}")

        lines.extend([
            "",
            "Rule Extraction from EBM",
            f"  Shape function rules: {self.extraction_report_.n_shape_rules}",
            f"  Interaction rules: {self.extraction_report_.n_interaction_rules}",
            f"  Injected as: {self.inject_ebm_rules_as}",
        ])

        if self.extraction_report_.top_interactions:
            lines.append("  Top interactions:")
            for f1, f2, imp in self.extraction_report_.top_interactions[:5]:
                lines.append(f"    {f1} x {f2} (importance={imp:.4f})")

        lines.extend([
            "",
            "ExpertRuleFit (Bootstrap-Stabilized Logistic)",
            f"  Stable features: {self.erf_.n_stable_rules_}",
            f"  Active rules: {len(self.erf_.get_selected_rules())}",
        ])

        if self.train_auc_erf_ is not None:
            lines.append(f"  Train AUC: {self.train_auc_erf_:.4f}")

        if hasattr(self.erf_, "confirmatory_all_active_"):
            status = "ALL PRESERVED" if self.erf_.confirmatory_all_active_ else "SOME ELIMINATED"
            lines.append(f"  Confirmatory rules: {status}")

        # EBM rules that survived
        ebm_rule_report = self.get_ebm_rules_in_model()
        n_survived = sum(1 for r in ebm_rule_report if r["survived_bootstrap"])
        n_total = len(ebm_rule_report)
        lines.extend([
            "",
            f"EBM Rules in Final Model: {n_survived}/{n_total} survived",
        ])
        for r in ebm_rule_report[:5]:
            marker = "+" if r["survived_bootstrap"] else "-"
            lines.append(
                f"  [{marker}] {r['name']} "
                f"(coef={r['erf_coefficient']:+.4f})"
            )

        # Test set comparison
        if X_test is not None and y_test is not None:
            try:
                import pandas as pd

                X_test = np.asarray(X_test, dtype=np.float64)
                y_test = np.asarray(y_test, dtype=np.float64).ravel()
                df_test = pd.DataFrame(X_test, columns=self.feature_names_)
                ebm_auc = roc_auc_score(y_test, self.ebm_.predict_proba(df_test)[:, 1])
                erf_auc = roc_auc_score(y_test, self.erf_.predict_proba(X_test)[:, 1])
                lines.extend([
                    "",
                    "Test Set Comparison",
                    f"  EBM AUC:            {ebm_auc:.4f}",
                    f"  ExpertRuleFit AUC:  {erf_auc:.4f}",
                    f"  Delta:              {ebm_auc - erf_auc:+.4f}",
                ])
            except Exception as exc:
                lines.append(f"\n  Test comparison failed: {exc}")

        lines.extend([
            "",
            "=" * 70,
        ])

        text = "\n".join(lines)

        if return_string:
            return text

        print(text)
        return None

    def summary(self, return_string=False):
        """Alias for comparison_summary (no test data)."""
        return self.comparison_summary(return_string=return_string)
