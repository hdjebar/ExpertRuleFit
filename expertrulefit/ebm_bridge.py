"""EBM → ExpertRuleFit bridge.

Uses InterpretML's Explainable Boosting Machine (EBM) to discover
feature interactions, then converts them to confirmatory/optional rules
for ExpertRuleFit.

Pipeline:
    1. Fit EBM (GA2M) → discovers pairwise interactions automatically
    2. Extract top-K interactions by importance
    3. For each interaction pair, find the best threshold combination
       by scanning data quantiles and maximizing correlation with target
    4. Convert to threshold rules: "feature_i > t_i AND feature_j > t_j"
    5. Return as confirmatory_rules list for ExpertRuleFit.fit()

Usage:
    from expertrulefit.ebm_bridge import discover_interaction_rules

    rules = discover_interaction_rules(X_train, y_train, feature_names,
                                       top_k=3, rule_type="confirmatory")
    erf = ExpertRuleFit(max_rules=50)
    erf.fit(X_train, y_train, feature_names=fn, confirmatory_rules=rules)
"""

import numpy as np


def discover_interaction_rules(
    X, y, feature_names, top_k=3, rule_type="confirmatory",
    max_interactions=10, random_state=42,
):
    """Discover interaction rules from EBM and format for ExpertRuleFit.

    EBM identifies WHICH feature pairs interact (step 1). Then for each
    pair, we scan data quantiles to find the threshold combination that
    creates the most discriminative binary rule (step 2).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data (numpy array or pandas DataFrame).

    y : array-like of shape (n_samples,)
        Binary target.

    feature_names : list of str
        Feature names matching columns of X.

    top_k : int, default=3
        Number of top interactions to extract.

    rule_type : str, default="confirmatory"
        Either "confirmatory" (near-zero penalty, never eliminated)
        or "optional" (reduced penalty, can be eliminated).  The returned
        rule dicts include a ``category`` key set to this value, and the
        list is ready to pass directly to the matching ``ExpertRuleFit.fit``
        parameter (``confirmatory_rules`` or ``optional_rules``).

    max_interactions : int, default=10
        Number of interactions for EBM to discover.

    random_state : int, default=42
        Random state for EBM.

    Returns
    -------
    rules : list of dict
        Each dict has ``name``, ``evaluate``, and ``category`` keys,
        compatible with ExpertRuleFit's ``confirmatory_rules`` or
        ``optional_rules`` parameter.

    ebm : ExplainableBoostingClassifier
        The fitted EBM model (for inspection / explanation).
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

    # Step 1: Fit EBM to discover which feature pairs interact
    df = pd.DataFrame(X, columns=feature_names)
    ebm = ExplainableBoostingClassifier(
        interactions=max_interactions,
        random_state=random_state,
        n_jobs=1,  # avoid joblib pickle issues on Python 3.14+
    )
    ebm.fit(df, y)

    # Collect interaction terms with importances
    interactions = []
    importances = ebm.term_importances()
    for i, term_feats in enumerate(ebm.term_features_):
        if len(term_feats) == 2:
            interactions.append({
                "index": i,
                "features": term_feats,
                "name": ebm.term_names_[i],
                "importance": importances[i],
            })

    # Sort by importance, take top_k
    interactions.sort(key=lambda x: x["importance"], reverse=True)
    interactions = interactions[:top_k]

    if rule_type not in ("confirmatory", "optional"):
        raise ValueError(
            f"rule_type must be 'confirmatory' or 'optional', got '{rule_type}'"
        )

    # Step 2: For each interaction, find optimal thresholds from data
    rules = []
    for inter in interactions:
        feat_a, feat_b = inter["features"]
        name_a = feature_names[feat_a]
        name_b = feature_names[feat_b]

        thresh_a, thresh_b, corr, direction = _find_best_thresholds(
            X, y, feat_a, feat_b
        )

        # Encode direction into the rule:
        #   "risk"       → (a > ta) & (b > tb)  → high values = positive class
        #   "protective" → invert output so coefficient sign stays intuitive
        #     The rule fires when the condition is NOT met (absence of
        #     protective factors), which yields a positive coefficient
        #     meaning "lacking this protection increases risk".
        if direction == "protective":
            rule_name = (
                f"EBM:NOT({name_a}>{thresh_a:.4f} & {name_b}>{thresh_b:.4f}) "
                f"({direction}, imp={inter['importance']:.3f})"
            )
            rule_fn = _make_rule_fn(
                feat_a, thresh_a, feat_b, thresh_b, invert=True,
            )
        else:
            rule_name = (
                f"EBM:{name_a}>{thresh_a:.4f} & {name_b}>{thresh_b:.4f} "
                f"({direction}, imp={inter['importance']:.3f})"
            )
            rule_fn = _make_rule_fn(feat_a, thresh_a, feat_b, thresh_b)

        rules.append({
            "name": rule_name,
            "evaluate": rule_fn,
            "category": rule_type,
        })

    return rules, ebm


def _find_best_thresholds(X, y, feat_a, feat_b):
    """Find threshold pair that creates the most discriminative interaction rule.

    Scans quantiles of both features and picks the combination that
    maximizes absolute correlation between the binary rule and target.

    Returns (thresh_a, thresh_b, correlation, direction).
    """
    percentiles = [20, 30, 40, 50, 60, 70, 80]
    quantiles_a = np.percentile(X[:, feat_a], percentiles)
    quantiles_b = np.percentile(X[:, feat_b], percentiles)

    best_corr = -1
    best_ta = quantiles_a[3]  # fallback: median
    best_tb = quantiles_b[3]
    best_sign = 1.0

    for ta in quantiles_a:
        for tb in quantiles_b:
            rule_val = ((X[:, feat_a] > ta) & (X[:, feat_b] > tb)).astype(float)
            # Skip rules that fire for <5% or >95% of samples
            frac = rule_val.mean()
            if frac < 0.05 or frac > 0.95:
                continue
            if rule_val.std() < 1e-10:
                continue
            corr = np.corrcoef(rule_val, y)[0, 1]
            if abs(corr) > best_corr:
                best_corr = abs(corr)
                best_ta = ta
                best_tb = tb
                best_sign = np.sign(corr)

    direction = "risk" if best_sign > 0 else "protective"
    return best_ta, best_tb, best_corr, direction


def _make_rule_fn(feat_a, thresh_a, feat_b, thresh_b, *, invert=False):
    """Create a rule evaluation function with captured thresholds.

    Parameters
    ----------
    invert : bool, default=False
        If True, return ``1 - base_rule`` so that the rule fires when the
        joint condition is *not* met (used for "protective" direction).
    """
    def evaluate(X, feature_names):
        X = np.asarray(X, dtype=np.float64)
        base = ((X[:, feat_a] > thresh_a) & (X[:, feat_b] > thresh_b)).astype(np.float64)
        return (1.0 - base) if invert else base
    return evaluate


def summarize_ebm_interactions(ebm, feature_names=None):
    """Print a summary of EBM's discovered interactions.

    Parameters
    ----------
    ebm : ExplainableBoostingClassifier
        Fitted EBM model.

    feature_names : list of str, optional
        If not provided, uses ebm.term_names_.
    """
    importances = ebm.term_importances()

    print("=" * 60)
    print("EBM Interaction Summary")
    print("=" * 60)

    # Main effects
    print("\nMain effects:")
    main_effects = []
    for i, tf in enumerate(ebm.term_features_):
        if len(tf) == 1:
            main_effects.append((ebm.term_names_[i], importances[i]))
    main_effects.sort(key=lambda x: x[1], reverse=True)
    for name, imp in main_effects:
        print(f"  {imp:.4f} | {name}")

    # Interactions
    print("\nInteractions:")
    interactions = []
    for i, tf in enumerate(ebm.term_features_):
        if len(tf) == 2:
            interactions.append((ebm.term_names_[i], importances[i]))
    interactions.sort(key=lambda x: x[1], reverse=True)
    for name, imp in interactions:
        print(f"  {imp:.4f} | {name}")

    print("=" * 60)
