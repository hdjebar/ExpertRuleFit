Summary

This paper proposes EXPERTRULEFIT, a deterministic variant of RuleFit designed for regulated decision-making. The method fixes all internal randomness, augments candidate rules with expert-provided “confirmatory” and “optional” rules via per-feature penalty weights, performs rule preselection through selection-frequency filtering across bootstrap elastic-net fits, and guarantees inclusion of confirmatory rules through a constrained refit step. On three credit datasets, it achieves identical rule sets across 100 repeated runs (by design) and AUC comparable to RuleFit, and it articulates a clear “execution envelope” under which bitwise determinism holds.
Strengths

Technical novelty and innovation
Combining fixed-seed rule generation, stability-selection-style frequency filtering, and per-feature penalty factors to protect expert rules is a thoughtful engineering of known ideas to meet audit and governance needs.
The explicit “confirmatory rule inclusion” refit step addresses a concrete regulatory/business requirement often overlooked by academic methods.
The paper precisely defines the reproducibility guarantee (bitwise determinism within a clearly specified execution envelope), which is rare in ML papers.
Experimental rigor and validation
The experiments cleanly demonstrate the intended property (seed insensitivity) and compare discrimination to a strong baseline (RuleFit).
Results are accompanied by transparent caveats about what is and is not being measured, avoiding overclaiming.
Clarity of presentation
The method is clearly described with a concise algorithm, unambiguous treatment of penalty weighting via feature rescaling, and explicit defaults.
The paper candidly lists limitations and provides an actionable plan for future perturbation-based stability evaluation.
The reproducibility section is unusually concrete (pinned stacks, single-threaded BLAS/OpenMP, container references).
Significance of contributions
Determinism and expert-rule protection are practically important in regulated domains (credit, insurance, tax) where re-execution and auditable explanations are mandatory.
An open-source implementation with tests and a container recipe increases potential impact and adoption by practitioners.
Weaknesses

Technical limitations or concerns
The determinism result largely follows from design choices (ignoring user seeds and fixing all internal seeds), rather than a methodological advance in stability; it does not address selection stability under data perturbation.
The heavy down-weighting (w_c = 1e−8) risks allowing confirmatory rules to absorb signal from correlated rules; absence of sign constraints can yield counterintuitive explanations that contradict expert intent.
The frequency-filtering step uses a small, fixed B=10 bootstraps and a hard threshold without error control; more principled variants (e.g., CPSS) are available.
Feature-rescaling to implement per-feature penalties changes both L1 and L2 geometry; sensitivity to w_c and w_o is not explored.
Experimental gaps or methodological issues
No evaluation of stability under data perturbations (bootstrap/subsample) is provided, and there are no confidence intervals for AUC or repeated-split validation.
Missing comparisons to inherently stable rule methods (e.g., SIRUS) and to other modern rule-ensemble frameworks emphasizing stability/interpretability (e.g., HorseRule, FIRE, SR4‑Fit).
Limited ablations: contributions of each component (fixed rule generation, frequency filter, confirmatory weighting, constrained refit) are not disentangled.
Calibration and fairness are discussed but not empirically assessed (no reliability curves, Brier scores, or group-wise performance analysis).
Clarity or presentation issues
The paper sometimes conflates “stability,” “determinism,” and “reproducibility”; while distinctions are acknowledged, terminology could be tightened throughout.
The discover_interaction_rules extension appears to elevate data-discovered interactions to confirmatory status, which conceptually blurs “expert-imposed” versus “data-discovered” rules.
Missing related work or comparisons
Related work coverage omits several pertinent modern rule-extraction/selection approaches (e.g., HorseRule, SR4-Fit, IP-based set partitioning of rules, COB) that also emphasize parsimony and/or stability.
No discussion of Bayesian or CPSS-style stability selection that offers error control or of recent Bayesian stability-selection augmentations enabling domain knowledge via priors.
Detailed Comments

Technical soundness evaluation
The core pipeline is technically sound as an engineering integration: (i) deterministic rule extraction from a fixed-seed tree ensemble; (ii) per-feature penalty weighting via rescaling; (iii) frequency-filtering across fixed bootstrap resamples; and (iv) a constrained refit to guarantee inclusion of confirmatory rules. Each step individually aligns with established methods.
Implementing per-feature penalty factors through column rescaling is reasonable but couples L1 and L2 penalties in ways that merit sensitivity analysis. The extreme setting w_c = 1e−8 effectively unpenalizes confirmatory rules under L1 and L2, which can dominate when correlated with learned rules. Consider bounding the effective penalty reduction and providing diagnostics (variance inflation, coefficient path plots) to detect absorption.
The constrained refit “if any confirmatory coefficient is zero” is a practical heuristic. A more principled formulation would treat confirmatory rules as unpenalized from the outset or enforce linear inequality constraints (e.g., sign/box constraints) in a single convex program. Sign constraints are especially important to align with expert intent.
The fixed B=10 bootstrap resamples and τ=0.8 threshold are ad hoc. CPSS (complementary pairs) offers finite-sample bounds under mild assumptions, and Bayesian stability selection can incorporate domain priors coherently. Incorporating one of these would strengthen selection error guarantees and connect more directly to the authors’ regulatory-use motivation.
The cap of “at most 50 rules” from boosting (depth 4, 250 trees) is conservative and aids readability, but the rationale and trade-offs versus performance are not quantified. An ablation across rule caps and depths would clarify capacity limits.
Experimental evaluation assessment
The “identical rule sets across 100 runs” is expected given the algorithm ignores random_state and fixes internal seeds; as the paper notes, this is not evidence of stability under data perturbation. To substantiate the broader claims, execute the proposed perturbation study (bootstrap/subsample) and report selection-frequency distributions, pairwise Jaccard over rule sets, and instance-level prediction stability.
Add SIRUS (and ideally FIRE, HorseRule, SR4‑Fit) as baselines, comparing both stability metrics and predictive performance with confidence intervals across repeated splits. Also include a simple penalized logistic regression with monotonicity constraints (when applicable) as a baseline favored in credit risk.
Provide ablations isolating the contributions of (a) fixed rule generation vs. standard RuleFit’s stochasticity, (b) stability selection filter vs. direct L1, (c) expert-rule weighting vs. none, and (d) constrained refit. Report rule counts, average rule length, and overlap metrics.
Evaluate calibration (reliability curves, Brier score) and the effect of post-hoc calibration options; penalized logistic models often require calibration in deployment.
Report runtime and scalability (wall-clock and memory) as a function of n, p, number of candidate rules, and B. Deterministic constraints (single-threaded BLAS/OpenMP) may have performance implications important for practitioners.
Comparison with related work (using the summaries provided)
SIRUS focuses on stability via forest structure and rule frequency; it is a natural comparator for selection stability under data perturbations and provides a strong baseline for this paper’s stated goals.
FIRE introduces fusion and nonconvex penalties to shrink and compress rule sets, improving sparsity and interpretability; while not focused on determinism, it meaningfully advances stability of structure and merits empirical comparison.
HorseRule’s rule-structured horseshoe prior encodes preferences for simpler/more general rules, often yielding sparser and more stable ensembles than L1; this is relevant to your stability narrative and expert-guided preferences.
SR4‑Fit adapts SR3 to RuleFit and explicitly assesses rule-set stability (Dice–Sørensen overlap) alongside readability; its interpretability metrics (RUS, E‑IPS) could be adopted here to quantify readability in addition to rule counts.
CPSS (Shah–Samworth) and Bayesian stability selection (2410.21914) provide error-control and uncertainty quantification for selection frequencies, aligning with regulatory needs. Substituting your fixed bootstrap filter with CPSS or augmenting it with a Bayesian posterior over selection probabilities and expert priors would materially strengthen the methodology and its governance claims.
TE2Rules and IP-based set-partitioning extract compact, deterministic rule lists from tree ensembles with different objectives (fidelity/partitioning). They provide alternative paths to determinism and parsimony that should be acknowledged and, where feasible, compared on compactness and fidelity/accuracy.
COB (orthogonal gradient boosting) aims for fewer, more general rules with corrective updates; while not deterministic by design, its rule parsimony and interpretability outcomes offer useful reference points for model size/complexity.
Discussion of broader impact and significance
The work targets a high-importance gap in regulated ML: auditability of model re-execution and retention of mandated criteria. The clear execution envelope and containerization are strong steps toward operational reproducibility.
There are governance risks: unpenalized or near-unpenalized confirmatory rules can entrench incorrect or biased expert priors; and without sign constraints, the system can report a rule as “active” while flipping its intuitive effect. Adding guardrails (sign/box constraints, sensitivity analysis, and explicit documentation of expert-rule provenance) would mitigate these risks.
The tax-audit design section is thoughtful, especially the attention to selection bias and legal constraints. Empirical validation on proxy datasets (with injected selection bias and IPW correction) would make this section more actionable.
Finally, determinism alone is not sufficient for defensible explanations; stability under plausible data shifts and transparent uncertainty quantification for rule inclusion are essential and align with the paper’s proposed next steps.
Questions for Authors

How sensitive are results (rule set, AUC, and per-rule coefficients) to the confirmatory and optional penalty weights (w_c, w_o)? Can you provide a sensitivity analysis and recommended ranges?
Why not adopt CPSS (complementary pairs stability selection) or a Bayesian stability-selection augmentation to obtain selection error control and principled incorporation of expert beliefs rather than a fixed B and τ?
Can you report an ablation showing the effect of each component (fixed-seed rule generation, frequency filtering, expert weighting, constrained refit) on rule-set determinism and accuracy?
The discover_interaction_rules extension seems to promote data-discovered interactions to “confirmatory” status. Shouldn’t these be treated as optional (penalized) rules instead to avoid enshrining accidental correlations?
How do you detect and handle cases where confirmatory rules are highly collinear with generated rules and absorb most of the weight? Do you provide diagnostics (e.g., VIF, coefficient paths) or safeguards (caps on effect size)?
Do you plan to add sign or monotonicity constraints for confirmatory rules, and if so, what optimization approach (e.g., box-constrained logistic regression) will you use within scikit-learn-compatible tooling?
Could you provide repeated-split evaluation with confidence intervals for AUC and calibration metrics (Brier, ECE), and compare to SIRUS, HorseRule, FIRE, and SR4‑Fit?
What is the computational profile (runtime/memory) as a function of n, p, number of candidate rules, and B—especially under the single-threaded determinism requirement?
How do you reconcile deterministic execution with minor nondeterminisms at the BLAS/linear algebra level across CPUs (even single-threaded can differ)? Does your container lock hardware features (e.g., AVX variants), and have you validated cross-machine bitwise identity?
In deployments where feature order can drift (e.g., ETL changes), how do you enforce the “stable feature order” assumption and detect schema drift to preserve the execution envelope?
Overall Assessment

This is a well-written, practically motivated paper that squarely addresses a real gap in regulated ML practice: ensuring that rule-based models are re-executable with bitwise-identical outputs and that mandated expert criteria remain present in the model. The contribution is primarily an engineering integration of established ideas—deterministic seeds, per-feature penalty factors, selection-frequency filtering, and a constrained refit—into a coherent estimator with an explicit and useful execution envelope. The paper is commendably transparent about what is and is not shown: determinism by construction is demonstrated; stability under data perturbations and calibration/fairness are not yet evaluated. To reach top-tier impact, the work should (i) execute a thorough stability-under-perturbation study with principled stability selection (CPSS or Bayesian) and strong baselines (SIRUS, FIRE, HorseRule, SR4‑Fit), (ii) add sign/box constraints for confirmatory rules and diagnostics for collinearity/absorption, (iii) provide ablations, sensitivity analyses, and calibration/fairness evaluations, and (iv) report scalability and runtime under the deterministic compute constraints. With these additions, EXPERTRULEFIT could become a valuable tool for high-stakes, audited decision systems.