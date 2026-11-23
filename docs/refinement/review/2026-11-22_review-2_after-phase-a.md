Summary

The paper proposes EXPERTRULEFIT, a deterministic, sparse rule-ensemble classifier designed for regulated decision-making settings where exact re-execution and inclusion of expert-specified rules are crucial. The method combines fixed-seed tree-based rule generation, per-feature penalty weighting to protect expert rules, a selection-frequency filter via fixed bootstrap elastic-net fits, and a final refit that attempts to ensure confirmatory rules remain active. On three credit-scoring datasets, the authors show bitwise-deterministic outputs (100/100 identical rule sets across user-provided seeds) with test AUC comparable to RuleFit, while clearly distinguishing determinism from stability under data perturbation and outlining a protocol for the latter.
Strengths

Technical novelty and innovation
Integrates stability selection ideas with deterministic seeds and differential penalty weights to provide an auditable, exactly reproducible rule-ensemble pipeline tailored to regulatory contexts.
Practical mechanism to “protect” expert-imposed rules through dramatically reduced penalties and a constrained refit; useful diagnostic tooling (sign checks, VIF, absorption indicators) anticipates common failure modes in expert/ML hybrids.
Clear and explicit articulation of an “execution envelope” for bitwise determinism (pinned software stack, single-thread BLAS/OpenMP, fixed seeds), which is underappreciated but critical for auditability.
Experimental rigor and validation
Honest and careful framing: results demonstrate determinism-by-design on a fixed split, not stability under data perturbation; limitations and a concrete perturbation protocol are provided.
Side-by-side comparison with RuleFit under seed variation highlights a real deployability pain point (seed sensitivity) and positions the contribution convincingly for its stated use case.
Clarity of presentation
The method is described succinctly and operationally (Algorithm 1 with key hyperparameters), including extensions (EBM-discovered interactions, stacking) and post-hoc calibration options.
Reproducibility boundaries and non-claims are stated crisply; terminology (confirmatory vs optional rules) is consistent and well-motivated.
Significance of contributions
Deterministic, auditable training and enforced inclusion of expert rules address practical blockers for adoption of rule ensembles in credit, insurance, and public-sector applications.
Open-source implementation with tests and containerization increases immediate utility for practitioners.
Weaknesses

Technical limitations or concerns
The “constrained refit” enforces zero penalty on confirmatory rules but does not strictly guarantee nonzero coefficients; inclusion could still result in βc = 0 at the (unpenalized) optimum in degenerate or highly collinear cases.
Very small penalty weights (wc = 1e−8) invite absorption of correlated signal by confirmatory rules; while diagnostics flag this, the method currently offers no mitigation beyond manual review.
No sign constraints are enforced; contradictory learned signs are only flagged, which can undermine the intent of expert policies in regulated contexts.
Rule generation and selection budgets (≤50 candidate rules; B = 10 bootstraps; τ = 0.8) may be too coarse for complex datasets; thresholding with fixed, non-resampled draws weakens the spirit of stability selection.
Experimental gaps or methodological issues
Evaluation focuses on determinism vs. seed sensitivity on a single split; no repeated-split CIs, no calibration curves/Brier scores, and no assessment under data perturbation (though proposed).
Limited baselines: SIRUS (explicitly designed for stability), L2-logistic, FIRE, LIRE, and modern boosting-based rule learners (e.g., COB) are not included empirically.
No ablations to disentangle the contributions of each pipeline component (fixed boosting seed, frequency filter, penalty weights, constrained refit) to AUC and rule-set stability.
Clarity or presentation issues
It remains ambiguous whether the “confirmatory” guarantee is logical inclusion (feature present) or numerical inclusion (β ≠ 0); the algorithm and text suggest the former but imply the latter in some places.
Preprocessing assumptions (missing data, categorical encoding, monotonic binning common in credit) are not fully specified within the execution envelope.
Missing related work or comparisons
While SIRUS, FIRE, and LIRE are cited, empirical comparisons to SIRUS (stability) and to combinatorial methods (BRL, MaxSAT-based IMLI) are absent; recent work on orthogonal boosting for compact rules is also relevant for stability/interpretability-accuracy trade-offs.
Group/differential penalization literature (e.g., gren) suggests principled alternatives to fixed penalty factors that could reduce absorption and tune expert rule influence.
Detailed Comments

Technical soundness evaluation
The scaling-based penalty weighting is standard and sound; using elastic-net with per-feature weights is appropriate. However, the extremely small confirmatory weight (1e−8) can cause ill-conditioning and dominance in correlated designs; consider bounding wc away from machine epsilon and validating the choice.
The “constrained refit” is implemented as an unpenalized refit for confirmatory rules; this does not strictly impose nonzero coefficients or expected-sign constraints. If the goal is true enforcement, formulate box constraints (βc ≥ ε or sgn(βc) enforced) and solve with a constrained optimizer (e.g., projected/proximal methods, LBFGS-B via a smooth penalty, or convex-constrained logistic if sign constraints only).
Fixed, small B with deterministic bootstrap seeds ensures reproducibility but weakens the inferential meaning of selection frequencies. If stability selection is an objective, adopt resampling schemes with error control (Shah & Samworth) and sufficient B; if determinism is paramount, keep fixed draws but characterize sensitivity to B and τ.
The candidate rule budget (≤50 from depth-4 boosting) may bottleneck accuracy on some tasks. Provide a heuristic or validation scheme to adapt the rule budget to dataset complexity and report coverage/length of retained rules.
Experimental evaluation assessment
The determinism benchmark is valuable for the stated goal but relatively easy to pass by construction. Strengthen empirical claims with:
Repeated train/test splits with CIs for AUC, ECE/Brier scores, and reliability plots (with and without calibration).
Perturbation stability as proposed: Jaccard overlaps of rule sets and per-sample local explanation stability (e.g., overlap of active rules per instance), together with discrimination/consistency under data resampling.
Ablations: (i) no expert-rule weighting; (ii) no frequency filter; (iii) variable B and τ; (iv) no constrained refit; report impacts on accuracy, sparsity, and stability.
Baselines: SIRUS (primary for stability), L2-logistic, RuleFit with fixed internal seeds (upper-bound reproducibility), FIRE/LIRE, and an orthogonal-boosting rule learner. For each, compare rule-set stability and rules-per-prediction.
Include computational cost: wall-clock time, sensitivity to single-threading, and memory, given the execution envelope constraints.
Comparison with related work (using the summaries provided)
SIRUS emphasizes stability via quantile-discretized splits and frequency thresholds; it reports stability under perturbation and near-deterministic reruns. A direct empirical/qualitative comparison is essential to situate EXPERTRULEFIT as “deterministic with expert-rule protection” versus “intrinsically stable rule mining.”
LIRE optimizes local interpretability (fewer rules per prediction) at comparable accuracy to RuleFit; even if targeted properties differ, reporting local support for EXPERTRULEFIT would contextualize cognitive load.
FIRE compresses ensembles via MCP and fused penalties, often improving sparse rule accuracy; assessing whether EXPERTRULEFIT’s determinism can be retained while adopting less biased sparsity (e.g., MCP) is an interesting direction.
IMLI (MaxSAT) and BRL show different stability/parsimonious regimes; while solver randomness can affect reproducibility, experiments under fixed seeds would help benchmark the interpretability–accuracy frontier.
Group-regularized elastic net (gren) provides a principled path for differential penalties; replacing fixed wc and wo with learned group multipliers informed by expert annotations could reduce absorption while preserving expert influence.
Recent orthogonal boosting (COB) achieves smaller, higher-coverage rules; while not designed for determinism per se, comparing complexity–risk trade-offs will clarify any accuracy sacrifices from EXPERTRULEFIT’s fixed-generation-and-filtering pipeline.
Discussion of broader impact and significance
The determinism envelope and expert-rule protection directly address auditability and governance requirements in credit and public administration. However, without sign constraints or robust perturbation stability, explanations may still shift between retrains—a material risk for user trust.
The tax-audit application design is thoughtful (selection bias, legality, human-in-the-loop). Consider extending the implementation with: inverse-probability weighting support, per-group performance reporting, monotonic constraints where mandated, and recourse-style local explanations to improve transparency.
Caution that determinism can be misconstrued as robustness or fairness; documentation and dashboards should foreground this distinction and provide stability diagnostics over time.
Questions for Authors

Does the “constrained refit” step provably ensure βc ≠ 0 for every confirmatory rule, or can coefficients still end at zero? If the latter, would you consider explicit lower-bound/sign constraints?
How sensitive are results to wc and wo? Can you report an ablation sweeping these weights (log-scale) and quantifying absorption (e.g., coefficient shifts, VIFs) and AUC?
Why cap candidate rules at 50 and depth at 4? What is the impact of relaxing these on performance, sparsity, and runtime, and does determinism still hold with larger candidate sets?
In the perturbation-stability protocol, which resampling scheme will you adopt given recent evidence that repeated low-fold CV or OOB bootstrap better approximates instability for elastic-net pipelines?
How are missing values and high-cardinality categoricals handled deterministically (imputation/bucketing/one-hot order), and are these steps part of the pinned, tested execution envelope?
For the EBM-based discover_interaction_rules, is the EBM trained only on the training fold when used inside CV/stacking? How do you avoid leakage when stacking with out-of-fold predictions?
Could adaptive or group-regularized penalties (e.g., gren-style empirical Bayes) replace fixed penalty factors for expert vs optional vs tree rules to balance protection and absorption?
Have you measured per-instance explanation stability (overlap of active rules per prediction) under data perturbation, in addition to global Jaccard overlaps of rule sets?
Would sign-constrained confirmatory rules via a convex-constrained logistic (or projected proximal) optimizer be feasible in your current codebase, and what trade-offs in runtime/accuracy do you anticipate?
Can you provide calibration results (reliability curves, ECE/Brier) before/after optional calibration, and discuss whether determinism affects calibration choices (e.g., isotonic’s piecewise constants)?
How does wall-clock time scale with n, p, and rule budget under single-thread constraints, and what are typical runtimes on the three datasets?
Do you plan to expose row-order invariance tests and hashes of key artifacts (rule sets, coefficients, predictions) to operationalize the re-execution guarantee in production?
Overall Assessment

This is a well-written, practitioner-focused contribution addressing a real deployment barrier: exact reproducibility and protection of expert-imposed rules in rule ensembles. The proposed pipeline is technically sound, the determinism guarantee is clearly and responsibly scoped, and the implementation details (diagnostics, containerization) reflect commendable engineering care. However, the current empirical evidence is narrow: determinism is largely guaranteed by design, not demonstrated under perturbations; baselines tailored to stability (SIRUS) and compactness (LIRE/FIRE/COB) are absent; and ablations clarifying which components matter are missing. The most significant methodological limitations are the lack of enforced sign/nonzero constraints for confirmatory rules and the potential for signal absorption due to near-unpenalized expert features. With the proposed perturbation-stability study, principled handling of confirmatory constraints, calibration evaluation, and richer baselines/ablations, this work could be a compelling, high-impact paper for applied ML and model governance venues. In its current form, it is a strong technical report with practical value to the community but not yet at the level of a top-tier research venue.
