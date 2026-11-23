Summary

This paper introduces ExpertRuleFit, a deterministic variant of RuleFit tailored for regulated decision-making. The method fixes all randomness in rule generation and selection, incorporates expert-provided “confirmatory” and “optional” rules with reduced penalties, filters candidate rules using selection frequency across fixed bootstrap elastic-net fits, and forces inclusion of confirmatory rules in a constrained refit. On three public credit datasets, it achieves bitwise-identical outputs across runs and test AUC within 0.005 of RuleFit’s mean, while explicitly scoping the contribution as determinism by design (not stability under data perturbation).
Strengths

Technical novelty and innovation
Introduces a practical combination of fixed-seed rule generation, frequency-based filtering, and per-feature penalty weighting to guarantee deterministic outputs.
Provides a workflow for protecting expert rules via reduced penalty weights and a final constrained refit to guarantee inclusion, along with diagnostics for absorption/collinearity.
Offers an implementation with pinned environments and a clear “execution envelope,” which is valuable for auditability in regulated contexts.
Experimental rigor and validation
Reproducibility protocol is carefully specified and checked at the bitwise level, including pinned versions and single-threaded execution.
Sensitivity study on penalty weights includes solver convergence diagnostics and highlights pitfalls of extreme weighting choices.
Clarity of presentation
The paper is transparent about what is and is not demonstrated; it clearly distinguishes determinism from stability under perturbation and proposes a future protocol for the latter.
The algorithm is described step-by-step with practical implementation details (standardization choices, solver settings, thresholds).
Significance of contributions
Addresses a real and under-discussed need in highly regulated domains: re-executability with auditable, fixed outputs and the ability to preserve mandated rules.
The auditability framing (rule provenance, confirmatory/optional distinction, envelope specification) can inform best practices beyond this specific method.
Weaknesses

Technical limitations or concerns
Determinism is achieved primarily by fixing seeds, using a pinned stack, and carefully controlling execution; this is more engineering/process than algorithmic novelty, and its generality across platforms or future library changes is fragile.
Confirmatory rules are included but not sign-constrained; the paper demonstrates that coefficients can conflict with expert intent in converged fits, undermining the goal of protecting expert knowledge as intended.
Frequency filtering uses a small fixed number of bootstraps (B=10) and a fixed threshold (τ=0.8), without leveraging stability-selection error control or quantifying the impact of these hyperparameters.
The refit step that unpenalizes confirmatory rules can increase collinearity and absorption effects; diagnostics are helpful but not corrective.
Experimental gaps or methodological issues
No stability-under-perturbation results (subsampling/bootstrapping across data) are provided; SIRUS is acknowledged but not compared experimentally.
Evaluation is confined to three relatively small credit datasets, single train/test splits, no confidence intervals, and no statistical testing; model complexity and calibration are not thoroughly assessed.
Comparisons omit additional relevant baselines for rule ensembles and interpretable models (e.g., FIRE, LIRE, Node Harvest, RCProb/RuleCOSI+, RPTE, EBM as a strong interpretable non-rules baseline).
Runtime overhead is substantial (minutes to hours single-threaded vs seconds for RuleFit) without a principled speed–accuracy–determinism trade-off analysis.
Clarity or presentation issues
More detail is needed on the precise rule generation details (e.g., how rules are limited to 50, the selection of linear vs rule terms in the elastic-net path, and how thresholds are derived/deduplicated).
The constrained refit is described succinctly; stronger mathematical specification (e.g., optimization problem with zero-penalty block and its effect on convexity/uniqueness) would help.
Missing related work or comparisons
Limited coverage of recent rule extraction and stability-focused methods: FIRE, RCProb/RuleCOSI+, TE2Rules, RPTE, Node Harvest, compressed rule ensembles, and group-penalty methods like gren that relate to differential shrinkage.
Monotonicity-constrained interpretable models (e.g., monotone GBMs, MNAMs) and globally interpretable scorecard-like architectures are relevant for regulated settings and should be discussed as alternative ways to embed expert knowledge.
Detailed Comments

Technical soundness evaluation
The scaling trick to implement per-feature penalty weights via z/√w is sensible and aligns with Bayesian interpretations of elastic-net penalties. The caution about solver conditioning and the empirical choice of wc=1e−2 are appreciated.
However, relying on a final refit that unpenalizes confirmatory rules can exacerbate multicollinearity. Consider formalizing this as a block-penalized problem (with λ=0 for confirmatory, λ>0 for others) and analyzing identifiability or using ridge on non-confirmatory blocks to mitigate variance inflation.
The sign-check without enforcement is a notable mismatch to the claimed use case; adopting bounded optimization (e.g., sign-constrained logistic regression) or monotonicity constraints for specified features/rules would align with domain needs and prevent contradictory coefficients.
The frequency-filter step is an adaptation of stability selection but without guarantees; a principled choice of B and τ, or recourse to Shah and Samworth bounds (at least approximately), would add rigor. Moreover, fixed bootstrap indices contribute to determinism but do not test stability; at minimum, report selection frequencies and sensitivity to τ.
Experimental evaluation assessment
The determinism demonstration is convincing as an engineering result, but its scientific value hinges on broader stability and utility metrics. Add:
Stability-under-perturbation experiments against SIRUS and L2-regularized logistic regression, including Jaccard similarities and rule-wise selection frequencies.
Multiple splits with CIs for AUC, and report calibration metrics (Brier score, ECE) and reliability curves given the regulatory context.
Model complexity and interpretability measures (number of active rules, average rule length, overlap/conflict rates, and per-case explanation size), plus ablations of each component (rule-frequency filter, penalty weighting, constrained refit, EBM-discovered rules).
Runtime breakdown (rule extraction vs selection vs refits) and speed-ups (e.g., reduce CV grids, warm-starting across λ, more efficient solvers, vectorized bootstraps, caching preprocessing).
Include additional datasets commonly used in credit and tabular interpretability (HELOC, GMSC, LendingClub-like datasets if permitted, OpenML tabular benchmarks) to assess generality.
Comparison with related work (using the summaries provided)
SIRUS targets stability from forest structures and reports perturbation stability; it is the most directly relevant baseline and should be included experimentally.
FIRE improves sparsity and selection under correlation via MCP and fusion penalties; although not designed for determinism, its selection behavior under correlation is relevant to your stability/absorption concerns and should be compared.
RCProb/RuleCOSI+ and TE2Rules provide efficient rule extraction/fidelity trade-offs; while their goals differ (explanation of ensembles vs standalone classifiers), comparisons on rule compactness and fidelity could be informative.
RPTE proposes strict source disjointness and an auditability-first structure; its formal audit-load measures and decomposition could complement your determinism focus, and a head-to-head on audit metrics would be valuable.
Group-regularized elastic net (gren) relates to your per-feature penalty idea; citing it and discussing differences (empirical-Bayes estimation of group multipliers vs fixed penalty weights) would strengthen the conceptual grounding.
Monotonic interpretable models (EBM, MNAMs, two-layer additive risk models) are competitive, regulator-friendly alternatives; they can incorporate expert constraints and might offer different stability behaviors. Including EBM as a baseline (you already use it for discovering interactions) would provide a helpful point of reference.
Discussion of broader impact and significance
The deterministic execution envelope is genuinely useful for auditability, but many production teams already achieve reproducibility by logging seeds, code and data hashes, and pinning environments; articulating the incremental benefit of “seed-agnostic” determinism and protected-rule inclusion over rigorous logging would clarify the unique value.
The risk that weak confirmatory rules absorb signal is nontrivial in regulated settings; beyond diagnostics, mitigation strategies (e.g., sign constraints, group penalties that link confirmatory and related discovered rules, partialling-out approaches, or post-selection inference) should be considered.
The tax-audit application design section is thoughtful; selection-bias handling and legal basis are rightly foregrounded. A small simulation or public proxy task demonstrating the full pipeline (including inverse-probability weighting) would make this section more concrete.
Questions for Authors

How sensitive are the selected rules and AUC to the choice of τ and B in the frequency filter? Can you provide a grid study (e.g., B ∈ {10, 30, 50}, τ ∈ {0.6, 0.7, 0.8, 0.9}) with selection-frequency plots?
Did you evaluate using Shah and Samworth’s stability selection bounds to select τ (or per-family thresholds) and thereby control false discoveries? If not, what impeded that integration?
How does ExpertRuleFit compare to SIRUS and EBM in stability-under-perturbation and AUC across multiple splits? Can you include these baselines in the proposed protocol?
What is the average number and length of active rules per test instance, and how does that compare to RuleFit, SIRUS, and FIRE? Any preliminary user/analyst feedback on interpretability?
Could the constrained refit be replaced by a single convex problem with a zero-penalty block (confirmatory) and sign constraints? Have you explored solvers (e.g., coordinate descent with bounds) that maintain determinism and scalability?
How do you prevent confirmatory rules from encoding protected characteristics indirectly? Do you envision an auditing checklist (e.g., causal screens, fairness constraints) before rules are promoted to “confirmatory”?
Why is ignoring user-supplied random_state preferable to accepting it but logging a canonical seed chain? Would reproducibility guarantees differ materially?
How portable is the bitwise determinism across CPUs/OS/BLAS implementations beyond the pinned container? Have you tested Docker vs Singularity vs bare metal?
Can you report calibration metrics (Brier, ECE) and the effect of the optional calibration step on both discrimination and determinism (e.g., fold ordering effects)?
What ablation results show the marginal contribution of each component (fixed-seed GBM, frequency filter, penalty-weighted expert rules, constrained refit, EBM-discovered rules) to stability and AUC?
Overall Assessment

This work foregrounds an important practical issue—deterministic, auditable model re-execution—in a rules-based classifier tailored to regulated domains, and it provides a clean engineering recipe with a public implementation. The manuscript is candid about scope and limitations and offers a clear algorithmic description and reproducibility envelope. However, as a research contribution for a top-tier venue, the novelty is modest: determinism largely follows from engineering controls and hyperparameter fixing, the confirmatory/penalty mechanics are variations of known per-feature penalty factors and unpenalized blocks, and the central empirical claim (determinism) is asserted by design rather than discovered empirically. The experimental evaluation is narrow (three small datasets, single split, no perturbation stability, no calibration analysis) and omits several strong, immediately relevant baselines (SIRUS, EBM, FIRE, RPTE). The most compelling next steps would be: (i) a comprehensive stability-under-perturbation study with principled stability selection thresholds, (ii) sign-constrained confirmatory rules with a sound optimization formulation, (iii) broader datasets with statistical testing and calibration, and (iv) ablations and auditability metrics to quantify benefits over existing interpretable alternatives. With these extensions, the paper could offer a stronger scientific case while preserving its practical value to practitioners in regulated environments.
