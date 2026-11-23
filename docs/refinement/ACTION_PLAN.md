# Action plan — ExpertRuleFit refinement

Ordered by impact on the paper's main weakness (no stability-under-perturbation evidence) and on governance risk. Reviewer references: "R:" points to the line of thought in `review/2026-11-22_paperreview-ai_review.md`.

Work on a branch (`refinement/<item>`), one PR per phase, tests green before merge. Keep the determinism guarantee intact: every new random draw must use a seed derived from `_BASE_SEED`.

## Phase A — quick governance fixes (≈ half a day)

### A1. Interaction rules found by EBM become optional by default
- **Why.** R: "elevates data-discovered interactions to confirmatory status… blurs expert-imposed vs data-discovered".
- **Do.** Give `discover_interaction_rules` (`expertrulefit/ebm_bridge.py`) a `role="optional"` default, documented. Update README examples that pass them as `confirmatory_rules`.
- **Done when.** Test asserts the default role; README and paper (section 3) say data-discovered rules are optional unless an expert promotes them.

### A2. Sign check and absorption diagnostics for confirmatory rules
- **Why.** R: no sign constraint, w_c = 1e-8 lets confirmatory rules absorb signal from correlated rules.
- **Do.**
  1. Let each confirmatory rule declare `expected_sign` (+1 / −1 / None) in `RuleSpec`.
  2. After fit, set `confirmatory_sign_ok_` and emit a `UserWarning` per rule whose sign contradicts the expectation.
  3. Add `diagnose_confirmatory()` returning, per confirmatory rule: coefficient, expected sign, max |correlation| with retained rules, VIF, and the coefficient change of correlated rules when the confirmatory rule is removed.
- **Done when.** Tests cover a synthetic case where the sign flips and one where absorption happens; `summary()` prints the diagnostics.

## Phase B — the missing evidence (≈ 1–1.5 days)

### B1. Perturbation stability study
- **Why.** Main weakness. R: "execute the proposed perturbation study… selection-frequency distributions, pairwise Jaccard, instance-level prediction stability".
- **Do.** New script `experiments/perturbation.py`:
  - for each dataset, 100 subsamples (90 %, without replacement) and 100 bootstraps of the training set, seeds derived from a fixed base;
  - fit RuleFit (fixed seed), ExpertRuleFit, and L2 logistic regression;
  - metrics: mean pairwise Jaccard of rule sets, per-rule selection frequency, rule count, mean rule length, instance-level prediction std on the fixed test set;
  - AUC, Brier and ECE with 95 % CI over 20 repeated stratified splits.
- **Done when.** Results saved as CSV + figures in `experiments/results/`, a table ready for the paper, and the run is reproducible from one command.

### B2. Ablation
- **Why.** R: "contributions of each component… not disentangled".
- **Do.** Variants: (a) full; (b) no frequency filter (direct elastic net); (c) no expert weighting; (d) no constrained refit; (e) random internal seed (to separate determinism from filtering). Same metrics as B1.
- **Done when.** One ablation table in the paper.

### B3. Sensitivity to penalty weights
- **Why.** R: sensitivity to w_c and w_o not explored.
- **Do.** Grid w_c ∈ {1e-8, 1e-6, 1e-4, 1e-2, 1}, w_o ∈ {0.1, 0.3, 1}; report AUC, rule-set Jaccard against default, confirmatory coefficient, absorption metric from A2.
- **Done when.** A small figure + a recommended range in the README.

### B4. Calibration
- **Do.** Reliability curves and Brier/ECE for `calibration=None`, `"sigmoid"`, `"isotonic"`, reusing the B1 splits.
- **Done when.** Paper states whether calibration should be on by default; README wording on "calibrated probabilities" adjusted.

## Phase C — baselines and method upgrades (≈ 2–3 days)

### C1. SIRUS baseline
- **Do.** Run SIRUS (R package `sirus`, via `rpy2` or a separate R script reading the same splits) on the B1 protocol. Add FIRE only if a maintained Python implementation exists.
- **Done when.** SIRUS row in the B1 tables.

### C2. Complementary-pairs stability selection (CPSS)
- **Why.** R: fixed B = 10 and τ = 0.8 are ad hoc; CPSS gives error control.
- **Do.** Option `selection="cpss"` with B complementary pairs (default 50), threshold chosen from the Shah–Samworth bound for a target expected number of false selections; deterministic seeds.
- **Done when.** B1 rerun with CPSS; paper reports both.

### C3. True sign constraints (optional)
- **Do.** Box-constrained logistic fit for confirmatory rules (e.g. scipy `L-BFGS-B` with bounds on those coefficients, elastic-net on the rest via proximal steps) — only if A2 shows sign flips are frequent.

### C4. Runtime and cross-machine determinism
- **Do.** Wall-clock and peak memory vs n, number of candidate rules and B, single-threaded. Run the container on two CPU types and compare output hashes.
- **Done when.** Runtime table + a statement of verified cross-machine identity (or its limits).

## Phase D — paper v2

- Rewrite sections 5–6 with B/C results; add ablation and sensitivity; tighten terminology (determinism / stability / reproducibility defined once in section 1 and used consistently — R: "sometimes conflates").
- Extend related work only with references verified in `notes/REFERENCES_TO_VERIFY.md`.
- Re-run paperreview.ai and Google Deep Review; store reviews in `review/`.

## Out of scope for now
- Regression / multiclass support.
- Tax-data experiments (no data access); a synthetic selection-bias + IPW experiment is possible later (R suggests it).

---

## Additions from review 2 (22/11/2026, after Phase A)

Review 2 (`review/2026-11-22_review-2_after-phase-a.md`) confirms review 1 and independently flags w_c = 1e-8 as an ill-conditioning risk, which matches the collapse measured in `FINDINGS_2026-11-22.md`. New items:

### A3. Define the confirmatory guarantee precisely (wording + test) — 1 hour
- **Why.** "Logical inclusion (feature present) or numerical inclusion (β ≠ 0)?" An unpenalised refit can still end with β_c = 0 in degenerate cases.
- **Do.** State in README and paper: *the rule is kept in the model and not shrunk by the penalty; its coefficient is estimated freely and may be ≈ 0 if the rule carries no signal.* Set `confirmatory_all_active_` from |β| > tol and warn otherwise. True enforcement (β ≥ ε or sign) moves to C3.

### A4. Preprocessing inside the execution envelope — with the w_c fix
- **Why.** Missing values, categorical encoding and column order are not specified.
- **Do.** With the internal standardisation (option 2), add a documented, deterministic preprocessing contract: numeric only or a fixed one-hot order, explicit missing-value policy (error by default), schema hash stored in the audit bundle.

### A5. Leakage check for EBM interactions and DualModel — 1–2 hours
- **Why.** Is the EBM trained only on the training fold inside CV/stacking?
- **Do.** Read `ebm_bridge.py` and `dual_model.py`; add a test proving out-of-fold use; fix if not.

### A6. Operational re-execution checks — 2 hours
- **Do.** Row-order invariance test (shuffle rows → same rule set, or documented difference), and SHA-256 hashes of rule set, coefficients and predictions in the audit bundle.

### B-updates
- **B1:** add *per-instance explanation stability* (overlap of active rules per test case across perturbations); state the resampling scheme (90 % subsampling + bootstrap; justify vs repeated low-fold CV).
- **B2:** add the baseline **RuleFit with fixed internal seeds** — the key control: it shows whether the frequency filter adds anything beyond fixing seeds.
- **B3:** also vary B ∈ {10, 25, 50} and τ ∈ {0.6, 0.7, 0.8, 0.9}; vary the rule budget (max_rules 50/100/200, depth 3/4/6) with runtime.
- **B4:** discuss isotonic calibration's piecewise-constant output under determinism.

### C-updates
- **C1:** baselines in priority: SIRUS > fixed-seed RuleFit (B2) > L2 logistic > FIRE/LIRE if maintained code exists. BRL, IMLI, COB: related work only, after verification.
- **C5 (new, optional):** group-regularised penalties (gren-style) instead of fixed w_c / w_o — research direction, not a fix.

### Positioning for paper v2
Both reviews converge: *strong technical report, practical value, not yet top-tier.* Target an applied ML / model governance venue or workshop; frame determinism as an auditability property, never as robustness or fairness (review 2 explicitly warns against this confusion).

---

## Additions from review 3 (23/11/2026, paper v2)

Review 3 (`review/2026-11-23_review-3_paper-v2.md`) repeats the core asks of reviews 1–2 (perturbation study, SIRUS, sign constraints, ablation, B/τ grid, calibration, runtime). Verdict unchanged: clean engineering recipe, modest novelty, narrow evaluation. New items:

### D1. Argue the value of seed-agnostic determinism over logging seeds — paper, 1 hour
- **Why.** "Many teams already reproduce by logging seeds, code and data hashes."
- **Do.** Add a paragraph in §1 or §4: logging a seed makes one run replayable, but the rule set still depends on an arbitrary choice among ~100 different rule sets. That leaves a degree of freedom (seed shopping: picking the seed whose rules look best) that an auditor cannot detect from the log. Seed-agnostic determinism removes it: there is only one model for given data and settings. Also answer "why ignore `random_state` rather than log a canonical seed chain".

### C3 (reworded). Single convex problem for confirmatory rules
- Formulate as block-penalised logistic regression: λ = 0 on the confirmatory block, elastic net on the rest, sign bounds on the confirmatory coefficients. Write the optimisation problem in the paper; discuss identifiability under collinearity (ridge on the non-confirmatory block). Keep the solver deterministic.

### C1 (extended). Baselines
- Add **EBM** as a baseline. It is cheap because it is already a dependency.
- Keep SIRUS as the primary baseline.
- RPTE, RCProb/RuleCOSI+, Node Harvest and TE2Rules go to related work only after verification (see `notes/REFERENCES_TO_VERIFY.md`).

### B5 (new). Interpretability and runtime metrics
- Per model: number of active rules, mean rule length, **active rules per test case**, rule overlap/conflict rate.
- Runtime breakdown: rule extraction / bootstraps / final CV / refit. Try cheap speed-ups that keep determinism: a smaller C grid, warm starts, `lbfgs` for L2-only stages.

### B6 (new, optional). More datasets
- Add HELOC (FICO) and Give Me Some Credit, or 3–5 OpenML tabular binary tasks, after the speed-ups.

### A7 (new). Promotion checklist for confirmatory rules
- **Why.** A confirmatory rule could encode a protected characteristic indirectly.
- **Do.** Add a short checklist in the README or docs: legal basis, provenance, a proxy screen (correlation with protected attributes when available), expected sign and the reviewer. Store it in the rule's metadata or the audit bundle.

### E1 (new, optional). Tax-audit simulation
- A synthetic task with injected selection bias, showing the full pipeline with inverse-probability weighting. This turns section 7 from a sketch into a demonstration.

### Stop rule for reviews
Three AI reviews now agree on the same gaps. Further reviews will repeat them until new results exist. Request the next review only after B1 (perturbation), C3 (sign constraints) and the ablation are done.
