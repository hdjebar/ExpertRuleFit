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
