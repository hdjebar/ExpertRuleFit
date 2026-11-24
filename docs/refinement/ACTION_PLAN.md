# Action plan — ExpertRuleFit refinement

> **P0 — BLOCKING, do first (from review 4, verified 23/11/2026).** A method with the *same name* and the *same core idea* already exists: **Expert RuleFit (ERF)** by Luisa Ebner (MSc thesis), R implementation at github.com/LuisaEbner/ERF. It extends RuleFit with *confirmatory* expert rules (exempt from penalisation) and *optional* expert rules (adaptive penalty factor). Our README, paper and Table 1 present this as new. Before anything else:
> 1. README: add a "Prior work and attribution" section at the top: ExpertRuleFit (Python) builds on Expert RuleFit (Ebner et al.); the confirmatory/optional concept and adaptive penalties are theirs; our additions are listed explicitly (determinism by design and execution envelope, bootstrap frequency filter, standardisation + conditioning fix, sign/absorption diagnostics, audit bundle, EBM-optional discovery).
> 2. Paper: cite Ebner (thesis and R package only), add a row to Table 1 ("Expert RuleFit (Ebner et al.) — rule ensemble — not measured — not stated — yes, penalty factor 0"), rewrite contribution 1 and the abstract so the confirmatory/optional mechanism is credited, not claimed.
> 3. Decide on the name. Options: keep "ExpertRuleFit" with explicit credit (acceptable but confusing), or rename (e.g. "DetERF", "AuditRuleFit"). Ask the author; do not rename on your own.
> 4. Bibliographic details: done 24/11/2026 (FINDINGS §14). Title and "MSc thesis" verified from the ERF README; university and year not stated there, so not given.

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
- **Status (24/11/2026).** The cross-machine part is measured (FINDINGS §20): on three CPU models the rules and AUC were identical and the probability bytes were not. "Same CPU model" is now an explicit condition of the guarantee (paper §3, README; author's decision). **Still to do in phase C:** pin the BLAS kernel (e.g. `OPENBLAS_CORETYPE`, or a reference BLAS build in the image) and retest bitwise identity across CPU models. Then decide whether the condition can be dropped.

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


---

## Additions from review 4 — Google Deep Review (23/11/2026, paper v2)

The most technical review so far. Besides P0 (prior work, top of this file):

### D2. Specify Step 10 (fallback refit) exactly — correctness
How is "refit with C unpenalised" run in scikit-learn, which has no per-feature penalty factor? Global `penalty=None` on all kept columns, or tiny w_c (which brings back the conditioning failure)? State the objective, the support (columns from step 7 or support of step 8 + C), the solver, and the scale on which |β_c| > 1e-10 is checked. Note: under saga, perfectly collinear columns split weight rather than one going to 0, so Definition 1's "collinear → flag false" needs a rank/QR check. **This argues for doing C3 with a coordinate-descent solver that supports per-feature penalty factors (v_c = 0) and box constraints, removing the √w rescaling entirely.**

### D3. Check whether imodels runs its own Lasso in step 1 — possibly major
Does `imodels.RuleFit` with `max_rules=50` select rules through its own L1 fit before our bootstrap filter? If so, the unstable Lasso selection happens upstream (merely frozen by seed 42) and expert rules are not considered when the 50 rules are chosen. Read the code path; if confirmed, extract candidate rules directly (rule-extraction helper, or SIRUS-style occurrence frequency) and say so in the paper.

### D4. Bootstrap with replacement + inner CV = duplicate-row leakage
n-out-of-n bootstrap puts duplicates in both CV train and validation folds → under-regularised fits in the filter. Replace with stratified subsampling without replacement, ideally CPSS (Shah & Samworth: complementary pairs of size n/2). Merge with C2.

### A8. Execution envelope completeness
- `PYTHONHASHSEED=0` in the container; sort candidate rules lexicographically before building Z.
- List all thread variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) and SIMD dispatch control (`OPENBLAS_CORETYPE` / `NPY_DISABLE_CPU_FEATURES`).
- Row order is a hidden seed (CV folds, saga sample order): **canonicalise row and column order inside `fit()`** (sort by key or row hash) and emit an audit manifest (data hash, params, commit, container digest, prediction hash). Supersedes A6.

### D1 (extended). Seed-shopping argument — make it auditor-proof
- Cite X-hacking (Sharma et al., ICML 2025, arXiv:2401.08513 — verify), predictive multiplicity (Marx, Calmon, Ustun, ICML 2020), model multiplicity (Black, Raghavan, Barocas, FAccT 2022).
- Admit the limits: within the envelope, RuleFit with a logged seed is also deterministic; shopping remains possible via row order and hyperparameters unless rows are canonicalised (A8) and hyperparameters pre-registered.
- New experiment (cheap): vary ERF's *internal* base seed over 20–50 values on the fixed split; report rule-set similarity and prediction ambiguity/discrepancy. This shows whether the frequency filter actually reduces arbitrariness or just hides the seed.

### D5. Absorption diagnostics definitions
Rename sklearn's C (inverse regularisation) to avoid clashing with the confirmatory set C; run the counterfactual refits at the final model's exact (C_reg, l1_ratio), not ridge; compute VIF from centred columns (with intercept).

### D6. Notation and pseudo-code
Define m_j piecewise (mean for linear terms, 0 for rule columns); use β^std in the penalty display; write the back-transformation β_j = β̃_j/(s_j√w_j), β_0 = β̃_0 − Σ β_j m_j; Algorithm 1 lines 6/9 use the 1e-10 threshold; spell out the l1_ratio grid and Cs grid; Z̃ notation.

### D7. Experiments table hygiene
- Run Table 2 at tol = 1e-4 (library default) or report both.
- Split "43, 42, 48 terms" into linear / tree rules / expert rules, for ERF and RuleFit (mean ± sd).
- Say "± = sd across 100 seeds on one split".
- Report the default (w_c = 1e-2, w_o = 0.3) row separately in the sensitivity study.
- State whether the Taiwan subsample is stratified; cite HMEQ's source.

### B1 (revised protocol, per review 4)
- 25–50 stratified 90 % subsamples without replacement (not 100 bootstraps), all preprocessing refitted inside each subsample.
- Metrics: Jaccard **and Sørensen–Dice** (SIRUS uses Dice); chance-corrected stability Φ̂ (Nogueira, Sechidis, Brown, JMLR 2018); **semantic coverage stability** (Hungarian matching of rule activation vectors on the test set, to absorb threshold jitter); prediction ambiguity/discrepancy, Spearman between fits.
- Ablation ladder: A0 fixed-seed RuleFit → A1 + expert weights, no filter (≈ Ebner's ERF) → A2 filter, no expert rules → A3 full → A4 full + sign-constrained deduplicated confirmatory block.
- Baselines: SIRUS, L1/L2 logistic on raw features + expert rules; optional scorecard baselines (RiskSLIM / FasterRisk, verify) and HELOC (has monotonicity directions).
- Calibration: Brier, log-loss, ECE (10 quantile bins); prefer Platt (keeps log-odds additivity) over isotonic.

### C3 (refined by review 4)
Sign constraints alone will pin β_c to 0 (violating Definition 1) or to a cosmetic ε when the effect is already carried by collinear terms (German: the ordinal checking-status variable). Before the bootstrap loop, **prune candidate rules that duplicate or strongly overlap a confirmatory rule**, and offer either excluding/residualising the raw column behind it, or a **two-stage offset fit** (fit the sign-constrained confirmatory block first, pass its linear predictor as an offset to the penalised selection of the rest).

### C4 (refined). Speed-ups that keep bit-identity
Parallel bootstraps across processes, aggregated in fixed order (bit-identical); default tol and a 2–3 value l1_ratio grid inside the bootstraps, full grid only in the final fit; deduplicate identical rule columns before step 3; pathwise coordinate descent (glmnet-style) with penalty factors and box constraints (also solves D2/C3).

### E2. Tax-audit section — legal and statistical precision
- **GDPR:** cite CJEU C-634/21 *SCHUFA* (7 Dec 2023), where a score a decision-maker "draws strongly" on can itself fall under Art. 22(1). Human review must be meaningful (WP251rev.01). Tax authorities rely on Art. 22(2)(b) and Art. 6(1)(e) and Art. 23 with national law; frame determinism and confirmatory legal criteria as Art. 22(2)(b) "suitable safeguards".
- **Fairness:** compare false-positive audit rates (unproductive audits on compliant taxpayers) on the random sample, not only score distributions.
- **IPW through Algorithm 1:** weighted moments for m_j and s_j; `sample_weight` to the tree generator and the elastic net; bootstrap stratified by audit source without weighting the draws (avoid v²); stabilise or clip the weights; AIPW. Rank by expected yield per audit hour.
- Positivity: positivity, MNAR (Crook & Banasik 2004, reject inference).

### Verified on 23/11 vs still to verify
- **Verified:** Ebner ERF exists (GitHub LuisaEbner/ERF; master thesis; Nalenz 2021 dissertation acknowledges Ebner as co-author on expert knowledge in rule ensembles).
- **To verify:** the 2021 workshop paper venue, Sharma et al. 2025 X-hacking, Black et al. 2022, Marx et al. 2020, Nogueira et al. 2018, Sechidis et al. 2019, Ustun & Rudin 2019 (RiskSLIM), Liu et al. 2022 (FasterRisk; page numbers differ between two mentions in the review), Chen et al. 2022 (DSS), Henderson et al. 2023 (IRS audits), CJEU C-634/21 (well known; check citation format).
