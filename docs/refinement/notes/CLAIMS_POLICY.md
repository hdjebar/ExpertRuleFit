# Claims policy — what can be said until new results exist

## P0 — prior work (23/11/2026, binding)
- The confirmatory/optional expert-rule mechanism, its terminology and the penalty exemption for confirmatory rules come from **Expert RuleFit** (Luisa Ebner, MSc thesis *Expert RuleFit – Complementing Rule Ensembles with Expert Knowledge*; R package github.com/LuisaEbner/ERF). Cite only the thesis and the R package. Any public text that describes "keeping the expert's rules" must say the library builds on Expert RuleFit (Ebner) and must not present that mechanism as our own.
- Our own contributions, as they may be described: determinism by design with canonical row order; the sign-constrained confirmatory block; diagnostics and SHA-256 fingerprints; the conditioning analysis of penalty weighting by column rescaling.
- The library name is unchanged pending the author's decision.

## Allowed now (supported by the code and the 100-seed benchmark)
- Identical rules and predictions on every run within the execution envelope (determinism, by design).
- Standard RuleFit returns the same rule set in 1 run out of 100 across seeds.
- Test AUC versus RuleFit: use only the figure from the 0.8.0 Table 1 (`experiments/results/table1_v080/`, see FINDINGS §18): within 0.007 of RuleFit's mean. The 0.7.0 figure (0.005) and the pre-fix figure (0.013) are retired. **Once 0.8.1 is released** (PR for decisions c/d merged), use the 0.8.1 Table 1 figure instead: within 0.010 of RuleFit's mean (`experiments/results/table1_v081/`, FINDINGS §20).
- RuleFit returned a different rule set for each of 100 seeds on all three datasets; ExpertRuleFit returned identical rules and probabilities in 3 of 3 runs (hash-checked).
- Confirmatory rules are always kept in the model. A rule without a declared sign gets a non-zero coefficient unless degenerate. In constrained mode (default since 0.8.0), a rule whose declared sign the data contradict is held at zero and flagged; it is then not "included" under Definition 1.
- **Allowed for constrained mode only (decided 23/11/2026):** "respects the sign declared by the expert; when the data contradict it, the coefficient is held at zero and flagged." 
- 124 automated tests (pinned stack, 0.8.1, 24/11/2026; 104 at 0.8.0); open source. Public texts still saying "39 tests" are out of date but not wrong at the time written.

## Not allowed until Phase B/C results exist
- "Stable" or "stability" without the qualifier "across seeds".
- "Stable under data changes / retraining". B1 (German, FINDINGS §17) found ExpertRuleFit *not* more stable than fixed-seed RuleFit; do not claim a stability advantage.
- "Calibrated probabilities" (README currently says this — change to "logistic probabilities; optional post-hoc calibration").
- "Production-ready", "guaranteed reproducibility" without the execution envelope.
- "Identical on any machine": bitwise identity of the probabilities was shown per machine only; across CPU models the rules and AUC matched but the probability bytes did not (FINDINGS §20).
- "Respects expert intent" in general; only the constrained-mode wording above is allowed, and never for refit mode.
- Any claim on tax data or audit performance.

## Public wordings currently in use (keep consistent)
- **CV (FR, 22/11/2026):** « elle donne des règles et des scores identiques à chaque exécution (déterminisme), là où RuleFit standard change de règles d'une graine à l'autre (1 exécution sur 100 identique), avec une AUC comparable ; elle conserve les règles métier imposées par l'expert. 39 tests automatisés. »
- **Cover letter (FR):** « une bibliothèque Python dont les scores se reproduisent à l'identique et qui conserve les règles posées par l'expert métier ».
- **Interview line (FR):** « Le déterminisme est garanti par conception ; la stabilité quand on rééchantillonne les données est l'expérience suivante, et je l'ai décrite dans la note technique. »

If Phase B shows good perturbation stability, update the CV line only after the results are committed to the repo.

**Action for the author (P0):** the CV and cover-letter lines above describe keeping the expert's rules without crediting Expert RuleFit. Before the next submission, add a credit (e.g. « en s'appuyant sur Expert RuleFit (L. Ebner, mémoire de MSc ; package R ERF) ») or reword around our own contributions (déterminisme, contraintes de signe, empreintes d'audit).

## About the AI review
paperreview.ai produced an AI-generated review. Do not present it as peer review or "external review". Accurate phrasing: "I had the method critically reviewed; the main gap is stability under perturbation, and here is the protocol / the results."
