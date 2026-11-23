# Claims policy — what can be said until new results exist

## Allowed now (supported by the code and the 100-seed benchmark)
- Identical rules and predictions on every run within the execution envelope (determinism, by design).
- Standard RuleFit returns the same rule set in 1 run out of 100 across seeds.
- Test AUC within 0.005 of RuleFit's mean on three credit datasets (single split; Table 1 rerun 23/11/2026 on the pinned stack, preprocessing on the training split). The earlier figure, 0.013, came from the pre-fix protocol: retire it.
- RuleFit returned a different rule set for each of 100 seeds on all three datasets; ExpertRuleFit returned identical rules and probabilities in 3 of 3 runs (hash-checked).
- Confirmatory rules are always included in the final model.
- 66 automated tests (pinned stack, 23/11/2026); open source. Public texts still saying "39 tests" are out of date but not wrong at the time written.

## Not allowed until Phase B/C results exist
- "Stable" or "stability" without the qualifier "across seeds".
- "Stable under data changes / retraining".
- "Calibrated probabilities" (README currently says this — change to "logistic probabilities; optional post-hoc calibration").
- "Production-ready", "guaranteed reproducibility" without the execution envelope.
- "Respects expert intent" (no sign constraint yet).
- Any claim on tax data or audit performance.

## Public wordings currently in use (keep consistent)
- **CV (FR, 22/11/2026):** « elle donne des règles et des scores identiques à chaque exécution (déterminisme), là où RuleFit standard change de règles d'une graine à l'autre (1 exécution sur 100 identique), avec une AUC comparable ; elle conserve les règles métier imposées par l'expert. 39 tests automatisés. »
- **Cover letter (FR):** « une bibliothèque Python dont les scores se reproduisent à l'identique et qui conserve les règles posées par l'expert métier ».
- **Interview line (FR):** « Le déterminisme est garanti par conception ; la stabilité quand on rééchantillonne les données est l'expérience suivante, et je l'ai décrite dans la note technique. »

If Phase B shows good perturbation stability, update the CV line only after the results are committed to the repo.

## About the AI review
paperreview.ai produced an AI-generated review. Do not present it as peer review or "external review". Accurate phrasing: "I had the method critically reviewed; the main gap is stability under perturbation, and here is the protocol / the results."
