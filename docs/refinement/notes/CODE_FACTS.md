# Verified code facts (repo `hdjebar/ExpertRuleFit`, `main` @ 76f850f, checked 22/11/2026)

## Layout
- `expertrulefit/expert_rulefit.py` (1,347 lines) — main estimator.
- `expertrulefit/dual_model.py` (399), `dual_glass_box.py` (851) — EBM + ERF stacking.
- `expertrulefit/ebm_bridge.py` (246) — `discover_interaction_rules`.
- `expertrulefit/rule_spec.py` (224) — `RuleSpec` dataclass, save/load.
- `expertrulefit_validation.py` (897) — the 100-seed benchmark.
- Tests: `tests/test_expert_rulefit.py` (17), `test_validation.py` (15), `test_dual_model.py` (4), `test_ebm_bridge.py` (3) = **39**.
- Docs: `docs/SOTA.md` (literature survey, Feb 2026), `docs/REPRODUCIBILITY_CONTAINERS.md`, `CODE_REVIEW.md` (internal review, all critical items resolved).

## Defaults (`ExpertRuleFit.__init__`)
| Parameter | Default |
|---|---|
| `n_estimators` | 250 |
| `tree_size` | 4 |
| `max_rules` | 50 |
| `random_state` | 42, **ignored** — `_BASE_SEED = 42` always used; `UserWarning` if another value is passed |
| `n_bootstrap` | 10 (bootstrap seeds `_BASE_SEED + b`) |
| `rule_threshold` | 0.8 |
| `confirmatory_penalty` | 1e-8 |
| `optional_penalty` | 0.3 |
| `l1_ratios` | [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] |
| `tol` | 1e-4 (saga) |
| `calibration` | None (options: "sigmoid", "isotonic"; 5-fold `CalibratedClassifierCV`) |

## Pipeline
1. imodels RuleFit with fixed seed → candidate rules + linear terms.
2. Append confirmatory / optional rules; weights w_j.
3. Columns rescaled X_j / sqrt(w_j) (L1 ∝ sqrt(w_j), L2 ∝ w_j).
4. 10 bootstraps × `LogisticRegressionCV(elasticnet, saga, cv=3)`.
5. Keep features non-zero in ≥ 80 % of bootstraps; confirmatory force-included.
6. Final `LogisticRegressionCV` on kept features.
7. If a confirmatory coefficient is zero → unpenalised refit; `confirmatory_all_active_` flag.
- **No sign or monotonicity constraint anywhere.**
- `RuntimeError` if all bootstraps fail; `UserWarning` if some fail.

## Validation script protocol
- One stratified 80/20 split, `random_state=42`.
- 100 runs with `random_state = 0..99` on that split, for RuleFit and ERF.
- Rule set = rules with non-zero coefficient; stability = count of runs sharing the modal rule set; Jaccard averaged over sampled pairs.
- ERF runs sequentially (comment: fixed internal seeds). **Because ERF ignores `random_state`, its 100/100 is by construction.**

## Reported results (README)
| Dataset | RuleFit stability | ERF stability | RuleFit AUC | ERF AUC |
|---|---|---|---|---|
| German Credit | 1/100 | 100/100 | 0.788 | 0.801 |
| Taiwan Credit | 1/100 | 100/100 | 0.764 | 0.763 |
| HMEQ | 1/100 | 100/100 | 0.903 | 0.903 |

RuleFit mean Jaccard ≈ 0.05; ERF 1.000; ERF AUC std 0.

## Data access
The three datasets come from UCI / public mirrors. The Claude cloud workspace usually blocks non-allowlisted downloads: check first, or have the user attach the CSVs, or run in Claude Code on a machine with internet.
