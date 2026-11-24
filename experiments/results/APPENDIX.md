# Results appendix (cited by the technical note)

Results labelled C3/B2/A6/B3 were produced with ExpertRuleFit 0.7.0 code (bootstrap resampling, filter on, refit by default); Table 1 (0.8.0) and B1 with 0.8.0 code. All results come from the pinned `docker/` image (Python 3.11.18, scikit-learn 1.7.2, NumPy 2.4.6, imodels 3.0.4), single-threaded, on one x86-64 machine. Each ExpertRuleFit configuration was run 3 times in separate processes with identical SHA-256 fingerprints, unless stated otherwise. Expert rules are standard credit-risk criteria chosen by the author.

| Item | What | Files | Command |
|---|---|---|---|
| Table 1 (0.8.1) | Seed sensitivity, 3 datasets, 0.8.1 defaults (canonical columns), on GitHub Actions; RuleFit fits reused from `table1_v080` and checked (`reuse_check_*.json`); FINDINGS §20 | `table1_v081/` (second run on another CPU: `run1_ffe5a57/`) | push to `run/table1-<name>` (`.github/workflows/experiments.yml`) |
| Table 1 (0.8.0) | Seed sensitivity, 3 datasets, library defaults (RuleFit × 100 seeds reused from `table1/`, ExpertRuleFit × 3 runs) | `table1_v080/` (filter-on intermediate run: `table1_v080_filter/`; 0.7.0: `table1/`) | `python experiments/table1.py --out experiments/results/table1_v080` |
| B1 | Stability under 25 stratified 90 % subsamples, German Credit: fixed-seed RuleFit vs ExpertRuleFit with and without the filter | `b1/german.md`, `b1/german_summary.json` | `python experiments/b1_stability.py --dataset german` |
| B3 | Penalty-weight grid w_c × w_o, German Credit | `sensitivity/` | `python experiments/sensitivity.py --datasets german` |
| C3 | Refit vs constrained confirmatory block, German Credit | `ablation/german.md` (rows erf_refit, erf_constrained*) | `python experiments/ablation.py` |
| B2 + C1 | Ablation with fixed-seed RuleFit and EBM baselines, German Credit | `ablation/german.md`, `ablation/german_summary.json` | `python experiments/ablation.py` |
| A6 | Row-order invariance, German Credit | `row_order/german.md` | `python experiments/row_order.py` |

Run each command inside the image: `docker run --rm --network none -v "$PWD":/work -w /work -e PYTHONPATH=/work erf-pinned <command>`. All scripts resume from their results files. HMEQ and Taiwan runs of B3 and B2 are pending: they are slow single-threaded (one ExpertRuleFit fit took up to 3.5 h with tol 1e-6). Use `--datasets hmeq taiwan` for `sensitivity.py`, and `--dataset hmeq` or `--dataset taiwan` for `ablation.py`.

## B2 / C3 — German Credit

| Variant | Runs identical | AUC | Brier | Terms (rules + linear) | Mean rule length | Active rules per case (mean / max) | Confirmatory coef | Sign / bound | Conv. warnings | Fit s | rule gen / bootstrap / final CV / refit-constr. (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| erf_refit | 3/3 | 0.797 | 0.1618 | 47 (36 + 11) | 2.92 | 12.9 / 21 | -1.204 | contradicted | 0 | 233 | 4 / 211 / 18 / 0 |
| erf_constrained | 3/3 | 0.792 | 0.1634 | 46 (36 + 10) | 2.97 | 13.2 / 21 | +0.000 | bound active | 0 | 241 | 4 / 225 / 17 / 0 |
| erf_constrained_ridge | 3/3 | 0.785 | 0.1688 | 50 (37 + 13) | 2.95 | 13.6 / 22 | +0.000 | bound active | 0 | 238 | 3 / 217 / 18 / 0 |
| erf_no_filter | 3/3 | 0.789 | 0.1683 | 61 (42 + 19) | 2.90 | 15.1 / 25 | -1.238 | contradicted | 0 | 38 | 3 / 0 / 35 / 0 |
| erf_no_weighting | 3/3 | 0.784 | 0.1706 | 50 (39 + 11) | 2.90 | 13.8 / 22 | -1.545 | contradicted | 0 | 121 | 3 / 110 / 8 / 0 |
| erf_no_enforcement | 3/3 | 0.797 | 0.1618 | 47 (36 + 11) | 2.92 | 12.9 / 21 | -1.204 | contradicted | 0 | 233 | 3 / 225 / 18 / 0 |
| erf_no_expert_rules | 3/3 | 0.793 | 0.1586 | 43 (36 + 7) | 3.06 | 12.8 / 21 | – | – | 0 | 103 | 3 / 94 / 6 / 0 |
| rulefit_fixed_seed | – | 0.800 | 0.1687 | 40 (40 + 0) | 3.00 | 14.6 / 23 | – | – | 0 | 3 | – |
| ebm | – | 0.803 | 0.1543 | 80 (0 + 0) | – | 80.0 / 80 | – | – | 0 | 19 | – |

## A6 — row order, German Credit

| Mode | Permutation | Rules | Jaccard vs original | Max abs prob. diff | AUC | Fingerprint equal |
|---|---|---|---|---|---|---|
| as_given | original | 44 | 1.000 | 0.00e+00 | 0.789 | True |
| as_given | 0 | 48 | 0.082 | 6.48e-01 | 0.770 | False |
| as_given | 1 | 44 | 0.048 | 6.18e-01 | 0.795 | False |
| canonical | original | 43 | 1.000 | 0.00e+00 | 0.792 | True |
| canonical | 0 | 43 | 1.000 | 0.00e+00 | 0.792 | True |
| canonical | 1 | 43 | 1.000 | 0.00e+00 | 0.792 | True |

## B3 — German Credit

| Dataset | w_c | w_o | AUC | Brier | Active | Jaccard vs default | Conf. coef | Sign OK | Optional kept | Displaced | Final C | saga iters (median) | ConvergenceWarnings | Fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| german | 1e-08 | 0.1 | 0.753 | 0.2289 | 48 | 0.73 | +0.064 | True | True | 0 | 0.359 | 22 | 361 | 1026 |
| german | 1e-06 | 0.1 | 0.797 | 0.1538 | 62 | 0.95 | +0.248 | True | True | 0 | 2.78 | 123 | 69 | 947 |
| german | 0.0001 | 0.1 | 0.785 | 0.1714 | 58 | 0.95 | -1.650 | False | True | 1 | 2.78 | 259 | 6 | 822 |
| german | 0.01 | 0.1 | 0.795 | 0.1629 | 48 | 0.78 | -1.248 | False | True | 1 | 0.359 | 61 | 0 | 266 |
| german | 1 | 0.1 | 0.793 | 0.1582 | 48 | 0.78 | -0.302 | False | True | 1 | 0.0464 | 49 | 0 | 142 |
| german | 1e-08 | 0.3 | 0.756 | 0.2284 | 55 | 0.84 | +0.067 | True | True | 0 | 0.0464 | 22 | 360 | 1043 |
| german | 1e-06 | 0.3 | 0.795 | 0.1542 | 62 | 0.95 | +0.250 | True | True | 0 | 2.78 | 120 | 66 | 943 |
| german | 0.0001 | 0.3 | 0.786 | 0.1705 | 59 | 1.00 | -1.628 | False | True | 1 | 2.78 | 285 | 3 | 792 |
| german | 0.01 | 0.3 | 0.797 | 0.1618 | 47 | 0.77 | -1.204 | False | True | 1 | 0.359 | 60 | 0 | 274 |
| german | 1 | 0.3 | 0.791 | 0.1587 | 46 | 0.78 | -0.307 | False | True | 1 | 0.0464 | 34 | 0 | 133 |
| german | 1e-08 | 1 | 0.756 | 0.2284 | 54 | 0.82 | +0.067 | True | True | 0 | 0.0464 | 22 | 360 | 1045 |
| german | 1e-06 | 1 | 0.795 | 0.1547 | 62 | 0.95 | +0.268 | True | True | 0 | 0.359 | 120 | 66 | 942 |
| german | 0.0001 | 1 | 0.787 | 0.1694 | 58 | 0.92 | -1.413 | False | True | 0 | 2.78 | 295 | 3 | 865 |
| german | 0.01 | 1 | 0.793 | 0.1579 | 44 | 0.75 | -0.624 | False | True | 1 | 0.0464 | 52 | 0 | 260 |
| german | 1 | 1 | 0.784 | 0.1706 | 50 | 0.79 | -1.545 | False | True | 1 | 2.78 | 42 | 0 | 132 |

## Table 1

| Dataset | RuleFit: identical rule sets | RuleFit: distinct sets | RuleFit: mean Jaccard | ERF: identical runs | RuleFit AUC (mean ± sd) | ERF AUC | ERF rules |
|---|---|---|---|---|---|---|---|
| German Credit | 1/100 | 100 | 0.045 | 3/3 | 0.788 ± 0.009 | 0.790 | 43 |
| Taiwan Credit | 1/100 | 100 | 0.058 | 3/3 | 0.764 ± 0.003 | 0.769 | 42 |
| HMEQ | 1/100 | 100 | 0.036 | 3/3 | 0.903 ± 0.003 | 0.903 | 48 |

Environment: expertrulefit 0.7.0 (/work/expertrulefit), python 3.11.18, numpy 2.4.6, scikit-learn 1.7.2, imodels 3.0.4, platform Linux-6.18.44-fc-v726-x86_64-with-glibc2.41

## B1 — German Credit (0.8.0 code)

| Arm | Fits | Jaccard (sd) | Dice | Semantic (sd) | Pred. SD | Ambiguity | Discrepancy | AUC mean (sd) | AUC full | Rules + linear | Fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rulefit_fixed_seed | 25 | 0.014 (0.006) | 0.028 | 0.530 (0.035) | 0.0563 | 0.210 | 0.095 | 0.785 (0.011) | 0.803 | 43.8 + 1.8 | 3 |
| erf_default | 25 | 0.003 (0.006) | 0.005 | 0.414 (0.076) | 0.0872 | 0.395 | 0.145 | 0.775 (0.016) | 0.770 | 29.0 + 1.1 | 190 |
| erf_no_filter | 25 | 0.025 (0.025) | 0.048 | 0.527 (0.041) | 0.0792 | 0.375 | 0.130 | 0.784 (0.013) | 0.791 | 41.5 + 6.5 | 24 |

## Table 1 — 0.8.0 defaults

| Dataset | RuleFit: identical rule sets | RuleFit: distinct sets | RuleFit: mean Jaccard | ERF: identical runs | RuleFit AUC (mean ± sd) | ERF AUC | ERF rules |
|---|---|---|---|---|---|---|---|
| German Credit | 1/100 | 100 | 0.045 | 3/3 | 0.788 ± 0.009 | 0.794 | 51 |
| Taiwan Credit | 1/100 | 100 | 0.058 | 3/3 | 0.764 ± 0.003 | 0.757 | 28 |
| HMEQ | 1/100 | 100 | 0.036 | 3/3 | 0.903 ± 0.003 | 0.902 | 37 |

Environment: expertrulefit 0.8.0 (/work/expertrulefit), python 3.11.18, numpy 2.4.6, scikit-learn 1.7.2, imodels 3.0.4, platform Linux-6.18.44-fc-v726-x86_64-with-glibc2.41
