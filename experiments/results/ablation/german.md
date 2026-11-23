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
