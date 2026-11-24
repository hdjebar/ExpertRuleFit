# Determinism hash-check, 0.8.1 paths (24/11/2026, pinned image erf-pinned)

Each configuration was run 3 times, each in a fresh `docker run --network none` process, with `fp.py` (this directory): 400 synthetic rows, 4 features, one confirmatory rule found by name, `n_estimators=60, max_rules=20`. The first column is the column order given to `fit`; with `column_order="canonical"` (default) every order must give the same hashes. `x0..x3` are already sorted, so the first row also equals the 0.8.0 output of the same script on `main` (`aeb7a371…`).

| Columns given | Options | Runs identical | coefficients_sha256 | predictions_sha256 |
|---|---|---|---|---|
| `x0,x1,x2,x3` | `{}` | 3/3 | `aeb7a371e145fbb3…` | `e245411118980f74…` |
| `x3,x1,x0,x2` | `{}` | 3/3 | `aeb7a371e145fbb3…` | `e245411118980f74…` |
| `x0,x1,x2,x3` | `{"calibration":"sigmoid"}` | 3/3 | `aeb7a371e145fbb3…` | `547d4c002d5e85ac…` |
| `x2,x0,x3,x1` | `{"calibration":"sigmoid"}` | 3/3 | `aeb7a371e145fbb3…` | `547d4c002d5e85ac…` |
| `x0,x1,x2,x3` | `{"candidates":"trees"}` | 3/3 | `a2f04c9af6e0cb91…` | `75ec767c5161e303…` |
| `x3,x2,x1,x0` | `{"candidates":"trees"}` | 3/3 | `a2f04c9af6e0cb91…` | `75ec767c5161e303…` |

Reproduce: `docker run --rm --network none -v "$PWD":/work -w /work -e PYTHONPATH=/work erf-pinned python experiments/results/hashcheck_v081/fp.py x3,x1,x0,x2 '{}'`.
