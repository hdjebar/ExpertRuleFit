# Experiments

## B1 — stability under data perturbation (`perturbation.py`)

```bash
pip install -e ".[dev]" ucimlrepo matplotlib
python experiments/fetch_data.py          # -> experiments/data/{german,taiwan,hmeq}.csv
python experiments/perturbation.py --jobs 8
```

`fetch_data.py` needs network access to UCI and creditriskanalytics.net. Without it, put the three CSVs in `experiments/data/` yourself: raw features plus a binary `target` column (1 = default/bad).

Outputs in `experiments/results/perturbation/`:

| File | Content |
|---|---|
| `<dataset>/fits.jsonl` | One record per fit: rule set, counts, AUC/Brier/ECE, test-set probabilities. Reruns skip fits already recorded. |
| `<dataset>/selection_freq_<model>_<scheme>.csv` | Per-rule selection frequency across perturbed fits |
| `<dataset>/stability.png` | Pairwise-Jaccard boxplots and selection-frequency curves |
| `stability_summary.csv` | Jaccard (exact rules and coarse signatures), rule counts, rule length, prediction SD |
| `performance_summary.csv` | AUC, Brier, ECE with 95 % t-intervals over repeated splits |
| `tables.md`, `tables.tex` | Paper-ready tables |

Protocol details (seeds, preprocessing inside each training set, Taiwan 5,000-row subsample as in the benchmark, `--taiwan-full` for all rows) are in the script docstring. Each fit is single-threaded; `--jobs` runs fits in parallel processes and does not change any fit's output.

**Rule identity.** "Exact" Jaccard compares rule strings with thresholds, which differ slightly between perturbed fits even when the rule is the same in substance. "Signature" Jaccard drops thresholds and compares features and directions only. ExpertRuleFit's positional prefix (`rule[i]:`) is stripped before comparison.

**Cost.** ExpertRuleFit runs 10 bootstraps × 3-fold elastic-net CV per fit, single-threaded, so it dominates the runtime. `fit_seconds` in `fits.jsonl` gives per-fit timings for planning.
