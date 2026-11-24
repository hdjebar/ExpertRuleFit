# Session handoff — ExpertRuleFit refinement

**Written:** Tue 24/11/2026, about 02:00 (Europe/Luxembourg). Replaces the handoff of 24/11 00:15.

## Repository state

| Item | Value |
|---|---|
| `main` | `a432213`: PR #19 (Ebner/Nalenz citations) on top of PR #16 (`ec79250`) and PR #17 (CI fix, `0d997f8`). Version 0.8.0. CI green (lint, Python 3.10–3.12). |
| Tag `v0.8.0` | On `ec79250`, pushed by the author (sessions cannot push tags: HTTP 403). |
| Open PR | #18 `feat/columns-calibration` (draft, up to date with `main`): 0.8.1, decisions (c) and (d), the "same CPU model" condition, the experiments workflow, and the Table 1 0.8.1 results. Stays a draft until B1 and decision (b) are in (author's decision). |
| Experiment branches | `run/b1-v081` and `run/table1-v081` start `.github/workflows/experiments.yml` on push. Results are committed by the workflow to `results/b1-v081` (pending) and `results/table1-v081` (done, `c88f961`). |
| Merged branches | `fix/ebner-citation`, `fix/ci`, `docs/nalenz-citation` and `refinement/phase-a` can be deleted; so can `run/source-nalenz` if it exists (it was never pushed). |
| Tests | 124 passed on the pinned image at PR #18 (104 on `main`). |

## Done in this session

1. **Ebner citation** (PR #16):
   - The paper, README, CLAIMS_POLICY, FINDINGS §14 and ACTION_PLAN cite only L. Ebner's MSc thesis *Expert RuleFit – Complementing Rule Ensembles with Expert Knowledge* and the R package github.com/LuisaEbner/ERF.
   - The title and "MSc thesis" were read from the ERF README (branch `master`). University, supervisors and year are not stated there, and the university repositories were unreachable, so they are not given ("n.d.").
   - The archived text of review 4 is unchanged.
2. **CI** (PR #17):
   - `ubuntu-24.04` runner.
   - Python 3.9 dropped and `requires-python >= 3.10`: current imodels cannot be imported on 3.9.
   - `six` declared as a dependency (imodels imports it without declaring it).
   - ruff pinned to 0.17.6 with an explicit rule set, then `ruff format` and the safe fixes. Fingerprints unchanged.
3. **Decisions (a)–(d)**, taken by the author on 24/11 and recorded in FINDINGS §19:
   - **(a)** Keep the name.
   - **(b)** Keep `candidates="rulefit"`; the B1 `erf_trees` arm decides.
   - **(c)** `column_order="canonical"` by default. Columns are sorted by `str(name)`, recorded in the audit bundle and in `fingerprint()["schema_sha256"]`, and a permutation test is included. Default outputs changed, hence **0.8.1**.
   - **(d)** `calibration="sigmoid"` allowed in constrained mode: Platt scaling on out-of-fold log-odds of the same block, a > 0, so the log-odds stay additive. Isotonic still raises.
4. **Experiments on GitHub Actions** (`.github/workflows/experiments.yml`):
   - The data job fetches each dataset once, with retries (the HMEQ host returned 429), and records SHA-256 sums.
   - One job per (dataset, subsample index), in the pinned image with `--network none`.
   - Results are committed to `results/<run>`.
5. **Table 1 under 0.8.1** (FINDINGS §20, `experiments/results/table1_v081/`):
   - ERF AUC 0.798 / 0.763 / 0.904 (German / Taiwan / HMEQ) against RuleFit means of 0.788 / 0.764 / 0.903.
   - 3/3 identical runs on every dataset; 49 / 38 / 33 terms.
6. **Determinism across machines** (FINDINGS §20):
   - Within one machine: bitwise identical.
   - Across three CPU models (Intel Xeon, AMD EPYC 9V45, EPYC 7763): rule sets identical and AUC bit-identical, but the probability bytes differ.
   - The reused 0.8.0 RuleFit fits are valid (same rules and AUC on the new machine).

## Key numbers (source files)

- Table 1 0.8.1: `experiments/results/table1_v081/table1.json` (+ `reuse_check_*.json`, `run1_ffe5a57/`).
- Table 1 0.8.0: `experiments/results/table1_v080/table1.json`.
- B1 German, 0.8.0: `experiments/results/b1/german_summary.json`. Semantic similarity 0.530 / 0.414 / 0.527 and prediction SD 0.056 / 0.087 / 0.079, for fixed-seed RuleFit / filter on / no filter.
- Hash-check of the 0.8.1 paths: `experiments/results/hashcheck_v081/README.md`.

## Pending, in order

1. **Citations: done** (PR #19, FINDINGS §14). The sources cited are:
   - Ebner (2021), master's thesis, Faculty of Science, VU Amsterdam (Nalenz 2021, p. 129, ref. 6);
   - Ebner et al. (2021), 19th International Conference on Artificial Intelligence in Medicine, KR4HC Workshop (as printed in Nalenz 2021, p. XI; reprint pp. 117–130; no LNCS volume);
   - Nalenz (2021), LMU dissertation (p. XII, §3.1.3), read from the PDF supplied by the author;
   - the ERF R package.
2. **B1 under 0.8.1** (`run/b1-v081`, Actions run 45302909823, source `9a2321d`):
   - Covers German, HMEQ and Taiwan; arms `rulefit_fixed_seed`, `erf_default` and `erf_trees`; 25 subsamples plus the full fit.
   - German jobs took about 8 min. HMEQ/Taiwan `erf_trees` fits may take hours (job limit 350 min).
   - When `results/b1-v081` appears:
     - copy it into PR #18 (`experiments/results/b1_v081/`);
     - write FINDINGS §21 and settle decision (b): switch only if `erf_trees` is more stable at similar AUC;
     - update the README B1 bullet and APPENDIX.
   - If jobs time out, rerun with `chunk=1` on fewer arms, or split `erf_trees` into its own run.
3. **SIRUS baseline** in B1 (same protocol), run on Actions too.
4. **Paper** (≤ 7 pages):
   - Table 1 → 0.8.1 numbers.
   - Section 4: column order is now canonical, and identity is per machine (replaces "not tested").
   - B1 on 3 datasets, SIRUS, decisions (b)–(d).
   - Check every number against the result files.
   - Then mark PR #18 ready and merge, and tag `v0.8.1`.
5. **No new AI review** until B1 (3 datasets) and SIRUS are in.
6. **Later:**
   - B2/B3 on HMEQ/Taiwan;
   - overlapping tree rules vs confirmatory rules (ACTION_PLAN review 4);
   - C2 (CPSS);
   - optional: test `OPENBLAS_CORETYPE` pinning for cross-CPU bitwise identity;
   - the author's P0 action: credit Ebner in the CV and cover-letter lines (CLAIMS_POLICY).

## Key files

- Findings log: `docs/refinement/notes/FINDINGS_2026-11-22.md` (§1–20)
- Claims allowed / forbidden: `docs/refinement/notes/CLAIMS_POLICY.md`
- Plan: `docs/refinement/ACTION_PLAN.md`
- Reviews: `docs/refinement/review/`
- Results index and commands: `experiments/results/APPENDIX.md`
- Paper: `docs/refinement/paper/ExpertRuleFit_paper.tex` (+ `.pdf`; `.md` mirror via `pandoc ExpertRuleFit_paper.tex -t gfm -o ExpertRuleFit_paper.md`; build needs `texlive-latex-extra` and `texlive-science`)
- Code: `expertrulefit/expert_rulefit.py`, `expertrulefit/constrained.py`
- Experiments:
  - `experiments/table1.py`, `b1_stability.py` (`--k-start/--k-stop`), `check_reuse.py`, `ablation.py`, `sensitivity.py`, `row_order.py`, `fetch_data.py`;
  - workflow `.github/workflows/experiments.yml`.
- Pinned stack: `docker/Dockerfile`, `docker/requirements.lock.txt`

## Known pitfalls

- **Chat-session machine:**
  - 4 vCPU, 15 GB RAM. Docker is available after `nohup dockerd &`.
  - **No data access**: UCI, creditriskanalytics.net, OpenML and Hugging Face are blocked by the network policy.
  - A **container restart killed background jobs** mid-session (files and the docker image survived).
  - Run long work on Actions only.
- **Tags cannot be pushed** from the session (HTTP 403); branches can.
- **Pinned image:**
  - Build from the repo root: `docker build -f docker/Dockerfile --secret id=ca,src=/root/.ccr/ca-bundle.crt -t erf-pinned .` (the CA secret is only needed behind the session proxy).
  - Run: `docker run --rm --network none -v "$PWD":/work -w /work -e PYTHONPATH=/work erf-pinned <command>`.
  - A full test run takes about 16–20 min.
  - An image built from `main` reports the installed version from its egg-info; the code actually run comes from the mounted checkout.
- **Determinism:** bitwise per machine only (FINDINGS §20). Hash-check every new path in 3 runs; run the full suite before each commit.
- **Experiments workflow:** `workflow_dispatch` only works once the file is on `main`; until then, push to `run/<name>` (`table1` in the name selects Table 1). A new push to the same `run/` branch cancels the running run.
- **imodels** has no `__version__`; use `importlib.metadata`.
- Python 3.11 f-strings: no reuse of the same quote type inside `{...}`.
- **Paper limit: 7 pages** including references.
- **Claims:** nothing public beyond CLAIMS_POLICY.

## Resume message (paste into a new session)

```
Resume the ExpertRuleFit refinement (repo hdjebar/ExpertRuleFit).
First read docs/refinement/SESSION_HANDOFF.md (on branch
feat/columns-calibration, PR #18), then CLAIMS_POLICY.md and
FINDINGS_2026-11-22.md §17–20. Check whether results/b1-v081 exists
(Actions run of experiments.yml on run/b1-v081); if so, do "Pending"
item 2 (B1 0.8.1 write-up, decision b), then item 3 (SIRUS on Actions)
and item 4 (paper). Keep the standing rules: hash-check new paths in 3
runs, full test suite in the pinned image before each commit, paper <= 7
pages, no claims beyond CLAIMS_POLICY, long jobs on GitHub Actions, no
new AI review until B1 (3 datasets) and SIRUS are done.
```
