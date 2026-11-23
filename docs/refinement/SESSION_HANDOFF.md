# Session handoff — ExpertRuleFit refinement

**Written:** Tue 24/11/2026, 00:15 (Europe/Luxembourg).

## Repository state

| Item | Value |
|---|---|
| Repo | `hdjebar/ExpertRuleFit` |
| `main` | `9052604` — merge of PR #15 (merged 23/11/2026 23:49) |
| `refinement/phase-a` | `8dd35c5` — fully contained in `main`; can be deleted |
| `claude/review-codebase-yYi5P` | `86b0ace` — older branch, not from this work; untouched |
| Open PRs | none (PR #15 merged) |
| Tags | none (v0.8.0 not tagged yet) |
| Package version | 0.8.0 (`expertrulefit/_version.py`) |
| Tests | 104 passed on the pinned image at `4588f20` |
| CI on `main` | **failing**. `ruff format --check` fails, and Python 3.9 is not available on the Ubuntu 26.04 runner, which cancels the other test jobs. It already failed before this work (runs of 22/02/2026). See pending item 3. |
| Uncommitted work | none (only LaTeX build files, which are not tracked) |

## Done in this work (PR #15, all on `main`)

Hashes are those on GitHub.

- **Phase A and the standardisation fix (0.7.0):**
  - A1: EBM rules are optional by default.
  - A2: sign check and absorption diagnostics.
  - Internal standardisation; `confirmatory_penalty` default 1e-2. The cause was saga's step size, which shrinks with tiny weights and lets it stop early without a warning (FINDINGS §1–8).
  - Benchmark scaler fitted on the training split only.
  - Pinned `docker/` stack.
  - B3 weight grid on German.
- **Review 3:**
  - `7b5a81e` pack sync
  - `3ab6ce3` D1 (seed shopping, paper §4)
  - `53331c2` A5 (leakage: per-fold rule discovery in DualModel)
  - `8ba5d2e` A6 (fingerprints; row order shown to act as a hidden seed)
  - `ad21d89` C3 (convex constrained confirmatory block, FISTA)
  - `fd3edee` B2 ablation and EBM results
  - `247bf2f` paper
- **Review 4 and decisions 1–5 (0.8.0):**
  - `d5d5fc7` pack sync with review 4
  - `d3d4d64` P0: Expert RuleFit (Ebner) credited in README and paper
  - `4588f20` 0.8.0 code:
    - canonical rows, constrained block and no filter by default;
    - `strict=True` raises `ConfirmatoryBoundError`;
    - D3: `candidates="trees"` option;
    - D4: stratified complementary half-samples;
    - 104 tests.
  - `dca6208` B1 (German) and Table 1 rerun with 0.8.0 defaults; FINDINGS §14–18; APPENDIX
  - `17cccb0` README brought within CLAIMS_POLICY
  - `8dd35c5` paper updated to 0.8.0, 7 pages

### Key numbers (source files)

- **Table 1, 0.8.0** (`experiments/results/table1_v080/table1.json`). ERF runs were identical in 3 of 3 runs on all datasets.

  | Dataset | ERF AUC | RuleFit AUC | ERF terms | ERF fit time |
  |---|---|---|---|---|
  | German | 0.794 | 0.788 ± 0.009 | 51 | 24 s |
  | Taiwan (5,000 rows) | 0.757 | 0.764 ± 0.003 | 28 | 1.6 min |
  | HMEQ | 0.902 | 0.903 ± 0.003 | 37 | 6.6 min |

  Max AUC gap 0.007. RuleFit gave 100 distinct rule sets per dataset.

- **B1, German** (`experiments/results/b1/german_summary.json`).

  | Arm | Semantic similarity | Prediction SD | Ambiguity | AUC |
  |---|---|---|---|---|
  | Fixed-seed RuleFit | 0.530 | 0.056 | 0.21 | 0.785 |
  | ERF, filter on | 0.414 | 0.087 | 0.40 | 0.775 |
  | ERF, no filter | 0.527 | 0.079 | 0.38 | 0.784 |

  So the filter is off by default, and **no stability-advantage claim** may be made.

## Decisions

| Topic | State |
|---|---|
| Row order | **Decided:** `row_order="canonical"` (default) |
| Confirmatory mode | **Decided:** `confirmatory_mode="constrained"` (default). Sign bounds only where `expected_sign` is declared; `"refit"` kept. Bound-active → warning + audit flag; `strict=True` raises. |
| Selection filter | **Decided:** `selection="none"` (default), per B1; `"bootstrap"` keeps it |
| Resampling (when the filter is on) | **Decided:** `resampling="subsample"` |
| (a) Library name | **Open** (P0: name clash with Ebner's Expert RuleFit) |
| (b) `candidates` default | **Open:** `"rulefit"` (current; imodels' own L1 selection, D3) vs `"trees"` |
| (c) Column order | **Open** (A8): a different column order changed the fit (AUC 0.770 vs 0.794). Canonicalise columns by name, or require a fixed schema? |
| (d) Calibration in constrained mode | **Open:** it currently raises `ValueError`. Allow it, or keep it refit-only? |

## Pending, in order — phase B prompt (next block of work)

1. **Ebner citation fix.**
   - Cite only Luisa Ebner's **thesis** and the **R package** (github.com/LuisaEbner/ERF).
   - **Remove every mention of KR4HC / AIME 2021 / LNCS** from the paper's `ebner2021` bibitem, README "Prior work and attribution", CLAIMS_POLICY (P0 box), FINDINGS §14, the paper's Table 1 row, and the PR text if reused.
   - Verify the thesis details (title, degree, university, year) from a source you actually open before writing them; do not guess them.
2. **Tag `v0.8.0`** on `main` after item 1 is merged (annotated tag, pushed).
3. **CI fix** (small; needed before a release looks credible). Drop Python 3.9 from the matrix (or pin `runs-on: ubuntu-24.04`), run `ruff format .` and `ruff check --fix .`, and confirm the tests pass on CI. Note that CI runs unpinned scikit-learn; the authoritative numbers come from the pinned image.
4. **Decisions (a)–(d)** above: ask the author, then implement (c) and (d) with tests (hash-check every new path in 3 runs).
5. **B1 on HMEQ and Taiwan via GitHub Actions.** Add a `workflow_dispatch` workflow that:
   - builds the pinned `docker/` image;
   - runs `python experiments/b1_stability.py --dataset hmeq` (then `taiwan`);
   - uploads `experiments/results/b1/` as an artifact.

   **No long jobs in a chat session.** Commit the downloaded results, then update FINDINGS (§17), APPENDIX, README and the paper.
6. **SIRUS baseline** in B1 (same protocol; R `sirus` or a Python port, run in CI too).
7. **Paper:** add the HMEQ/Taiwan/SIRUS results and the decisions, stay **≤ 7 pages**, and check every number against the result files. Request no new AI review until these are in.
8. **Later:**
   - B2/B3 on HMEQ/Taiwan;
   - the ACTION_PLAN review-4 item on overlapping tree rules vs confirmatory rules (prune, or a two-stage offset fit);
   - C2 (CPSS with error control) if the filter is revisited;
   - the author's action: credit Ebner in the CV and cover-letter lines (CLAIMS_POLICY).

## Key files

- Findings log: `docs/refinement/notes/FINDINGS_2026-11-22.md` (§1–18)
- Claims allowed / forbidden: `docs/refinement/notes/CLAIMS_POLICY.md`
- Plan: `docs/refinement/ACTION_PLAN.md` (phases A–D, plus additions from reviews 2–4)
- Reviews: `docs/refinement/review/` (paperreview.ai, review 2, review 3, review 4 = Google Deep Review with the P0 box)
- Results index and commands: `experiments/results/APPENDIX.md`
- Paper: `docs/refinement/paper/ExpertRuleFit_paper.tex` (and `.pdf`; the `.md` is a pandoc mirror, regenerate with `pandoc ExpertRuleFit_paper.tex -t gfm -o ExpertRuleFit_paper.md`)
- Other notes: `docs/refinement/notes/CODE_FACTS.md`, `REFERENCES_TO_VERIFY.md`
- Code: `expertrulefit/expert_rulefit.py`, `expertrulefit/constrained.py`
- Experiments:
  - `experiments/table1.py`, `b1_stability.py`, `ablation.py`, `sensitivity.py`, `row_order.py`, `fetch_data.py`
  - Data CSVs are fetched by `fetch_data.py` and are not committed.
- Pinned stack: `docker/Dockerfile`, `docker/requirements.lock.txt`

## Known pitfalls

- **Python 3.11 f-strings:** the pinned image runs 3.11, where an f-string cannot reuse its own quote type inside `{...}` (allowed from 3.12). Precompute values or alternate quotes.
- **Foreground limit:** about 10 minutes per shell command in a chat session. **Background jobs are stopped** a few minutes after a reply ends. Run anything long (HMEQ/Taiwan fits: minutes to hours) in GitHub Actions, or leave a resumable command; scripts resume from their result files.
- **Pinned docker image:**
  - Build: `docker build -t erf-pinned docker/`.
  - If Docker Hub rate-limits (429), build with `--build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.11-slim@sha256:07d48240…` (same digest).
  - Run: `docker run --rm --network none -v "$PWD":/work -w /work -e PYTHONPATH=/work erf-pinned <command>`.
  - In a chat sandbox the docker daemon may need restarting (`nohup dockerd &`).
  - A stale `*.egg-info` can report a wrong version; the version is read from `_version.py`.
  - Stack: Python 3.11.18, scikit-learn 1.7.2 (capped < 1.8 because of imodels), NumPy 2.4.6, imodels 3.0.4, single-threaded.
- **Determinism rules:** seeds derive from `_BASE_SEED`; hash-check every new path in 3 runs; run the full test suite before each commit.
- **Paper limit: 7 pages** including references. Cut words; do not shrink type or margins.
- **Claims:** nothing public beyond CLAIMS_POLICY; treat methods suggested by reviews as unverified until their source is opened.
- **GitHub from a chat session:** GraphQL is blocked, so `gh pr view` fails; use REST (`gh api repos/hdjebar/expertrulefit/...`).
- **Not read:** the Ebner workshop PDF (a fetch needed approval that never came).

## Resume message (paste into a new session)

```
Resume the ExpertRuleFit refinement (repo hdjebar/ExpertRuleFit, branch main).
First read docs/refinement/SESSION_HANDOFF.md, then
docs/refinement/notes/CLAIMS_POLICY.md and FINDINGS_2026-11-22.md §14–18.
Do the "Pending, in order" list, starting with item 1 (Ebner citation:
thesis + R package only, remove KR4HC/AIME/LNCS everywhere; verify the
thesis details from a source you open), then item 3 (CI fix) and item 2
(tag v0.8.0). Work on a new branch per item, open a draft PR, keep all
standing rules in the handoff (determinism hash-checks, full test suite
before each commit, pinned docker image, paper ≤ 7 pages, no claims
beyond CLAIMS_POLICY, no long jobs in the chat: use GitHub Actions for
B1 on HMEQ/Taiwan). Ask me decisions (a)–(d) before implementing them.
```
