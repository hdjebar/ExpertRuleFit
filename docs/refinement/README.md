# ExpertRuleFit — refinement pack

Handoff folder to continue improving ExpertRuleFit (code + paper) in a new conversation or a Claude Code session. Prepared 22/11/2026.

## Start here

1. Open a new session on the repo `hdjebar/ExpertRuleFit`.
2. Paste the contents of `PROMPT_next_session.md` as the first message, and attach this folder (or copy it into the repo under `docs/refinement/`).
3. Work through `ACTION_PLAN.md` in order. Each item has acceptance criteria.

## Contents

| File | What it is |
|---|---|
| `PROMPT_next_session.md` | First prompt (Phase A) — done |
| `PROMPT_after_review3.md` | Prompt for the next session after review 3 |
| `ACTION_PLAN.md` | Prioritised backlog mapped to the review, with effort and acceptance criteria |
| `notes/CODE_FACTS.md` | Verified facts about the current implementation (defaults, seeds, validation protocol), so nothing has to be re-derived |
| `notes/CLAIMS_POLICY.md` | What may and may not be claimed in the paper, README and CV until new results exist |
| `notes/REFERENCES_TO_VERIFY.md` | References already cited and those suggested by the reviewer, with verification status |
| `paper/ExpertRuleFit_paper.tex` | Paper source (current version, 6 pages) — the source of truth |
| `paper/ExpertRuleFit_paper.pdf` | Compiled version submitted to paperreview.ai |
| `paper/ExpertRuleFit_paper.md` | Markdown conversion of the paper, for reading or diffing |
| `review/2026-11-22_paperreview-ai_review.md` | Review 1 (paperreview.ai) |
| `review/2026-11-22_review-2_after-phase-a.md` | Review 2, of the paper after Phase A |
| `review/2026-11-23_review-3_paper-v2.md` | Review 3, of paper v2 (7 pages) |

## Status on 22/11/2026

- **Repo:** `main` at commit `76f850f` (23/02/2026). No refinement code written yet.
- **Paper v1:** reviewed once by paperreview.ai. Verdict in substance: well written, honest about limits, engineering integration of known ideas; needs perturbation stability results, ablations, baselines (SIRUS first), sign constraints, calibration, sensitivity and runtime.
- **Living version of the note:** Claude Docs doc "ExpertRuleFit: reproducible, expert-constrained rule ensembles — technical note" (in the author's artifacts). The `.tex` in this folder is more recent on calibration and is the reference.
- **Deep Review (Google):** not yet run.

## Context

ExpertRuleFit is also cited in a job application (Data Scientist, Administration des contributions directes, Luxembourg; submission 26/11/2026, interviews later). The CV wording was narrowed to determinism on 22/11. See `notes/CLAIMS_POLICY.md` before changing any public wording.
