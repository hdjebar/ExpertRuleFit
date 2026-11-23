# Prompt for the next session (paste as first message)

---

I'm continuing the refinement of **ExpertRuleFit** (repo `hdjebar/ExpertRuleFit`, Python, deterministic RuleFit variant with protected expert rules). I'm attaching a handoff folder `ExpertRuleFit_refinement/`. Read these files before doing anything, in this order:

1. `README.md` — status and contents.
2. `notes/CODE_FACTS.md` — verified facts about the current code; trust these instead of re-deriving, but re-check if the repo has moved past commit `76f850f`.
3. `ACTION_PLAN.md` — the backlog, with acceptance criteria.
4. `notes/CLAIMS_POLICY.md` — what may and may not be claimed publicly.
5. `review/2026-11-22_paperreview-ai_review.md` — the AI review we are answering.
6. `paper/ExpertRuleFit_paper.tex` — the paper (source of truth).

Then:

- Clone or open the repo, create a branch `refinement/phase-a`, and do **Phase A (A1, A2)** with tests. Keep the determinism guarantee: every new random draw uses a seed derived from `_BASE_SEED`. Run the full test suite before and after.
- Check whether the three datasets (German Credit, Taiwan Credit, HMEQ) can be loaded in this environment. If not, tell me which files to attach, and prepare `experiments/perturbation.py` (Phase B1) so it runs from local CSVs.
- Do not change README/paper/CV claims beyond what `CLAIMS_POLICY.md` allows. When results come in, update the paper `.tex`, recompile the PDF, and list each claim with the file that supports it.
- Treat references suggested by the AI reviewer as unverified until you have opened their source page (`notes/REFERENCES_TO_VERIFY.md`).
- Commit with clear messages; open a PR per phase; summarise what changed, test results, and the next item.

Working style: short progress notes, no recap of steps, a task list for the phases, and a final check of numbers against the result files.

---
