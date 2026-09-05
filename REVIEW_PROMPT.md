# Review brief: Project TIDE repository audit

You are auditing a research repository for a manuscript being prepared for submission to *Reliability Engineering & System Safety*. You have not seen this project before. Do not trust any summary claim (including this one) without checking it against the actual files. Your job is to find discrepancies, redundancy, and structural problems — not to re-derive the science from scratch.

## What this project is (30-second version)

TIDE studies whether a machine-learning surrogate trained on a physical model's field data correctly locates a safety-critical stability boundary in a two-phase (boiling) flow channel — or whether it can look accurate on average while being badly wrong exactly at the boundary that matters. The central claim: this failure occurs specifically at a genuine mathematical degeneracy (a Bogdanov–Takens point), not at every place two instability types coexist. The paper tests this with a real physical model, real statistics (seed-level replication, not single runs), and reports several self-caught bugs and a methodological correction transparently rather than hiding them.

## Where to start

1. **`PROJECT_LOG.md`** is the single source of truth for the project's history — a chronological, numbered log (currently ~43 sections) of every decision, bug, and result, with reasoning, not just outcomes. Read the end first (the last 2-3 sections) for current state, then spot-check earlier sections referenced from the manuscript.
2. **`manuscript/manuscript.md`** is the paper draft itself (Markdown source; `manuscript/RESS_manuscript_draft.docx` is the same content converted to Word).
3. **`PHASE3_FINDINGS.md`** is an *earlier* standalone findings document written mid-project, before the manuscript existed. Check whether it is now fully superseded by the manuscript or still contains information the manuscript is missing — this is one of the redundancy questions this audit should answer, not something to assume either way.
4. **`src/tide/`** is the actual implementation: `physics/` (governing equations), `continuation/` (fold/Hopf/codimension-2 point finding), `surrogates/` (the ML models), `eval/` (statistical comparison machinery).
5. **`tests/`** — one test file roughly per source module (naming isn't always 1:1; e.g. `test_mlp_surrogate.py` tests `surrogates/mlp.py`, `test_bt_point.py` and `test_bt_point_second.py` between them test `surrogates/bt_point_data.py` at two different physical points).
6. **`manuscript/supplementary_material.md`** documents bug fixes and corrections in detail — cross-check its claims against `PROJECT_LOG.md`'s own account of the same events.

## Specific things to check — do not skip any of these

### A. Numerical consistency
Every quantitative claim in `manuscript/manuscript.md` (Table 2, 3, 4; every ratio, error value, and p-value in Section 5) should trace back to a specific entry in `PROJECT_LOG.md`. Pull at least 10 numbers from the manuscript at random and confirm each one appears, matching, in the log. Flag any number that appears in the manuscript but not the log, or that differs between the two.

### B. Redundant or superseded files
- Is `PHASE3_FINDINGS.md` still needed, or should it be merged into / replaced by the manuscript and archived?
- `src/tide/continuation/augmented_systems.py` is described in `PROJECT_LOG.md` (search for "augmented_systems") as a deliberately-kept record of a dead-end investigation, not a recommended tool. Confirm nothing else in the codebase actually depends on it for a real result, and that the manuscript doesn't cite it as if it were live.
- Check for any other module the log describes as "superseded" and confirm current code doesn't silently still call it.

### C. Manuscript-to-code correspondence
Every function, file, or hyperparameter the manuscript or supplementary material names (e.g. `wedge_boundaries`, `_extract_surrogate_upper`, specific architectures like `(128,128,128)`, `eps_w=0.1`) must actually exist in the code with that name and behavior. Do not assume the manuscript's prose description is accurate — grep for each cited symbol.

### D. Test suite integrity
Run the full suite (`pytest tests/ -q` inside the project's `.venv`) and confirm it passes. Check that the three most recently added surrogate variants (`fourier_mlp.py`, `log_distance_mlp.py`, `boundary_weighted_mlp.py`) each have real assertions in their test files, not just smoke tests that would pass on broken code.

### E. Manuscript completeness and submission-readiness
- `[Author Name]` / `[Institution]` placeholders are intentionally unfilled — confirm every occurrence is a genuine placeholder, not an accidental blank elsewhere.
- References [11], [17], [18], [19] are flagged in the manuscript's own reference list as incomplete (missing full author lists / DOIs). Confirm no *other* reference silently has the same problem without being flagged.
- Elsevier submission constraints already checked and fixed once (Highlights ≤85 characters each, Abstract ≤200 words, ≤6 keywords, ≤13,000 words total) — re-verify these are still satisfied after any further edits, since violating them again is easy to reintroduce accidentally.

### F. Stated limitations vs. actual scope
The manuscript states the two genuine Bogdanov–Takens points require a friction parameter $\Lambda \to 0$ limit specific to this model, and that a transversal point at realistic friction was independently confirmed. Verify both claims against `PROJECT_LOG.md` (search "low-friction", "realistic parameters", "Pandey"). Flag if the manuscript's framing overstates or understates what was actually shown.

## How to report findings

For each issue found, give: the file and line/section, what the discrepancy is, and which of the two conflicting sources (if any) appears correct based on the underlying code/data — not just "these don't match." Distinguish between (1) a real factual/numerical error, (2) a structural or redundancy issue, and (3) a stylistic inconsistency, so the author can prioritize.
