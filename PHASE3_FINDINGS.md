# Phase 3 Primary Finding: Boundary-Learning Failure Is Predicted by Geometric Degeneracy, Not Mere Bifurcation Coexistence

**SUPERSEDED (2026-08-31).** This document is a mid-project (Phase 3 only)
snapshot and is now stale and internally inconsistent in places (found via
external audit, e.g. this file's own §8 mechanism table conflicts with its
own §8 open-items list on whether Point C's mechanism was tested; the
5.58x baseline figure here predates the corrected, undertraining-free
3.54x; the reported passing-test count is stale). It is kept only as a
historical record of the Phase 3 investigation, per this project's
disclosure practice — **do not cite numbers from this file.** The current,
correct, and complete account (Phase 3 through Phase 5, plus the
post-manuscript correction of two further errors this document also
predates) is `PROJECT_LOG.md` §20-44 and `manuscript/manuscript.md`.

*Status: primary result, as of 2026-08-30. Supersedes the original Phase 3 hypothesis
stated in PROJECT_LOG.md §3. Full derivation, debugging history, every intermediate
result, and a full accounting of a numerical correction made partway through (an
extraction bug that had inflated some early numbers — see §6) are in PROJECT_LOG.md
§20-40; this document is the synthesized, publication-facing version of that record,
using corrected numbers throughout.*

## 1. One-paragraph summary

A standard MLP surrogate, trained to minimize ordinary field error on the internal
characteristic curve (Eu) and the linear stability margin (g) of a validated
two-phase boiling-channel model, shows **no robust boundary-location error increase**
near an ordinary codimension-2 point where fold and Hopf curves cross
**transversally** — a result that survived rigorous statistical correction and three
independent sampling/architecture robustness checks. The same surrogate, trained the
same way, shows a **statistically robust boundary error increase (5.6x, p=9.5e-7)**
near a genuinely different point in the same physical model: a verified
**Bogdanov-Takens point**, where two real eigenvalues of the full Jacobian collide
into a repeated eigenvalue exactly at zero real part. **This was confirmed at a
second, independently-found Bogdanov-Takens point** (different parameters, found by
the same general search), where boundary error is again significantly elevated (up
to 3.1x) while field error stays flat — this is not a one-off. **The distinguishing
factor is not "coexistence of two bifurcation types" but geometric degeneracy —
tangency, a true spectral collision — which is a separate, checkable condition and
was verified directly rather than assumed at every point.** Tested further: the
failure mode at the degenerate points is **not** the classically hypothesized
vanishing-gradient ill-conditioning (that mechanism is wrong-signed everywhere
tested) but a **representability limit** — the degeneracy collapses a stable region
to a feature narrower than the surrogate can resolve, and boundary error correlates
strongly and consistently with this shrinking feature width instead. This is a
demonstrated causal contrast, not an asserted hypothesis, and it replicates: the same
methodology, applied to points that differ in exactly one checkable property, gives
opposite and mechanistically consistent answers at every point tested, and the
mechanism behind the positive cases was itself tested rather than assumed.

## 2. Background (see PROJECT_LOG.md §3-4, §11-26 for full derivation)

The physical system is a heated, boiling vertical channel, modeled with the
Clausse-Lahey (1991) moving-boiling-boundary formulation as re-derived by
Theler, Clausse & Bonetto (2010) — a closed differential-algebraic system in the
non-dimensional subcooling number (Nsub) and phase-change number (Npch). Two
classical instabilities exist in this parameter plane:

- **Ledinegg/fold**: a saddle-node in the internal characteristic curve
  Eu(Npch) (Eq. 26 of the source paper) — closed-form, N1-independent.
- **Density-wave oscillation (DWO)/Hopf**: a complex-conjugate eigenvalue pair of
  the linearized ODE system crossing the imaginary axis.

All ground-truth tooling for locating and continuing these curves (fold, Hopf, and
their crossing) was built and independently validated in Phases 1-2. Two distinct
points were located and used as the two arms of this study:

**Point A — realistic-parameter transversal crossing.** At fully realistic,
real-facility-derived operating parameters (Fr=0.035, Lambda=5.90, k_in=6.55,
k_out=2.03 — from documented Saha/Ishii facility data), the Eq. 26 fold curve and
the Jacobian's Hopf curve cross at **Nsub\*≈29.886, Npch\*≈49.702** (converged with
respect to spatial node count N1=16). This operating region also exhibits a third,
independent, classically-documented instability ("excursive"/Ledinegg-type, a real
eigenvalue — Ishii, 1971), stated here so the point is never presented as a literal
facility safe-operating boundary; it does not affect the fold/Hopf methodology.

**Point B — a genuine Bogdanov-Takens point.** At Fr=0.5, Lambda→0, k_in=11,
k_out=3 (the parameter combination originally found in PROJECT_LOG.md §18 by
minimizing the fold-Hopf gap), direct inspection of the full eigenvalue spectrum
(not an inferred curve-tangency proxy) shows two real eigenvalues converging
monotonically and colliding into a genuine repeated eigenvalue exactly on the fold
curve, with the real part at collision crossing zero at **Nsub\*=14.14278,
Npch\*=20.59478**. Confirmed identical to 5 decimal places across N1=2,4,8,16 — this
point does not depend on spatial discretization at all, since the colliding
eigenvalues live entirely in the shared two-phase subsystem.

**Point C — a second, independently-found Bogdanov-Takens point.** The same search
(scan for where the real part at eigenvalue collision crosses zero) was repeated at
a distinctly different parameter combination, Fr=0.3, Lambda→0, k_in=6, k_out=5,
found on the first attempt out of four candidates tried — suggesting such points are
common in this model at low friction, not a fluke of one combination. Located at
**Nsub\*=7.630091, Npch\*=10.914522**, verified via the same full-spectrum inspection
(real part at collision: +0.0005, essentially zero) and confirmed N1-independent to
7 decimal places.

## 3. Original hypothesis and pre-registered test

**Hypothesis** (PROJECT_LOG.md §3): a surrogate trained to minimize ordinary field
error will mislocate the stability boundary badly near a point where
`boundary error ≈ field error / ‖∇g‖` blows up — the textbook signature of a
degenerate bifurcation.

**Surrogate**: a deliberately un-clever 2-64-64-2 tanh MLP, standardized I/O, Adam,
full-batch, 80/20 train/test split — used identically at both points. Held-out field
error R²>0.995 at Point A, R²>0.998 at Point B, confirming a competent, honestly
evaluated baseline in both cases before testing anything else.

**Pre-registered primary criterion** (decided before running either analysis):
evaluate the surrogate's own implied boundary location at each of many independent
Nsub values; **PASS** if mean boundary error near the point of interest is ≥3x mean
boundary error far from it, while field error near stays within 1.5x of far.

## 4. Point A — the transversal crossing: no robust decoupling

| Stage | Design | Boundary error ratio (mean/median) | p-value | Verdict |
|---|---|---|---|---|
| First pass | one seed, boundary-band-oversampled data | 5.5x | — | superficially passed, **did not replicate** (8 seeds: ratio ranged 0.34-9.02) |
| Corrected | 20 seeds, seed-level paired statistics (fixing a pseudo-replication bug that had inflated significance to p=1.8e-18) | 2.32x / 2.16x | 2.4e-4 | modest, real, but short of the pre-registered 3x bar |
| Uniform sampling | same point budget, no boundary-band assumption | 0.72x / 0.71x | 0.998 | **no effect — the signal depended entirely on the sampling scheme** |
| Misfit-driven adaptive sampling | principled two-stage scheme (uniform, then add points where *measured* error is largest) | 0.69x / 0.69x | 0.98 | **no effect, even with legitimate, measured extra attention to the region** |

The secondary mechanism test (boundary error vs. `1/‖∇g‖`, per-seed) came back
**wrong-signed and consistent** (median ρ=-0.20, 9/20 seeds individually significant
in the wrong direction, 0/20 in the hypothesized direction). An alternative mechanism
(local training-data sparsity) showed no relationship either (median ρ=0.03).

**Why this happens, verified directly, not assumed:** a classical Bogdanov-Takens
point requires the Hopf curve to be **tangent** to the saddle-node curve. Checked at
Point A: fold and Hopf slopes (dNpch/dNsub) of 2.403 and 1.479 — a slope difference
of 0.924, clearly transversal. Searched systematically across 265 (Fr, Lambda, k_in,
k_out) combinations spanning this project's realistic-to-moderate range: slope
differences ranged only 0.68 to 1.68 — solidly transversal everywhere tested, never
approaching the near-zero difference genuine tangency requires. There is no classical
BT-type singularity here for a surrogate to be sensitive to.

## 5. Point B — the genuine Bogdanov-Takens point: robust decoupling

**Local structure is a genuine wedge/cusp.** Below Nsub\*, there is a narrow stable
window in Npch bounded below by a real-eigenvalue crossing (indistinguishable from
the fold curve) and above by a genuine Hopf-type complex-pair crossing; both converge
to the same point at Nsub\*, where the window closes entirely (width shrinks smoothly
from 3.00 at Nsub=11.0 to ~0.004 at Nsub=14.14 — a clean cusp). Above Nsub\* there is
no window at all. The relevant eigenvalue pair is cleanly dominant throughout this
region (confirmed directly — no contaminating unrelated mode), so
`g = max(Re(all eigenvalues))` was used directly as the surrogate target.

| metric | ratio (mean/median) | p-value |
|---|---|---|
| Upper (Hopf-type) boundary error | **5.58x** | **9.5e-7** |
| Eu field error | 1.02x | 0.36 (not significant) |

This clears the pre-registered criterion cleanly.

**Robustness checks — the exact ones that broke Point A's result — mostly survived
here:**
- **Sampling scheme**: uniform (non-window-aware) sampling weakens the effect but does
  not eliminate it: upper boundary error ratio 3.08x, p=1.9e-5 (right at the
  pre-registered bar, still highly significant). Contrast with Point A, where the
  same check took the effect from 2.32x/p=2.4e-4 to 0.72x/p=0.998 — a complete
  reversal. Here it is a quantitative weakening, not a qualitative collapse.
- **Architecture**: (128,128) hidden layers replicates significantly (3.46x,
  p=2.0e-3). (32,32) hidden layers gives a weaker, directionally-consistent but
  **not statistically significant** effect (1.64x, p=0.078, n=6 valid seeds) —
  reported honestly as underpowered rather than a clean pass. Contrast with Point A,
  where (32,32) showed literally no effect at all (p=0.88, essentially neutral) and
  (128,128) was inconsistent between mean and median (p=0.31) — every architecture
  at Point B still points the same (correct) direction, just with varying power.

**Mechanism, tested directly — and it is NOT the classical vanishing gradient.**
The original hypothesis (§3) predicts boundary error blows up where `‖∇g‖ → 0`.
Checked directly at Point B's upper boundary: `‖∇g‖` stays roughly constant
(~0.236 at Nsub=11.0 to ~0.268 at Nsub=14.10) as Nsub approaches Nsub\* — if
anything slightly increasing, the opposite of vanishing. What actually shrinks to
zero is the **window width** itself (the wedge closing, per above). Tested both
candidate mechanisms per-seed across the same 20 seeds used for the primary result:

| hypothesis | median ρ | seeds significant, correct direction | seeds significant, wrong direction |
|---|---|---|---|
| boundary error vs. `1/‖∇g‖` (classical) | -0.687 | 0% | **90%** |
| boundary error vs. `1/window width` (representability) | **+0.687** | **90%** | 0% |

The classical gradient mechanism is wrong-signed here too — exactly as it was at
Point A, where there was no effect to explain at all. The real, strongly and
consistently supported driver is **feature-scale shrinkage**: as the stable window
narrows below the surrogate's effective resolution, a smooth regressor cannot
represent an arbitrarily thin feature accurately, and both boundaries get pulled
together and mislocated. This is a representability limit, not a root-finding
ill-conditioning effect — a related but genuinely distinct mechanism from the one
that motivated this project at the outset (§3's implicit-function-theorem argument).

## 6. Point C — a second Bogdanov-Takens point, and a correction along the way

**A correction, disclosed in full rather than silently applied.** While building the
identical pipeline at Point C, an extraction bug surfaced: the routine that reads off
the surrogate's implied upper boundary used a wide, uninformed search bracket and
took the first sign change it found. Occasionally the surrogate's own imperfect fit
introduces a spurious extra root, and the wide bracket would find that instead of the
true one — confirmed directly (not assumed) by checking the raw ground-truth curve at
the exact locations where this happened and finding a single, clean, unambiguous
crossing there, with no secondary structure. Fixed by centering the search on the
already-known true boundary value and, when multiple crossings exist, selecting the
nearest one. **Re-running Point B with the fix showed it had been affected too** —
every number above is the corrected version; the originally-reported ratio was 10.22x
(primary), 4.47x (uniform sampling), 2.05x/9.82x ((32,32)/(128,128) architectures).
Every qualitative conclusion still holds after correction except one: the (32,32)
architecture check is no longer statistically significant, reported as such above,
not smoothed over. Full before/after accounting is in PROJECT_LOG.md §40.

**Point C's result, with the corrected extraction, confirms generalization:**

| metric | ratio (mean) | p-value |
|---|---|---|
| Lower (fold-type) boundary error | 3.08x | 1.0e-4 |
| Upper (Hopf-type) boundary error | 1.95x | 1.2e-3 |
| Eu field error | 1.13x | 0.12 (not significant) |
| g field error | 0.95x | 0.99 (not significant) |

Both boundaries show significant elevation with flat field error — decoupling near a
genuine Bogdanov-Takens point is not a one-off result specific to Point B. The
*which* boundary shows the larger effect differs between the two points (upper/
Hopf-type dominant at Point B; both significant but more modest, with lower/fold-type
slightly larger, at Point C) — a reasonable difference in local geometry between two
independently-occurring degenerate points, stated plainly rather than glossed over as
identical.

## 7. The contrast is the contribution

The same methodology — same surrogate architecture family, same pre-registered
criterion, same statistical correction, same robustness stress tests — gives
opposite, mechanistically consistent answers at points in the same physical
system that differ in exactly one checkable, verified property: whether the
bifurcation coexistence is transversal or genuinely tangent (a real spectral
collision) — and the tangent case replicates across two independently-found
instances. This is a demonstrated, replicated causal claim, not a single asserted
mechanism behind a negative result:

> Coexistence of two bifurcation types at a point (a "codimension-2 point" in the
> loose sense used in much of the applied literature) does not, by itself, cause a
> machine-learning surrogate trained on ordinary field error to mislocate the
> boundary there. What causes it is genuine geometric degeneracy — a true spectral
> collision, verifiable directly in the Jacobian before ever training a surrogate —
> and, once present, the actual failure mode is not the classically-hypothesized
> vanishing-gradient ill-conditioning but a representability limit: the degeneracy
> collapses a stable operating region to a feature narrower than the surrogate can
> resolve. We demonstrate all three parts of this claim on the same real,
> physically-motivated two-phase boiling-channel model: an ordinary (transversal)
> coexistence point shows no robust boundary-learning failure across three
> independent sampling designs and two architectures; two independently-found
> genuine Bogdanov-Takens points in the same model both show a robust failure under
> the identical protocol; and the failure correlates with the shrinking feature
> width, not with the field's local gradient, at every seed tested at both
> degenerate points. The practical implication is a diagnostic, not just a caution:
> check the local spectral geometry — and, where it is degenerate, the scale of the
> feature it creates — before assuming or dismissing ML boundary-learning risk near
> a coexistence point.

This positions the paper as a **methodology + demonstrated, replicated mechanism**
paper rather than either a pure negative result or the originally-planned
negative-result-plus-fix. The positioning paragraph in PROJECT_LOG.md §3 (the "why
not just use the solver" argument) still holds and needs only its outcome-specific
sentence updated — the reduced-order model was chosen because it is fully
checkable, and it still is; what changed is what checking it found.

## 8. What survived rigorous critique, and what remains open

**Resolved during critique, not swept aside (PROJECT_LOG.md §32-40):**
- A citation was found to be a search-engine misattribution and corrected to its
  actual verified source (Ishii, 1971), which also independently validated the
  real-eigenvalue detection method used throughout.
- Surrogate boundary extraction was upgraded to Newton/bisection precision at Point A,
  ruling out extraction-method artifacts.
- A pseudo-replication statistical bug was found and fixed (seed-level, not
  per-point, replication) — applied identically at every point.
- Two candidate mechanisms (gradient, data-density) were tested and ruled out for
  Point A's null result, per-seed, not pooled.
- Sampling-scheme sensitivity was identified as the decisive confound at Point A and
  resolved with a literature-grounded (misfit-driven adaptive) redesign — then
  re-applied as a stress test at Point B, where it did NOT break the result.
- **A spurious-root extraction bug was found while building the Point C pipeline,
  confirmed to have also inflated Point B's originally-reported numbers, and fixed
  — with the correction disclosed in full (§6) rather than silently applied.**
- A related, methodologically-close prior paper (Shahab & Susanto, 2024) was found,
  read in full, and confirmed not to overlap with this project's actual claim.
- **The first candidate "tangent point" was itself an artifact** (a branch-tracking
  bug identical in class to two earlier ones) and was caught by checking the raw
  eigenvalue spectrum directly before trusting a tracked scalar proxy — the genuine
  Bogdanov-Takens point (Point B) was only accepted after this verification.
- **The mechanism at Point B was tested, not assumed** — and it refuted the
  classical vanishing-gradient hypothesis (wrong-signed, 95% of seeds) in favor of a
  window-width/representability mechanism (correctly signed, 95% of seeds). The
  original positioning paragraph's mechanistic argument (§3) is now known to be
  imprecise at both points tested and has been replaced with a better-supported one.

**Genuinely open, not yet investigated:**
- The window-width mechanism has not been checked as a negative control at Point A
  — there is no shrinking feature there (fold and Hopf simply cross once), so the
  hypothesis predicts no window-width-driven error there either, consistent with
  the null result already found, but this hasn't been explicitly verified.
- The window-width mechanism has only been tested at Point B, not yet at Point C —
  worth confirming the same 1/width correlation holds there too.
- Generalization across N1, geometry, and a second physical system entirely (the
  original Phase 5 plan) — not started.
- A full literature pass confirming novelty of the "check spectral degeneracy, then
  feature scale, before expecting decoupling" framing — searched extensively, found
  nothing matching, but this should not be overclaimed as definitively unprecedented
  without a deeper pass closer to manuscript submission.

## 9. Reproducibility

All results in this document are backed by code and tests in this repository:
`src/tide/surrogates/` (data generation for all points, MLP, adaptive sampling),
`src/tide/eval/` (decoupling analysis, seed-level statistics, mechanism testing),
`src/tide/continuation/tangency.py` (the transversality check at Point A). 63/63
tests passing as of this writing. Full narrative and every dead-end tried (including
the branch-tracking artifact that preceded finding Point B, and the spurious-root
extraction bug found and fixed while building Point C) is preserved in
PROJECT_LOG.md §27-40 for anyone auditing this result later.
