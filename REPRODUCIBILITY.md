# TIDE: Complete Reproducible Workflow

This document traces every number, table, and figure in `manuscript/manuscript.md`
back to the exact code that produces it, in dependency order. Every claim below
was checked by reading the actual source file(s) named, not inferred from
docstrings, filenames, or `PROJECT_LOG.md` prose alone. Status tags:

- ✅ **Verified real & runnable** — real code, calling real physics, checked this session
- 🔧 **Real, but has a known bug to fix first** — genuine computation, wrong current output
- ⚠️ **Real building blocks exist, no single script wires them together yet**
- ❌ **Not applicable / out of scope** — (nothing in the core pipeline is tagged this; used only to note the excluded `Figures/` tree, see bottom)

Scope: this covers the actual manuscript (Tables 2–4, Figures 1–4 as they appear
in `manuscript/manuscript.md`). The separate `Figures/figure_1`–`figure_8` tree
is explicitly out of scope for this document — at least one of its data
generators (`scripts/generate_fig4_ground_truth_data.py`) was confirmed to be
100% synthetic/hardcoded, unrelated to the manuscript, and the whole tree is
excluded from "reproducible" until/unless it's separately triaged.

---

## Stage 0 — Governing equations (physics core)

| Component | File | Status |
|---|---|---|
| Eq. 26 closed-form Euler number `Eu(Npch, Nsub, Fr, Λ, ki, ke)` | `src/tide/physics/ledinegg_curve.py::euler_number` | ✅ Verified term-by-term against the actual source paper's own plain-text formula (its `mochin` input file appendix), and numerically self-consistent |
| Full moving-boiling-boundary ODE system, general even N1 | `src/tide/physics/clausse_lahey.py::state_derivative_general` / `steady_state_general` | ✅ Verified: steady-state plugged into the ODE gives residual ~1e-15 (machine precision) at N1=2 and N1=16 |
| HEM closed-form validation anchor (Hurley 2023) | `src/tide/physics/hem_boundary.py` | Used only as a Phase-1 cross-check, not part of the reported results — not independently re-verified this session |

**Reproduce:** `pytest tests/test_ledinegg_curve.py tests/test_clausse_lahey.py tests/test_physics_validation.py tests/test_general_n1.py tests/test_hem_boundary.py -q`

---

## Stage 1 — Locating and verifying Points A, B, C (Table 2)

| Task | File(s) | Status |
|---|---|---|
| Fold curve (Eq. 26 Npch-extremum) | `src/tide/continuation/curves.py`, `codim2_convergence.py::_fold` | ✅ real |
| Hopf curve (leading complex-pair eigenvalue crossing) | `codim2_convergence.py::hopf_npch_general` | ✅ real; endpoint-bisection bug (exact root at bracket edge) fixed per Supplementary S4.5 |
| Fast batched screening (coarse (Fr,Λ,ki,ke,Nsub) sweeps, before refining) | `src/tide/continuation/fast_gap_scan.py::gap_scan_general` | ✅ real, JIT+vmap batched |
| Transversality check (Point A: fold/Hopf local-slope difference) | `src/tide/continuation/tangency.py::crossing_slopes` | ✅ real |
| Genuine double-zero (BT) point solver (Points B, C) | `src/tide/continuation/double_zero.py::find_double_zero_point` | ✅ real — 2D Newton on (sum, product) of the two smallest-magnitude eigenvalues; converges to residual <1e-9 |
| Real-mode (excursive) instability threshold check | `src/tide/continuation/real_mode_boundary.py::find_nsub_crit` | ✅ real — confirms Nsub* < Nsub_crit so the codim-2 point is physically meaningful |
| Known dead end, correctly isolated | `src/tide/continuation/augmented_systems.py` | Confirmed not imported by anything except its own regression test; not used for any reported result |

**What's scripted today:** none of the above is wired into one runnable "find and verify Points A/B/C from scratch" script — each was invoked ad hoc with hand-chosen search grids/initial guesses, recorded as prose in `PROJECT_LOG.md` (§16–21 for Point A's realistic-parameter search, including the exact grid `Fr∈{0.03,0.035,0.042,0.05,0.08}, Λ∈{2.0,2.8,3.5,4.5,5.9,7.0,8.5}, k_in∈{2.85,6.55,10,15,17.8}, k_out∈{0.03,0.5,2.03,5,10.66}` — 875 combinations; §37/40/44 for Points B/C's `double_zero.py` solve from an initial guess near the low-friction region; the 265-combination transversality sweep at `Fr∈[0.03,1.0], Λ∈[0.5,5.9], ki∈[2,11], ke∈[2,5]`, §2076-2081).

**⚠️ Missing piece:** a `scripts/find_and_verify_points.py` that runs `fast_gap_scan.gap_scan_general` over the documented grids, refines with `double_zero.find_double_zero_point` / `curves`'s Newton-refined fold+Hopf, and checks `tangency.crossing_slopes` — reproducing Table 2 from the search grids alone rather than from hand-picked starting points. All the underlying functions are real and already verified; only the top-level orchestration script doesn't exist yet.

**Reproduce what exists today** (re-verify, not re-discover, the already-known coordinates):
```python
from tide.continuation.double_zero import find_double_zero_point
# Point B
find_double_zero_point(x0=(14.0, 20.5), Fr=0.5, Lam=0.001, ki=11.0, ke=3.0, N1=16)
# Point C
find_double_zero_point(x0=(7.5, 10.8), Fr=0.3, Lam=0.001, ki=6.0, ke=5.0, N1=16)
```

---

## Stage 2 — Dataset generation

| Dataset | File | Status |
|---|---|---|
| Point A: boundary-band + background sampling | `src/tide/surrogates/data_generation.py::generate_stability_dataset` | ✅ real — calls `_fold`, `hopf_npch_general`, `euler_number`, LHS-samples Npch near both boundaries plus background coverage |
| Point A: misfit-driven adaptive augmentation (2-stage) | `src/tide/surrogates/adaptive_sampling.py::measure_misfit_and_augment` | ✅ real — trains a surrogate on Stage-1 data, measures true misfit on a uniform candidate pool, adds the highest-misfit points |
| Point B/C: BT-point dataset | `src/tide/surrogates/bt_point_data.py::generate_bt_dataset` | ✅ real — already used in `scripts/run_phase4_experiments.py` |
| Point B/C: misfit-driven adaptive augmentation | `src/tide/surrogates/bt_adaptive_sampling.py::measure_misfit_and_augment_bt` | ✅ real, same scheme adapted to the BT dataset format |

**Important, verified nuance:** `g` for Point A's dataset is the leading real part among the **complex-conjugate pair only** (`data_generation.py::_leading_real_part`), *not* `max` over all eigenvalues — because a persistent real "excursive" eigenvalue dominates everywhere in Point A's region and never changes sign, which would make the fold/Hopf boundary invisible to a surrogate trained on the naive all-eigenvalue max. This exactly matches Section 4.4's stated difference in stability-indicator definition between the Point A analysis and the Point B/C analysis (`bt_point_data.py::_g`, which *does* use the all-eigenvalue max, correctly, since Points B/C don't have this contaminating mode nearby).

**⚠️ Missing piece:** no single script currently generates and saves Point A's three named sampling-scheme datasets (uniform, boundary-band-oversampled, misfit-driven-adaptive) as committed artifacts the way `run_phase4_experiments.py` does for Point B/C.

---

## Stage 3 — Surrogate training

| Variant | File | Status |
|---|---|---|
| Baseline (2×64, tanh) | `src/tide/surrogates/mlp.py::train_surrogate` | ✅ real |
| Random Fourier features | `src/tide/surrogates/fourier_mlp.py` | ✅ real, tested (`tests/test_fourier_mlp.py` — real coverage, not smoke-test) |
| Log-distance-from-BT feature | `src/tide/surrogates/log_distance_mlp.py` | ✅ real, tested |
| Boundary-weighted loss (`1/(|g_true|+ε)`) | `src/tide/surrogates/boundary_weighted_mlp.py::train_surrogate_boundary_weighted` | ✅ real, formula confirmed to match Section 5.4's description exactly |

---

## Stage 4 — Statistical evaluation

| Analysis | File | Status |
|---|---|---|
| Point A primary test (near/far boundary vs. field error, seed-level paired Wilcoxon) | `src/tide/eval/decoupling.py::run_decoupling_analysis`, `src/tide/eval/stats.py::run_and_summarize` | ✅ real — near ≤1.0 / far ≥4.0 units, matches Section 4.4 exactly |
| Point A mechanism test (gradient vs. training-data-density) | `src/tide/eval/stats.py::run_mechanism_study` | ✅ real, per-seed Spearman correlation, not pooled |
| Point B/C primary test | `src/tide/eval/bt_decoupling.py::run_bt_decoupling_analysis` | ✅ real — near ≤0.5 / far ≥2.0 units, matches Section 4.4 exactly. Uses the corrected, true-value-centered `_extract_surrogate_upper` (Supplementary S3's fix) |
| Point B/C window-width mechanism test (Section 5.3) | Not a dedicated module — `wedge_boundaries_true` (window width) + `bt_decoupling.true_gradient_norm_at` would need to be combined | ⚠️ **Missing piece**: no script currently runs this correlation and reports it; the two ingredient functions exist and are real |

---

## Stage 5 — Tables and Figures

| Manuscript item | Source | Status |
|---|---|---|
| **Table 2** (Points A/B/C coordinates) | Stage 1 | ✅ coordinates verified to match `double_zero.py` output; the discovery process itself isn't a single script (see Stage 1) |
| **Table 3/4** (Point B/C training results) | `scripts/run_phase4_experiments.py` | 🔧 Real code, but the last saved run (`data/generated/phase4_results_full_20260831_161035.json`) mixed genuinely-executed configs with 3 rows hand-copied from a prior killed run (`scripts/run_remaining_configs.py`'s `ALREADY_DONE_B`) — **this is exactly what the HPC pipeline in `hpc/01_tide_phase4_rerun/` fixes**, by executing all 7 (point, config) pairs fresh, split into 22 seed-batch tasks |
| **Section 5.1 numbers** (Point A: R²>0.99, sampling/architecture robustness, 265-combo sweep) | Stages 1/2/3/4 combined | ⚠️ **Missing piece** — never consolidated into a script; only exists as `PROJECT_LOG.md` prose. All ingredients are real and verified above |
| **Section 5.3 mechanism numbers** (gradient vs. window-width correlation, Point B) | Stage 4 | ⚠️ **Missing piece**, same as above |
| **Fig. 1** (S-curve schematic) | `manuscript/make_fig1_scurve.py` | ✅ deliberately illustrative (cubic schematic), matches caption, no data dependency |
| **Fig. 2** (fold/Hopf maps, Points A & B) | `manuscript/make_fig2_bifurcation_maps.py` | 🔧 **Bug**: imports the superseded `wedge_boundaries`/`NSUB_BT`/`NPCH_BT` (pre-audit BT coordinates) instead of `wedge_boundaries_true` + Table 2's corrected coordinates — the actual image likely shows the wrong Point B |
| **Fig. 3** (window-width scaling) | `scripts/generate_fig3_ground_truth_data.py` → `manuscript/make_fig3_window_scaling.py` | ✅ Verified real — calls actual continuation/physics solvers, uses corrected coordinates and `wedge_boundaries_true` |
| **Fig. 4** (fix comparison, Table 3 visualized) | `manuscript/make_fig4_fix_comparison.py` | 🔧 Reads real Table 3 data but is missing one row (`4×128 + boundary-weighted loss`, the specific example Section 5.4's text discusses) |

---

## Dependency graph (top to bottom = execution order)

```
Stage 0 (physics equations, self-consistency)
   |
Stage 1 (locate & verify Points A, B, C  ->  Table 2)
   |
Stage 2 (generate datasets per point, per sampling scheme)
   |
Stage 3 (train surrogates: baseline + each fix, per point)
   |
Stage 4 (statistical evaluation: primary test + mechanism test, per point)
   |
Stage 5 (tables, figures)
```

## What's needed to make this 100% complete and script-driven

Three new orchestration scripts, each combining only already-verified real
functions (no new physics, no new statistics — just wiring):

1. `scripts/find_and_verify_points.py` — Stage 1, reproducing Table 2 from
   documented search grids rather than hand-picked starting points.
2. `scripts/run_point_a_analysis.py` — Stages 2–4 for Point A: all three
   sampling schemes × baseline + 3 architectures, reproducing Section 5.1's
   numbers with committed per-seed data (same discipline as
   `run_phase4_experiments.py`).
3. `scripts/run_mechanism_test.py` — Stage 4's window-width correlation for
   Section 5.3, at Point B (and, per the manuscript's own text, a 6-seed
   recheck at whatever point is relevant).

Plus two fixes to existing, already-real code:
4. Fix `manuscript/make_fig2_bifurcation_maps.py` to use `wedge_boundaries_true`
   and Table 2's corrected coordinates.
5. Add the missing `4×128 + boundary-weighted loss` row to
   `manuscript/make_fig4_fix_comparison.py`.

None of this requires fabricating anything — every function these five items
would call already exists, is real, and was verified this session.
