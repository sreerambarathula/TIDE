"""Precise, general-even-N1 codim-2 point finder, built specifically to answer
the node-count convergence question (PROJECT_LOG.md Sec. 8 TODO): does the
codim-2 point found at N1=4 (Sec. 21, Nsub*~31.36) hold up as N1 increases, or
does its location keep moving?

Why not just reuse fast_gap_scan.py's gap_scan_general for this: that tool's
Hopf value comes from a coarse (default 60-point) Npch grid scan plus linear
interpolation, which is fine for screening but produces a visibly quantized,
staircase-shaped gap(Nsub) (confirmed directly: fine Nsub scans at N1=6/8
showed the gap value jumping in discrete steps every ~0.01 in Nsub, an
artifact of which grid cell the interpolation lands in, not a real physical
feature) -- not precise enough to tell a genuine node-count shift apart from
grid noise.

This module instead does a real Newton-bisection refinement (same standard as
curves.py's hopf_npch, generalized from N1=2 to general even N1) for the Hopf
boundary, and reuses fold_npch (curves.py) unchanged since the fold is proven
independent of N1 (Sec. 16). Reuses pseudo_arclength._eigvals_jit for the
compiled eigenvalue evaluation -- same JIT fix (Sec. 23 update), so this is
fast despite being precise.
"""

import jax.numpy as jnp
import numpy as np

from tide.continuation.fast_gap_scan import _fold_npch_for_nsub
from tide.continuation.pseudo_arclength import _eigvals_jit

_FOLD_GRID_POINTS = 4000


def _fold(nsub, Fr, Lam, ki, ke):
    """Jitted fold Npch (fast_gap_scan._fold_npch_for_nsub), not curves.fold_npch
    -- curves.fold_npch's unjitted 8000-point Python-loop grid search (~1s/call,
    documented in fast_gap_scan.py's own module docstring) made the original
    version of this sweep redundantly call it twice per Nsub per N1, several
    hundred times total across a handful of N1 values, and it ran past 300s
    before being stopped. See the module-level cross-check at the bottom of
    this file's test coverage for confirmation that this substitution doesn't
    change the answer at the already-validated N1=4 point."""
    npch_grid = jnp.linspace(nsub * 1.02, nsub * 3.0, _FOLD_GRID_POINTS)
    val = float(_fold_npch_for_nsub(nsub, Fr, Lam, ki, ke, npch_grid))
    return None if np.isnan(val) else val


def _leading_complex_real_part_general(nsub, npch, Fr, Lam, ki, ke, N1):
    """Max real part among complex-conjugate eigenvalues at (nsub, npch), general
    N1. Returns -inf if no complex pair exists at this point."""
    eigvals = np.array(_eigvals_jit(nsub, npch, Fr, Lam, ki, ke, N1))
    complex_eigs = eigvals[np.abs(eigvals.imag) > 1e-9]
    if len(complex_eigs) == 0:
        return -np.inf
    return float(np.max(complex_eigs.real))


def hopf_npch_general(nsub, Fr, Lam, ki, ke, N1, bracket, tol=1e-9, max_iter=60):
    """Newton-bisection root-find of the leading complex pair's real part = 0,
    general even N1 -- the general-N1 analogue of curves.hopf_npch. Caller must
    supply a bracket known to contain a sign change."""
    lo, hi = bracket
    f_lo = _leading_complex_real_part_general(nsub, lo, Fr, Lam, ki, ke, N1)
    f_hi = _leading_complex_real_part_general(nsub, hi, Fr, Lam, ki, ke, N1)
    # An exact (or near-exact) root at either endpoint must be handled
    # explicitly -- see curves.hopf_npch's identical fix for why.
    if abs(f_lo) < tol:
        return lo
    if abs(f_hi) < tol:
        return hi
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        f_mid = _leading_complex_real_part_general(nsub, mid, Fr, Lam, ki, ke, N1)
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _gap(nsub, Fr, Lam, ki, ke, N1, hopf_bracket_halfwidth_frac):
    """hopf_npch_general - fold, both at this Nsub. Returns (gap, fold) or
    (None, fold) if the Hopf bracket has no sign change. fold is returned
    alongside so callers never need a second fold evaluation."""
    fold = _fold(nsub, Fr, Lam, ki, ke)
    if fold is None:
        return None, None
    hb = (fold * (1 - hopf_bracket_halfwidth_frac), fold * (1 + hopf_bracket_halfwidth_frac))
    hopf = hopf_npch_general(nsub, Fr, Lam, ki, ke, N1, hb)
    if hopf is None:
        return None, fold
    return hopf - fold, fold


def find_codim2_point(Fr, Lam, ki, ke, N1, nsub_bracket, coarse_points=25,
                       hopf_bracket_halfwidth_frac=0.03, nsub_tol=1e-6, max_iter=60):
    """Bisect on Nsub for gap(Nsub) = hopf_npch_general - fold = 0, i.e. the
    codimension-2 point, for a general even N1. Locates the sign-changing
    Nsub bracket via a coarse scan first (gap is not guaranteed monotonic far
    from the root, but is well-behaved near it -- same approach as the manual
    refinement in Sec. 21, now automated and generalized to any N1)."""
    lo, hi = nsub_bracket
    nsub_scan = np.linspace(lo, hi, coarse_points)
    gaps = [_gap(nsub, Fr, Lam, ki, ke, N1, hopf_bracket_halfwidth_frac)[0] for nsub in nsub_scan]

    bracket = None
    for i in range(len(gaps) - 1):
        if gaps[i] is None or gaps[i + 1] is None:
            continue
        if (gaps[i] > 0) != (gaps[i + 1] > 0):
            bracket = (nsub_scan[i], nsub_scan[i + 1])
            break
    if bracket is None:
        return None

    def g(nsub):
        return _gap(nsub, Fr, Lam, ki, ke, N1, hopf_bracket_halfwidth_frac)[0]

    a, b = bracket
    ga = g(a)
    for _ in range(max_iter):
        if b - a < nsub_tol:
            break
        mid = 0.5 * (a + b)
        gm = g(mid)
        if gm is None:
            return None
        if ga * gm < 0:
            b = mid
        else:
            a, ga = mid, gm
    nsub_star = 0.5 * (a + b)
    npch_star = _fold(nsub_star, Fr, Lam, ki, ke)
    return nsub_star, npch_star
