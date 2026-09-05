"""Phase 3-at-the-BT-point training data (PROJECT_LOG.md Sec. 37-38): the
positive-counterpart experiment to data_generation.py's negative result.

At the genuine Bogdanov-Takens point found in Sec. 37 (Fr=0.5, Lambda->0,
k_in=11, k_out=3, Nsub*=14.14278, Npch*=20.59478), the local structure is
fundamentally different from the transversal codim-2 point tested in Sec.
29-34: below Nsub*, there is a narrow STABLE WINDOW in Npch bounded below by
a real-eigenvalue zero-crossing (indistinguishable from the fold curve to
high precision, Sec. 37) and above by a genuine Hopf-type complex-pair
crossing. Both boundaries converge to the same point at Nsub*, where the
window closes entirely -- a genuine wedge/cusp, not a transversal crossing.
Above Nsub*, no such window exists at all (confirmed directly: the sign
pattern of g over Npch is a single "+" throughout, Sec. 37).

Because the relevant eigenvalue pair is cleanly DOMINANT throughout this
region (confirmed directly in Sec. 37 -- no contaminating unrelated mode
the way the "excursive" mode contaminated the other point's naive target),
g = max(Re(all eigenvalues)) is used directly as the surrogate target here,
UNLIKE data_generation.py which had to restrict to the complex-pair-only
real part. This is simpler and still exactly right for this point.
"""

import numpy as np
from scipy.stats import qmc

from tide.continuation.codim2_convergence import _fold
from tide.continuation.pseudo_arclength import _eigvals_jit
from tide.physics.ledinegg_curve import euler_number

# Corrected post-audit double-zero coordinates (PROJECT_LOG.md Sec. 44,
# src/tide/continuation/double_zero.py), converged to residual <1e-11.
# Previously 14.14278/20.59478 -- the pre-audit fold-curve-constrained
# values, which direct spectral evaluation showed were NOT a genuine
# double-zero point (two distinct real eigenvalues +0.0439/-0.0451, not a
# repeated zero). Fixed here at the source so nothing downstream (e.g.
# log_distance_mlp.py's log|Nsub_BT - Nsub| feature) can silently inherit
# the stale value.
NSUB_BT = 14.142794816
NPCH_BT = 20.597778032
FR_BT, LAM_BT, KI_BT, KE_BT = 0.5, 0.001, 11.0, 3.0


def _g(nsub, npch, Fr, Lam, ki, ke, N1):
    eigvals = np.array(_eigvals_jit(nsub, npch, Fr, Lam, ki, ke, N1))
    if not np.all(np.isfinite(eigvals)):
        return None
    return float(np.max(eigvals.real))


def _bisect_g(nsub, lo, hi, Fr, Lam, ki, ke, N1, tol=1e-9, max_iter=60):
    f_lo, f_hi = _g(nsub, lo, Fr, Lam, ki, ke, N1), _g(nsub, hi, Fr, Lam, ki, ke, N1)
    if f_lo is None or f_hi is None:
        return None
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
        f_mid = _g(nsub, mid, Fr, Lam, ki, ke, N1)
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def wedge_boundaries(nsub, Fr, Lam, ki, ke, N1):
    """SUPERSEDED (see PROJECT_LOG.md, post-manuscript-draft external
    audit): this used the Eq. 26 fold value as the `lower` boundary,
    assuming it "indistinguishable from the true real-eigenvalue crossing
    to high precision" (Sec. 37). That assumption is FALSE near the BT
    tip -- confirmed directly: at Point B, Nsub=14.14, the fold sits at
    20.590725, entirely OUTSIDE the true stable window (20.592276,
    20.594954), a ~58% width overstatement. Kept only as a documented
    record of the error, per this project's disclosure practice -- do not
    use for any new result. Use `wedge_boundaries_true` instead."""
    fold = _fold(nsub, Fr, Lam, ki, ke)
    if fold is None:
        return None, None
    for offset in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        upper = _bisect_g(nsub, fold * (1 + offset), fold * 1.6, Fr, Lam, ki, ke, N1)
        if upper is not None:
            return fold, upper
    return fold, None


def _fine_scan_crossings(nsub, lo, hi, Fr, Lam, ki, ke, N1, n_grid):
    npch_grid = np.linspace(lo, hi, n_grid)
    g_vals = [_g(nsub, p, Fr, Lam, ki, ke, N1) for p in npch_grid]
    crossings = []
    for i in range(len(npch_grid) - 1):
        a, b = g_vals[i], g_vals[i + 1]
        if a is None or b is None:
            continue
        if a * b < 0:
            crossings.append((npch_grid[i], npch_grid[i + 1]))
    return crossings


def wedge_boundaries_true(nsub, Fr, Lam, ki, ke, N1, n_grid=4000):
    """(lower, upper): the two TRUE g=0 boundaries of the stable window,
    found by directly scanning for sign changes in g rather than assuming
    the Eq. 26 fold is one of them (see `wedge_boundaries` docstring for
    why that assumption was wrong). The Eq. 26 fold is used only as a
    rough anchor for where to center the search, never as an answer.

    Progressively narrows the search band (and implicitly increases grid
    resolution relative to the window, since n_grid is fixed) around the
    fold anchor, since the window can be far narrower than any single
    fixed search width once close to the BT point -- a fixed-width scan
    that works at moderate distance from the BT point can completely miss
    a sub-0.01-wide window near the tip."""
    fold = _fold(nsub, Fr, Lam, ki, ke)
    if fold is None:
        return None, None
    for radius_frac in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005):
        lo = fold * (1 - radius_frac)
        hi = fold * (1 + max(radius_frac * 4, 0.6))
        crossings = _fine_scan_crossings(nsub, lo, hi, Fr, Lam, ki, ke, N1, n_grid)
        if len(crossings) >= 2:
            crossings = sorted(crossings, key=lambda c: c[0])[:2]
            lower = _bisect_g(nsub, *crossings[0], Fr, Lam, ki, ke, N1, tol=1e-10)
            upper = _bisect_g(nsub, *crossings[1], Fr, Lam, ki, ke, N1, tol=1e-10)
            if lower is not None and upper is not None:
                return lower, upper
    return None, None


def generate_bt_dataset(Fr, Lam, ki, ke, N1, nsub_range, n_nsub=60,
                         n_window_per_nsub=25, n_background_per_nsub=25,
                         window_pad_frac=0.3, background_frac=(1.05, 2.0),
                         seed=0):
    """Generate (Nsub, Npch) -> (Eu, g) training data around the BT wedge.
    Only meaningful for nsub_range entirely below NSUB_BT (above it there is
    no window to sample around, Sec. 37). Sampling deliberately straddles
    the (lower, upper) window boundaries -- same anti-rigging principle as
    data_generation.py, Sec. 5."""
    rng = np.random.default_rng(seed)
    nsub_values = np.linspace(nsub_range[0], nsub_range[1], n_nsub)

    rows_nsub, rows_npch, rows_eu, rows_g = [], [], [], []

    for nsub in nsub_values:
        lower, upper = wedge_boundaries(nsub, Fr, Lam, ki, ke, N1)
        if lower is None or upper is None:
            continue

        npch_samples = []
        for boundary in (lower, upper):
            lo_b, hi_b = boundary * (1 - window_pad_frac), boundary * (1 + window_pad_frac)
            sampler = qmc.LatinHypercube(d=1, seed=rng.integers(0, 2**32 - 1))
            u = sampler.random(n_window_per_nsub).ravel()
            npch_samples.append(lo_b + u * (hi_b - lo_b))

        lo_bg, hi_bg = background_frac[0] * nsub, background_frac[1] * upper
        sampler = qmc.LatinHypercube(d=1, seed=rng.integers(0, 2**32 - 1))
        u = sampler.random(n_background_per_nsub).ravel()
        npch_samples.append(lo_bg + u * (hi_bg - lo_bg))

        npch_values = np.concatenate(npch_samples)
        npch_values = npch_values[npch_values > nsub * 1.001]

        for npch in npch_values:
            g = _g(nsub, npch, Fr, Lam, ki, ke, N1)
            if g is None:
                continue
            eu = float(euler_number(npch, nsub, Fr, Lam, ki, ke))
            rows_nsub.append(nsub)
            rows_npch.append(npch)
            rows_eu.append(eu)
            rows_g.append(g)

    return dict(
        nsub=np.array(rows_nsub), npch=np.array(rows_npch),
        Eu=np.array(rows_eu), g=np.array(rows_g),
    )
