"""Fold (Ledinegg) and Hopf (DWO) boundary curves in the (Nsub, Npch) plane, and
their intersection -- the codimension-2 point.

Fold: N_pch,fold(N_sub) is found from the critical point of the closed-form Euler
number relation (ledinegg_curve.euler_number, Eq. 26 of Theler/Clausse/Bonetto
2010) -- dEu/dNpch = 0. This is a property of the steady-state momentum balance
alone; it does NOT require the full ODE system, and does NOT show up as a zero
eigenvalue of the N1=2 Jacobian at fixed Npch (verified empirically -- see
PROJECT_LOG.md Sec. 15: at fixed Npch, Nsub the steady state is a single-valued
function, so there is no degeneracy for the Jacobian to detect; the fold is a
non-invertibility of the Npch-to-Eu map, not of the dynamics at a fixed point).

Hopf: N_pch,Hopf(N_sub) is found from the N1=2 Clausse-Lahey ODE Jacobian
(clausse_lahey.state_derivative_n1_2) -- the Npch where the leading
complex-conjugate eigenvalue pair's real part crosses zero. This is a genuine
eigenvalue-crossing condition, validated against the source paper's own example
(PROJECT_LOG.md Sec. 13-14).
"""

import jax
import jax.numpy as jnp
import numpy as np

from tide.physics.clausse_lahey import state_derivative_n1_2, steady_state_n1_2
from tide.physics.ledinegg_curve import euler_number


def fold_npch(n_sub, Fr, Lam, ki, ke, search_mult=3.0, grid_points=8000,
              tol=1e-12, max_iter=100):
    """Find the fold (local max of Eu(Npch) at fixed Nsub) by first locating it on
    a dense grid, then refining with bisection on dEu/dNpch=0.

    NOTE: an earlier version took a caller-supplied bracket and bisected directly.
    That gave a non-monotonic, implausible fold curve (see PROJECT_LOG.md Sec. 15)
    -- a badly-seeded bracket converged to a spurious root of the derivative that
    wasn't the actual visible extremum. Confirmed by direct dense-grid inspection
    that only one extremum exists per Nsub in this regime; do not reintroduce a
    caller-supplied bracket without re-verifying against a grid scan first.
    Returns None if no interior maximum exists (i.e. no fold at this Nsub --
    confirmed to genuinely happen below Nsub~7-7.5 for Fr=5,Lam=3,ki=6,ke=2)."""
    npch_grid = np.linspace(n_sub * 1.02, n_sub * search_mult, grid_points)
    eu_grid = np.array([
        float(euler_number(p, n_sub, Fr, Lam, ki, ke)) for p in npch_grid
    ])
    imax = int(np.argmax(eu_grid))
    if imax == 0 or imax == len(npch_grid) - 1:
        return None  # no interior maximum -> no fold in this search range

    dEu = jax.grad(lambda npch: euler_number(npch, n_sub, Fr, Lam, ki, ke))
    lo, hi = npch_grid[imax - 1], npch_grid[imax + 1]
    f_lo, f_hi = float(dEu(lo)), float(dEu(hi))
    if f_lo * f_hi > 0:
        return npch_grid[imax]  # grid resolution was already tight enough

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = float(dEu(mid))
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _leading_complex_real_part(n_sub, npch, Fr, Lam, ki, ke):
    """Max real part among eigenvalue(s) of the N1=2 Jacobian with nonzero
    imaginary part, at the steady state for (n_sub, npch). Returns -inf if no
    complex pair exists (shouldn't happen for this 4-state system away from
    codim-2 points, but guarded)."""
    Eu = float(euler_number(npch, n_sub, Fr, Lam, ki, ke))
    x0 = steady_state_n1_2(n_sub, npch)
    params = dict(Nsub=n_sub, Npch=npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)
    f = lambda x: state_derivative_n1_2(x, params)
    J = np.array(jax.jacfwd(f)(x0))
    eigvals = np.linalg.eigvals(J)
    complex_eigs = eigvals[np.abs(eigvals.imag) > 1e-9]
    if len(complex_eigs) == 0:
        return -np.inf
    return float(np.max(complex_eigs.real))


def hopf_npch(n_sub, Fr, Lam, ki, ke, bracket, tol=1e-8, max_iter=60):
    """Root-find the leading complex pair's real part = 0, by bisection on Npch
    within `bracket`. Caller must supply a bracket known to contain a sign change
    (stable -> unstable) -- this is a 1D root-find, not a search."""
    lo, hi = bracket
    f_lo = _leading_complex_real_part(n_sub, lo, Fr, Lam, ki, ke)
    f_hi = _leading_complex_real_part(n_sub, hi, Fr, Lam, ki, ke)
    # An exact (or near-exact) root at either endpoint must be handled
    # explicitly: f_lo*f_mid<0 is never true if f_lo==0, so the loop below
    # would silently walk lo all the way to hi instead of returning the
    # endpoint root (found via external audit, PROJECT_LOG.md).
    if abs(f_lo) < tol:
        return lo
    if abs(f_hi) < tol:
        return hi
    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _leading_complex_real_part(n_sub, mid, Fr, Lam, ki, ke)
        if abs(f_mid) < tol or (hi - lo) < 1e-8:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
