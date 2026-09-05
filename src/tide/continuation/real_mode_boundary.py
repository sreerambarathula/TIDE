"""A third instability boundary, found by accident while building Phase 3 data
(PROJECT_LOG.md Sec. 27): a persistent REAL (non-oscillatory) eigenvalue of the
Clausse-Lahey Jacobian that dominates the spectrum at high Nsub, independent of
N1 (checked at N1=2 through 16) and not specific to low-Froude parameters
(checked Fr=0.035 through 3.0). Confirmed genuine, not a bug, by direct
nonlinear trajectory integration from a perturbed steady state (grows
monotonically, matching the eigenvalue's sign and magnitude) -- see the
PROJECT_LOG entry for the full check.

This is NOT the fold (Sec. 16: the fold is a non-invertibility of the
algebraic Npch->Eu map, invisible to the ODE Jacobian at fixed (Nsub, Npch))
and NOT the Hopf (that's a complex-pair crossing; this is real). It appears to
be a genuine additional instability mode of the full ODE system: past some
Nsub_crit(Fr, Lam, ki, ke), NO Npch value keeps the (unique, closed-form)
steady state linearly stable at all -- confirmed directly, not inferred, by
scanning densely over Npch and finding the minimum eigenvalue real part stays
positive.

Why this matters for codim-2 point selection: a fold-Hopf(complex) crossing at
Nsub* > Nsub_crit is mathematically real but physically moot -- the system was
never stable there to begin with, for any Npch. A codim-2 point is only a
meaningful "safe vs. unsafe operating boundary" if Nsub* < Nsub_crit for the
same (Fr, Lam, ki, ke).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from tide.continuation.pseudo_arclength import _eigvals_jit


@functools.partial(jax.jit, static_argnames=("N1",))
def _max_real_eig_batch(nsub, npch_batch, Fr, Lam, ki, ke, N1):
    """max(Re(eigenvalues)) (over ALL eigenvalues, not just complex ones) at a
    batch of Npch values, fixed Nsub. NaN-safe: returns +inf for any Npch where
    the eigendecomposition isn't finite (e.g. Npch<=Nsub), so it never looks
    artificially stable due to a domain error."""

    def one(npch):
        eigvals = _eigvals_jit(nsub, npch, Fr, Lam, ki, ke, N1)
        ok = jnp.all(jnp.isfinite(eigvals))
        return jnp.where(ok, jnp.max(eigvals.real), jnp.inf)

    return jax.vmap(one)(npch_batch)


def best_case_stability_margin(nsub, Fr, Lam, ki, ke, N1, npch_grid_points=200, npch_mult=4.0):
    """min over Npch of max(Re(eigenvalues)) at this Nsub -- the best-case
    linear stability achievable at ANY Npch. Negative means some Npch is
    stable; positive means NO Npch is stable at this Nsub (Nsub is past the
    real-mode threshold)."""
    npch_grid = jnp.linspace(nsub * 1.02, nsub * npch_mult, npch_grid_points)
    margins = _max_real_eig_batch(nsub, npch_grid, Fr, Lam, ki, ke, N1)
    return float(jnp.min(margins))


def find_nsub_crit(Fr, Lam, ki, ke, N1, nsub_bracket, tol=1e-3, max_iter=40, **scan_kwargs):
    """Bisect for the Nsub where best_case_stability_margin crosses zero --
    the real-mode instability threshold. Returns None if the margin doesn't
    change sign across nsub_bracket (either always stable-capable or always
    not, in that range)."""
    lo, hi = nsub_bracket
    f_lo = best_case_stability_margin(lo, Fr, Lam, ki, ke, N1, **scan_kwargs)
    f_hi = best_case_stability_margin(hi, Fr, Lam, ki, ke, N1, **scan_kwargs)
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        f_mid = best_case_stability_margin(mid, Fr, Lam, ki, ke, N1, **scan_kwargs)
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
