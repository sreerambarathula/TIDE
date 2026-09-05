"""JIT+vmap batched version of the fold-Hopf gap scan (see curves.py for the
original, correctness-first but very slow implementation -- ~1s per fold_npch call
and ~0.1-0.4s per Hopf eigenvalue evaluation, unbatched, made a systematic sweep
over many (Nsub, Lambda) combinations take hours instead of seconds; see
PROJECT_LOG.md Sec. 17).

This module trades the grid-search-then-bisect approach for direct, fully
vectorized dense grids evaluated in one JIT-compiled batch call per Lambda value --
correctness comes from grid resolution, not iterative refinement, which is fine for
a scanning/screening tool (not for final precise root locations -- use curves.py's
Newton-refined functions for that once a promising region is found here).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from tide.physics.clausse_lahey import state_derivative_n1_2, state_derivative_general
from tide.physics.ledinegg_curve import euler_number


@jax.jit
def _fold_npch_for_nsub(nsub, Fr, Lam, ki, ke, npch_grid):
    """Fold Npch via dense-grid argmax of Eu(Npch), fully vectorized."""
    eu = jax.vmap(lambda p: euler_number(p, nsub, Fr, Lam, ki, ke))(npch_grid)
    imax = jnp.argmax(eu)
    is_interior = (imax > 0) & (imax < npch_grid.shape[0] - 1)
    return jnp.where(is_interior, npch_grid[imax], jnp.nan)


@jax.jit
def _leading_complex_real_part_batch(nsub, npch_batch, Fr, Lam, ki, ke, Eu_batch):
    """Vectorized version of curves._leading_complex_real_part over a batch of
    npch values (with matching Eu values, since Eu depends on npch)."""

    def one(npch, Eu):
        lam0 = nsub / npch
        ui0 = nsub / npch
        rho_e0 = 1.0 / (1.0 + npch * (1.0 - lam0))
        r0 = 1.0 / rho_e0
        m0 = lam0 + (1.0 - lam0) * jnp.log(r0) / (r0 - 1.0)
        x0 = jnp.array([lam0 / 2.0, lam0, m0, ui0])
        params = dict(Nsub=nsub, Npch=npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)
        f = lambda x: state_derivative_n1_2(x, params)
        J = jax.jacfwd(f)(x0)
        eigvals = jnp.linalg.eigvals(J)
        is_complex = jnp.abs(eigvals.imag) > 1e-9
        real_parts = jnp.where(is_complex, eigvals.real, -jnp.inf)
        return jnp.max(real_parts)

    return jax.vmap(one)(npch_batch, Eu_batch)


@functools.partial(jax.jit, static_argnames=("N1",))
def _leading_complex_real_part_batch_general(nsub, npch_batch, Fr, Lam, ki, ke, Eu_batch, N1):
    """Same as _leading_complex_real_part_batch but for general even N1, using
    state_derivative_general.

    MUST be @jax.jit'd at MODULE LEVEL with Fr/Lam/ki/ke as genuine traced
    arguments (not closed over inside an inner, un-jitted function) -- an earlier
    version created a fresh `jax.jit(...)` object *inside* the function body on
    every call, with Fr/Lam/ki/ke baked in as plain Python constants (since the
    outer function itself was untraced). That forced a full XLA recompilation on
    every single (Fr, Lam, ki, ke) combination in a sweep -- confirmed directly:
    an 875-combination N1=4 search that should have taken ~2-3 minutes (by
    analogy with N1=2's 83s) was still running after 27 minutes, burning ~333%
    CPU continuously (genuinely computing, not stuck) purely on repeated
    recompilation. See PROJECT_LOG.md Sec. 21. N1 is marked static (it controls
    Python-level loop unrolling and array shapes, so it genuinely can't be
    traced) -- JAX compiles once per distinct N1 value and reuses across every
    (Fr, Lam, ki, ke, Nsub) combination, exactly like the N1=2 version already did."""

    def one(npch, Eu):
        lam0 = nsub / npch
        ui0 = nsub / npch
        node_positions_0 = jnp.array([lam0 * (n / N1) for n in range(1, N1 + 1)])
        rho_e0 = 1.0 / (1.0 + npch * (1.0 - lam0))
        r0 = 1.0 / rho_e0
        m0 = lam0 + (1.0 - lam0) * jnp.log(r0) / (r0 - 1.0)
        x0 = jnp.concatenate([node_positions_0, jnp.array([m0, ui0])])
        params = dict(Nsub=nsub, Npch=npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)
        f = lambda x: state_derivative_general(x, params, N1)
        J = jax.jacfwd(f)(x0)
        eigvals = jnp.linalg.eigvals(J)
        is_complex = jnp.abs(eigvals.imag) > 1e-9
        real_parts = jnp.where(is_complex, eigvals.real, -jnp.inf)
        return jnp.max(real_parts)

    return jax.vmap(one)(npch_batch, Eu_batch)


def gap_scan_general(Fr, Lam, ki, ke, nsub_values, N1, npch_grid_points=4000, hopf_scan_points=60):
    """Same as gap_scan but using the general-N1 Hopf detector. Fold detection is
    UNCHANGED regardless of N1 -- the fold is a property of the momentum/mass
    balance (Eq. 26) alone, independent of single-phase nodalization (established
    in PROJECT_LOG.md Sec. 15-16)."""
    results = []
    for nsub in nsub_values:
        npch_grid_fold = jnp.linspace(nsub * 1.02, nsub * 6.0, npch_grid_points)
        fold = float(_fold_npch_for_nsub(nsub, Fr, Lam, ki, ke, npch_grid_fold))

        npch_scan = jnp.linspace(nsub * 1.05, nsub * 6.0, hopf_scan_points)
        eu_scan = jax.vmap(lambda p: euler_number(p, nsub, Fr, Lam, ki, ke))(npch_scan)
        re_scan = np.array(
            _leading_complex_real_part_batch_general(nsub, npch_scan, Fr, Lam, ki, ke, eu_scan, N1)
        )
        npch_scan_np = np.array(npch_scan)

        hopf = None
        for i in range(len(npch_scan_np) - 1):
            a, b = re_scan[i], re_scan[i + 1]
            if np.isfinite(a) and np.isfinite(b) and a < 0 < b:
                hopf = npch_scan_np[i] + (0 - a) * (npch_scan_np[i + 1] - npch_scan_np[i]) / (b - a)
                break

        gap = (hopf - fold) if (hopf is not None and np.isfinite(fold)) else None
        results.append((nsub, fold if np.isfinite(fold) else None, hopf, gap))
    return results


def gap_scan(Fr, Lam, ki, ke, nsub_values, npch_grid_points=4000, hopf_scan_points=60):
    """For each Nsub in nsub_values, find the fold Npch (dense grid argmax) and the
    Hopf Npch (dense grid scan for the sign change of the leading complex
    eigenvalue's real part, refined by linear interpolation -- not a full Newton
    bisection, adequate for screening). Returns arrays (nsub, fold, hopf, gap)."""
    results = []
    for nsub in nsub_values:
        npch_grid_fold = jnp.linspace(nsub * 1.02, nsub * 6.0, npch_grid_points)
        fold = float(_fold_npch_for_nsub(nsub, Fr, Lam, ki, ke, npch_grid_fold))

        npch_scan = jnp.linspace(nsub * 1.05, nsub * 6.0, hopf_scan_points)
        eu_scan = jax.vmap(lambda p: euler_number(p, nsub, Fr, Lam, ki, ke))(npch_scan)
        re_scan = np.array(
            _leading_complex_real_part_batch(nsub, npch_scan, Fr, Lam, ki, ke, eu_scan)
        )
        npch_scan_np = np.array(npch_scan)

        hopf = None
        for i in range(len(npch_scan_np) - 1):
            a, b = re_scan[i], re_scan[i + 1]
            if np.isfinite(a) and np.isfinite(b) and a < 0 < b:
                # linear interpolation for a quick refined estimate
                hopf = npch_scan_np[i] + (0 - a) * (npch_scan_np[i + 1] - npch_scan_np[i]) / (b - a)
                break

        gap = (hopf - fold) if (hopf is not None and np.isfinite(fold)) else None
        results.append((nsub, fold if np.isfinite(fold) else None, hopf, gap))
    return results
