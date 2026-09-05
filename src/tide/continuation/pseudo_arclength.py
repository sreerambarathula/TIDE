"""Curve tracing for the Hopf (DWO) boundary in the (Nsub, Npch) plane, at fixed
channel/friction parameters (Fr, Lambda, k_in, k_out).

Why this exists, not just "more general than curves.py": the existing Hopf finder
(curves.py's hopf_npch / fast_gap_scan's batched version) re-scans a wide Npch
window and picks "the eigenvalue with the largest real part among complex ones" at
each Nsub independently. That is fragile exactly where it matters most -- near
codimension-2 points, where eigenvalues transition between real and complex, that
selection can silently jump to a DIFFERENT eigenvalue pair between one Nsub and
the next (observed directly during the Sec. 15 investigation: erratic values like
-inf and -16 to -19 right where a complex pair was merging into two real
eigenvalues).

Design note -- narrow bisection, not full pseudo-arclength or secant (see
PROJECT_LOG.md Sec. 23 for the debugging history): two earlier attempts (proper
tangent+arclength stepping, then a secant-based natural-parametrization version)
both kept overshooting into NaN or locking onto the wrong eigenvalue branch, each
time confirmed directly by tracing through the failure rather than assumed. This
version keeps the ONE thing that actually matters -- identifying the SAME
eigenvalue continuously from step to step -- but locates it at each step with
plain, robust bisection inside a NARROW bracket centered on the previous point
(not a secant extrapolation, which can jump far on a single bad step). Simpler,
and it is what actually stopped failing.

Safety note (PROJECT_LOG.md Sec. 16): this module only ever evaluates the physics
Jacobian at the CLOSED-FORM steady state (steady_state_general) -- it never leaves
the state x free to be found by an unconstrained Newton solve, so it does not share
the extraneous-branch vulnerability found for the general augmented fold/Hopf
systems in augmented_systems.py.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from tide.physics.clausse_lahey import state_derivative_general, steady_state_general
from tide.physics.ledinegg_curve import euler_number


@functools.partial(jax.jit, static_argnames=("N1",))
def _eigvals_jit(nsub, npch, Fr, Lam, ki, ke, N1):
    """Compiled eigenvalue spectrum at the closed-form steady state. MUST stay
    defined at module level with Fr/Lam/ki/ke/nsub/npch as genuine traced
    arguments (N1 static, since it drives Python-level loop unrolling in
    state_derivative_general/steady_state_general) -- this is the exact same bug
    class documented twice already (PROJECT_LOG.md Sec. 17, 21): the original
    version of this module built `f = lambda x: ...` and called jax.jacfwd(f)(x0)
    directly inside the untraced per-point wrapper, with no @jax.jit at all, so
    every one of the ~60-85 evaluations per bisected point ran through eager
    JAX dispatch from scratch. Confirmed directly: 21 traced points took 168s
    (~8s/point) before this fix (see the background timing run this fix is
    checked against)."""
    Eu = euler_number(npch, nsub, Fr, Lam, ki, ke)
    x0 = steady_state_general(nsub, npch, N1)
    params = dict(Nsub=nsub, Npch=npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)
    f = lambda x: state_derivative_general(x, params, N1)
    J = jax.jacfwd(f)(x0)
    return jnp.linalg.eigvals(J)


def _jacobian_eigenvalues(nsub, npch, Fr, Lam, ki, ke, N1):
    """Full eigenvalue spectrum at the closed-form steady state, or None if the
    point is outside the model's valid domain (e.g. npch <= nsub, or the
    algebraic solve inside the physics model fails) -- NaN-safe by construction,
    since bisection needs to handle "invalid point" gracefully, not crash."""
    if npch <= nsub:
        return None
    try:
        eigvals = np.array(_eigvals_jit(nsub, npch, Fr, Lam, ki, ke, N1))
        if not np.all(np.isfinite(eigvals)):
            return None
        return eigvals
    except Exception:
        return None


def _tracked_real_part(nsub, npch, target, Fr, Lam, ki, ke, N1):
    """Real part of the eigenvalue closest to `target`, or None if the point is
    invalid. Following a specific root continuously (nearest to the last known
    value) rather than re-picking "the biggest" fresh at every point is what
    prevents branch-jumping."""
    eigvals = _jacobian_eigenvalues(nsub, npch, Fr, Lam, ki, ke, N1)
    if eigvals is None:
        return None, None
    idx = np.argmin(np.abs(eigvals - target))
    return eigvals[idx], eigvals[idx].real


def _bisect_npch(nsub, npch_center, target, Fr, Lam, ki, ke, N1, tol=1e-9, max_iter=60):
    """Find npch near npch_center where the tracked eigenvalue's real part is
    zero, via bisection inside a bracket built by widening OUTWARD from
    npch_center in small relative steps -- narrow and centered on the previous
    point, so the same eigenvalue branch stays identifiable throughout."""
    ev_center, g_center = _tracked_real_part(nsub, npch_center, target, Fr, Lam, ki, ke, N1)
    if ev_center is None:
        return None, None

    lo = hi = npch_center
    g_lo = g_hi = g_center
    ev_lo = ev_hi = ev_center
    for widen in [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]:
        lo = npch_center * (1 - widen)
        hi = npch_center * (1 + widen)
        ev_lo, g_lo = _tracked_real_part(nsub, lo, ev_center, Fr, Lam, ki, ke, N1)
        ev_hi, g_hi = _tracked_real_part(nsub, hi, ev_center, Fr, Lam, ki, ke, N1)
        if g_lo is not None and g_hi is not None and g_lo * g_hi < 0:
            break
    else:
        return None, None  # no sign change found in any tried bracket

    for _ in range(max_iter):
        if hi - lo < tol:
            break  # early exit once the bracket is already tighter than tol --
            # the loop used to always run the full max_iter=60 regardless (a
            # second, independent source of wasted work on top of the missing
            # JIT compilation); bisection converges to the same midpoint either
            # way, just without the wasted iterations once converged.
        mid = 0.5 * (lo + hi)
        ev_mid, g_mid = _tracked_real_part(nsub, mid, ev_lo, Fr, Lam, ki, ke, N1)
        if g_mid is None:
            return None, None
        if g_lo * g_mid < 0:
            hi, g_hi = mid, g_mid
        else:
            lo, g_lo, ev_lo = mid, g_mid, ev_mid
    return 0.5 * (lo + hi), ev_lo


def trace_hopf_curve(nsub0, npch0, Fr, Lam, ki, ke, N1, dnsub=0.2, n_steps=100, direction=1.0):
    """Trace the Hopf curve by stepping Nsub in small increments and bisecting
    for Npch (re-identifying the tracked eigenvalue) at each step, seeded from
    the previous point. Returns a list of dicts with nsub, npch, omega (the
    tracked eigenvalue's imaginary part), and the eigenvalue itself.

    `direction`: +1.0 to increase Nsub, -1.0 to decrease it.
    """
    eigvals0 = _jacobian_eigenvalues(nsub0, npch0, Fr, Lam, ki, ke, N1)
    if eigvals0 is None:
        raise ValueError("starting point is outside the model's valid domain")
    complex0 = eigvals0[eigvals0.imag > 1e-9]
    if len(complex0) == 0:
        raise ValueError("no complex eigenvalue found at the starting point")
    target = complex0[np.argmin(np.abs(complex0.real))]

    npch0, target = _bisect_npch(nsub0, npch0, target, Fr, Lam, ki, ke, N1)
    if npch0 is None:
        raise ValueError("could not snap the starting point onto the Hopf curve")

    results = [dict(nsub=nsub0, npch=npch0, omega=target.imag, eigenvalue=target)]

    nsub, npch, ev = nsub0, npch0, target
    for _ in range(n_steps):
        nsub_next = nsub + direction * dnsub
        npch_next, ev_next = _bisect_npch(nsub_next, npch, ev, Fr, Lam, ki, ke, N1)
        if npch_next is None or abs(ev_next.imag) < 1e-9:
            break  # curve ended (ran off the domain, or the pair went real)
        nsub, npch, ev = nsub_next, npch_next, ev_next
        results.append(dict(nsub=nsub, npch=npch, omega=ev.imag, eigenvalue=ev))

    return results
