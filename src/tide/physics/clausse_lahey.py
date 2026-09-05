"""Clausse-Lahey (1991) moving boiling-boundary model, single two-phase node
(N1=1 single-phase node too -- the minimal case), in the rho_e-based reduced form
that eliminates the enthalpy-slope variable eta.

Source, verified by direct page-image transcription: G. Theler, A. Clausse,
F. Bonetto, "The moving boiling-boundary model of a vertical two-phase flow channel
revisited," Mecanica Computacional Vol. XXIX, pp. 3949-3976 (2010), Eq. (42), p. 3970.

State vector x = (lambda, m, u_i):
  lambda : boiling boundary position (dimensionless, 0-1)
  m      : total non-dimensional fluid mass in the channel
  u_i    : inlet velocity (dimensionless)

rho_e (exit density) is algebraic, slaved to (lambda, m) via Eq. (41):
  (1 - lambda) * ln(1/rho_e) / (1/rho_e - 1) = m - lambda
solved by a short differentiable Newton iteration on r = 1/rho_e.
"""

import jax
import jax.numpy as jnp


def _solve_r(lam, m, iters=60):
    """Solve (1-lam)*ln(r)/(r-1) = m-lam for r=1/rho_e > 1, via Newton's method.
    Differentiable: JAX autodiffs straight through the unrolled iteration."""
    target = m - lam

    def g(r):
        return (1.0 - lam) * jnp.log(r) / (r - 1.0) - target

    def dg_dr(r):
        return (1.0 - lam) * ((1.0 / r) * (r - 1.0) - jnp.log(r)) / (r - 1.0) ** 2

    r = jnp.asarray(2.0, dtype=jnp.float64)  # generic starting guess, r>1

    def body(r, _):
        r_new = r - g(r) / dg_dr(r)
        return jnp.clip(r_new, 1.0 + 1e-9, jnp.inf), None

    r, _ = jax.lax.scan(body, r, xs=None, length=iters)
    return r


def state_derivative(x, params):
    """f(x) = xdot for x = (lambda, m, u_i), N1=1. params is a dict with
    Nsub, Npch, Fr, Lam, ki, ke, Eu.

    NOTE: N1=1 is ODD. The source paper explicitly warns (citing Garea 1998):
    "if N1 is odd, the model has a mathematical pathology that is avoided by using
    an even number of nodes." Confirmed empirically: trajectories integrated from
    this N1=1 model diverge once oscillation amplitude grows past the linear regime
    (see PROJECT_LOG.md Sec. 13/14) -- the Jacobian at the fixed point is still
    correct (linearization doesn't see node-count pathologies), but the nonlinear
    trajectory is not trustworthy. Kept here for the steady-state/Jacobian checks
    that already validated cleanly; use `state_derivative_n1_2` for any nonlinear
    time-domain integration.
    """
    lam, m, ui = x

    # node equation, N1=1: 0 = 0.5*(0 + lam_dot) + 1*(lam-0) - ui  =>  lam_dot = 2(ui-lam)
    lam_dot = 2.0 * (ui - lam)

    m_dot, ui_dot = _two_phase_derivatives(lam, lam_dot, m, ui, params)

    return jnp.array([lam_dot, m_dot, ui_dot])


def steady_state(Nsub, Npch):
    """Analytical steady state (lambda0, m0, ui0) from Eqs. (14),(16),(23),(41).

    log(r0)/(r0-1) has a removable singularity at r0=1 (i.e. Nsub=Npch,
    lam0=1): naive evaluation gives 0/0 = NaN (found via external audit:
    steady_state(8,8) returned [1, nan, 1]). The true limit as r0->1 is 1
    (standard log(x)/(x-1) -> 1 limit), so m0 -> lam0 there -- consistent
    with the lam0=ui0=1 pattern already returned in that case. Uses the
    "safe-input" where-trick (substitute a safe value INTO the risky branch
    before evaluating it, not just select it out afterward) since this
    project differentiates through physics functions via jax.jacfwd
    elsewhere -- a plain jnp.where over an already-NaN branch gives a
    correct forward value but can still produce a NaN gradient."""
    lam0 = Nsub / Npch
    ui0 = Nsub / Npch
    rho_e0 = 1.0 / (1.0 + Npch * (1.0 - lam0))
    r0 = 1.0 / rho_e0
    near_one = jnp.abs(r0 - 1.0) < 1e-9
    r0_safe = jnp.where(near_one, 2.0, r0)
    log_ratio = jnp.where(near_one, 1.0, jnp.log(r0_safe) / (r0_safe - 1.0))
    m0 = lam0 + (1.0 - lam0) * log_ratio
    return jnp.array([lam0, m0, ui0])


# ---------------------------------------------------------------------------
# N1=2 variant: single-phase zone split into two moving nodes (l1, lambda=l2).
#
# N1=1 (above) is ODD, and the source paper explicitly warns (citing Garea 1998,
# p. 3965): "if N1 is odd, the model has a mathematical pathology that is avoided
# by using an even number of nodes." N1=2 avoids that pathology by construction.
# The two-phase side (momentum equation, rho_e algebra) is completely unchanged --
# only the single-phase node equations (Eq. 42's first line, for n=1,2) change.
# ---------------------------------------------------------------------------


def state_derivative_n1_2(x, params):
    """f(x) = xdot for x = (l1, lambda, m, u_i), N1=2."""
    l1, lam, m, ui = x

    # Node equations, N1=2 (Eq. 42, n=1,2; l0=0 constant):
    #   n=1: 0 = 0.5*(0 + l1_dot) + 2*(l1-0) - ui       => l1_dot = 2*ui - 4*l1
    #   n=2: 0 = 0.5*(l1_dot + lam_dot) + 2*(lam-l1) - ui => lam_dot = 2*ui - 4*(lam-l1) - l1_dot
    l1_dot = 2.0 * ui - 4.0 * l1
    lam_dot = 2.0 * ui - 4.0 * (lam - l1) - l1_dot

    # Everything else (two-phase side) is identical in form to the N1=1 case --
    # reuse it by calling the shared momentum/mass machinery with this lambda_dot.
    m_dot, ui_dot = _two_phase_derivatives(lam, lam_dot, m, ui, params)

    return jnp.array([l1_dot, lam_dot, m_dot, ui_dot])


def _two_phase_derivatives(lam, lam_dot, m, ui, params):
    """Mass and momentum equations (Eq. 42, lines 2 onward) -- shared between the
    N1=1 and N1=2 (and any N1) variants, since the two-phase side only ever sees
    lambda (=the boiling boundary, the last single-phase node) and its lambda_dot."""
    Nsub, Npch, Fr, Lam, ki, ke, Eu = (
        params["Nsub"], params["Npch"], params["Fr"], params["Lam"],
        params["ki"], params["ke"], params["Eu"],
    )

    r = _solve_r(lam, m)
    rho_e = 1.0 / r
    ue = ui + Nsub * (1.0 - lam)

    m_dot = ui - rho_e * ue

    def r_of(lam_, m_):
        return _solve_r(lam_, m_)

    _, r_dot = jax.jvp(r_of, (lam, m), (lam_dot, m_dot))
    rho_e_dot = -r_dot / r**2

    r1 = 1.0 / rho_e - 1.0

    bracket = Nsub / r1 * (
        -(1.0 - lam) * m_dot - (1.0 - m) * lam_dot
        + (1.0 - lam) * (1.0 - m) / (r1 * rho_e**2) * rho_e_dot
    )

    friction = Lam * (
        m * ui**2 + 0.5 * (Nsub**2 / Npch) * lam**2
        + (ue - ui) / r1 * (
            (m - lam) * ((1.0 - lam) / r1 * Nsub - 2.0 * ui)
            + lam * Nsub * ((1.0 - lam) / r1 - 1.0)
            + Nsub * (0.5 - (1.0 - lam) / r1)
            + 2.0 * ui * (1.0 - lam)
        )
    )

    rest = rho_e * ue**2 - ui**2 + m / Fr - Eu + ki * ui**2 + ke * rho_e * ue**2

    ui_dot = -(ui * m_dot + bracket + rest + friction) / m

    return m_dot, ui_dot


def steady_state_n1_2(Nsub, Npch):
    """Analytical steady state (l1_0, lambda0, m0, ui0). At steady state the
    single-phase nodes are evenly spaced in z (linear enthalpy profile), so
    l1_0 = lambda0 * (1/N1) = lambda0/2."""
    lam0, m0, ui0 = steady_state(Nsub, Npch)
    l1_0 = lam0 / 2.0
    return jnp.array([l1_0, lam0, m0, ui0])


# ---------------------------------------------------------------------------
# General N1 (even) variant: single-phase zone split into N1 moving nodes
# (l_1, ..., l_{N1-1}, lambda=l_{N1}). Generalizes state_derivative_n1_2, which
# is kept as a separate, independently-validated special case rather than
# rewritten in terms of this general version -- N1=2 already has machine-
# precision cross-validation against Eq. 26 (PROJECT_LOG.md Sec. 13) and a
# published-trajectory match (Sec. 14); that validation should stay pinned to
# fixed, hand-written code, not silently inherit whatever this general version
# does. Use this for N1=4 and above.
#
# Motivation (PROJECT_LOG.md Sec. 20): a realistic-parameter codim-2 point was
# found in a published natural-circulation model (Pandey & Singh 2017) that
# this project's N1=2 model could not reproduce at realistic friction. N1=4 is
# the cheaper of two suspects to check (the other being missing riser/downcomer
# physics) before concluding the model needs deeper structural changes.
# ---------------------------------------------------------------------------


def state_derivative_general(x, params, N1):
    """f(x) = xdot for x = (l_1, ..., l_{N1-1}, lambda, m, u_i), general even N1.

    Node equations (Eq. 42, n=1..N1, l_0=0 fixed, l_{N1}=lambda):
        0 = 0.5*(l_{n-1}_dot + l_n_dot) + N1*(l_n - l_{n-1}) - u_i
        => l_n_dot = 2*u_i - 2*N1*(l_n - l_{n-1}) - l_{n-1}_dot
    a forward recurrence starting from l_0_dot = 0 (l_0 = 0 is a fixed constant,
    the channel inlet -- not a state).

    N1 must be even -- an odd node count hits the documented mathematical
    pathology (Sec. 14/PROJECT_LOG.md, citing Garea 1998) this general
    formulation exists specifically to avoid; found via external audit
    that this was accepted silently rather than rejected."""
    if N1 % 2 != 0:
        raise ValueError(
            f"state_derivative_general requires an even N1 (odd N1 has a "
            f"documented mathematical pathology, PROJECT_LOG.md Sec. 14); "
            f"got N1={N1}. Use state_derivative (N1=1) or "
            f"state_derivative_n1_2 (N1=2) if you specifically need an odd "
            f"or minimal case as a cross-check, not this general path."
        )
    ui = x[-1]
    node_positions = x[:N1]  # [l_1, ..., l_{N1-1}, lambda]  (lambda = l_{N1})
    m = x[-2]

    l_prev = 0.0
    l_dot_prev = 0.0
    node_dots = []
    for n in range(N1):
        l_n = node_positions[n]
        l_n_dot = 2.0 * ui - 2.0 * N1 * (l_n - l_prev) - l_dot_prev
        node_dots.append(l_n_dot)
        l_prev = l_n
        l_dot_prev = l_n_dot

    lam = node_positions[-1]
    lam_dot = node_dots[-1]

    m_dot, ui_dot = _two_phase_derivatives(lam, lam_dot, m, ui, params)

    return jnp.array(node_dots + [m_dot, ui_dot])


def steady_state_general(Nsub, Npch, N1):
    """Analytical steady state: single-phase nodes evenly spaced in z at steady
    state (linear enthalpy profile), l_n_0 = lambda_0 * (n/N1).

    N1 must be even -- see state_derivative_general's docstring."""
    if N1 % 2 != 0:
        raise ValueError(
            f"steady_state_general requires an even N1; got N1={N1}. "
            f"See state_derivative_general's docstring."
        )
    lam0, m0, ui0 = steady_state(Nsub, Npch)
    node_positions_0 = [lam0 * (n / N1) for n in range(1, N1 + 1)]
    return jnp.array(node_positions_0 + [m0, ui0])
