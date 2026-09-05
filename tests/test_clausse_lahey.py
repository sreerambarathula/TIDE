"""Validate the Clausse-Lahey moving boiling-boundary ODE (Theler, Clausse &
Bonetto 2010, Eq. 42, N1=1 case) against:
1. Self-consistency with the independently-transcribed Eq. 26 steady state (machine
   precision -- these come from different sections of the same paper).
2. The paper's own claimed instability at their Fig. 5/6 example point, via linear
   stability (Jacobian eigenvalues), not by reproducing their nonlinear simulation.
"""

import jax
import jax.numpy as jnp
import numpy as np

import tide  # noqa: F401
from tide.physics.clausse_lahey import (
    state_derivative,
    state_derivative_n1_2,
    steady_state,
    steady_state_n1_2,
)
from tide.physics.ledinegg_curve import euler_number

# Paper's Fig. 5/6 example parameters (Nsub, Npch, Fr, Lambda, ki, ke).
# NOTE: the paper's own caption states Eu=9.4987 "according to equation (26)", but
# our Eq. 26 (already validated against the paper's Fig. 3 elsewhere) gives 9.1376
# for these inputs -- see PROJECT_LOG.md Sec. 13. Use our computed Eu, not theirs.
NSUB, NPCH, FR, LAM, KI, KE = 6.5, 14.0, 1.0, 3.0, 6.0, 2.0


def test_steady_state_matches_eq26_to_machine_precision():
    Eu = float(euler_number(NPCH, NSUB, FR, LAM, KI, KE))
    x0 = steady_state(NSUB, NPCH)
    params = dict(Nsub=NSUB, Npch=NPCH, Fr=FR, Lam=LAM, ki=KI, ke=KE, Eu=Eu)
    f0 = np.array(state_derivative(x0, params))
    assert np.max(np.abs(f0)) < 1e-10, f"steady-state residual too large: {f0}"


def test_jacobian_predicts_unstable_hopf_focus():
    # This is the actual falsifiable check: does our from-scratch Jacobian predict
    # the same instability the paper demonstrates by direct time-domain simulation
    # (their Fig. 5: growing oscillation; Fig. 6: unstable focus + stable limit cycle)?
    Eu = float(euler_number(NPCH, NSUB, FR, LAM, KI, KE))
    x0 = steady_state(NSUB, NPCH)
    params = dict(Nsub=NSUB, Npch=NPCH, Fr=FR, Lam=LAM, ki=KI, ke=KE, Eu=Eu)

    f = lambda x: state_derivative(x, params)
    J = np.array(jax.jacfwd(f)(x0))
    eigvals = np.linalg.eigvals(J)

    complex_eigs = eigvals[np.abs(eigvals.imag) > 1e-6]
    assert len(complex_eigs) == 2, "expected a complex-conjugate pair"
    assert np.all(complex_eigs.real > 0), (
        f"expected an unstable (Re>0) oscillatory focus, got {complex_eigs}"
    )


def test_n1_1_odd_node_pathology_causes_blowup():
    """Documents the actual root cause found by debugging, not just asserts a fix.
    N1=1 is odd; the source paper (citing Garea 1998) explicitly warns odd N1 has
    "a mathematical pathology." Confirmed empirically: integrating the N1=1 model
    from a small perturbation at the paper's own Fig. 5/6 example point diverges to
    NaN well before the paper's oscillation would saturate (t~16 here; their
    limit cycle is still bounded past t=50)."""
    Eu = float(euler_number(NPCH, NSUB, FR, LAM, KI, KE))
    x0 = steady_state(NSUB, NPCH)
    params = dict(Nsub=NSUB, Npch=NPCH, Fr=FR, Lam=LAM, ki=KI, ke=KE, Eu=Eu)
    f = lambda x: state_derivative(x, params)

    dt = 0.01
    x = x0 + jnp.array([0.0, 0.0, -0.02])

    def rk4_step(x, _):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    xf, _ = jax.lax.scan(rk4_step, x, xs=None, length=2000)  # t: 0 to 20
    assert bool(jnp.any(jnp.isnan(xf))), (
        "expected N1=1 to have already diverged by t=20 at this parameter point"
    )


def test_n1_2_even_node_count_fixes_the_pathology():
    """The actual fix: N1=2 (even) integrates through the same window without
    diverging, and settles into a bounded oscillation whose amplitude range
    matches the paper's Fig. 5 (~0.15-0.8) to within the precision of reading a
    printed figure."""
    Eu = float(euler_number(NPCH, NSUB, FR, LAM, KI, KE))
    x0 = steady_state_n1_2(NSUB, NPCH)
    params = dict(Nsub=NSUB, Npch=NPCH, Fr=FR, Lam=LAM, ki=KI, ke=KE, Eu=Eu)
    f = lambda x: state_derivative_n1_2(x, params)

    dt = 0.005
    x_start = x0 + jnp.array([0.0, 0.0, 0.0, -0.02])

    def rk4_step(x, _):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next, x_next

    _, traj = jax.lax.scan(rk4_step, x_start, xs=None, length=10000)  # t: 0 to 50
    traj = np.array(traj)

    assert not np.any(np.isnan(traj)), "N1=2 should not diverge over this window"

    ui_final_quarter = traj[7500:, 3]
    assert 0.05 < ui_final_quarter.min() < 0.15, "trough amplitude off from paper Fig. 5"
    assert 0.75 < ui_final_quarter.max() < 0.95, "peak amplitude off from paper Fig. 5"
