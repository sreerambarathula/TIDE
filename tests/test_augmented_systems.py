"""Documents a real finding from Phase 2 (PROJECT_LOG.md Sec. 16): the raw N1=2
Clausse-Lahey system (Eq. 42) admits an extraneous, energy-conservation-violating
equilibrium branch when the state x is left fully free in an augmented Newton
system. This test locks in the diagnostic (h(lambda) != 0 on that branch), not a
recommendation to use the unconstrained fold solver -- see the module docstring and
PROJECT_LOG.md Sec. 16 for why the closed-form approach (ledinegg_curve.py /
curves.py) is the correct tool for fold detection in this model, not this one.
"""

import jax
import jax.numpy as jnp
import numpy as np

import tide  # noqa: F401
from tide.physics.clausse_lahey import steady_state_n1_2
from tide.physics.ledinegg_curve import euler_number
from tide.continuation.augmented_systems import solve_fold

NSUB, FR, LAM, KI, KE = 8.0, 5.0, 3.0, 6.0, 2.0


def test_unconstrained_fold_solver_finds_extraneous_energy_violating_root():
    """Reproduces the exact finding: seeded near the known physical fold
    (Npch=10.0118), the unconstrained augmented system converges to a different
    equilibrium whose lambda violates the boiling-boundary definition
    h(lambda) = lambda - Nsub/Npch = 0 by several percent. This is expected
    behavior of the (deliberately unconstrained) `_f` residual, not a bug to fix in
    this test -- it's the reason the constrained/closed-form approach is used
    everywhere else in the project for fold detection."""
    npch_known = 10.0118
    Eu_fold = float(euler_number(npch_known, NSUB, FR, LAM, KI, KE))
    x_known = steady_state_n1_2(NSUB, npch_known)

    x0 = x_known + jnp.array([0.01, -0.02, 0.015, -0.01])
    v0 = jnp.array([0.304, 0.494, 0.349, 0.735])
    npch0 = npch_known + 0.5

    result = solve_fold(x0, v0, npch0, NSUB, FR, LAM, KI, KE, Eu_fold, max_iter=200)

    assert result["residual_history"][-1] < 1e-8, "Newton should still converge cleanly"

    lam_found = float(result["x"][1])
    npch_found = result["npch"]
    h_lambda = lam_found - NSUB / npch_found

    # It should NOT land back on the known physical point...
    assert abs(npch_found - npch_known) > 0.5
    # ...and the boiling-boundary definition should be measurably violated there,
    # confirming it's an extraneous (unphysical) root rather than a second valid
    # physical equilibrium.
    assert abs(h_lambda) > 0.01, (
        f"expected a clear energy-conservation violation, got h(lambda)={h_lambda}"
    )


def test_closed_form_branch_satisfies_boiling_boundary_definition():
    """Sanity check the other direction: the closed-form steady state used
    everywhere else in the project DOES satisfy h(lambda)=0 exactly, at any Npch,
    Nsub -- confirming it (not the extraneous root) is the physically valid branch."""
    for nsub, npch in [(8.0, 10.0), (6.5, 14.0), (10.0, 20.0)]:
        x = steady_state_n1_2(nsub, npch)
        lam = float(x[1])
        h_lambda = lam - nsub / npch
        assert abs(h_lambda) < 1e-10
