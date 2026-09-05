"""Validates the physics-API robustness fixes from the external audit
(PROJECT_LOG.md Sec. 44, finding 11): odd-N1 rejection in the general
(even-N1-only) formulation, and the steady_state removable-singularity fix
at Nsub=Npch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tide.physics.clausse_lahey import (state_derivative_general,
                                         steady_state, steady_state_general)
from tide.physics.ledinegg_curve import euler_number


def test_steady_state_no_longer_nans_at_nsub_equals_npch():
    result = np.array(steady_state(8.0, 8.0))
    assert np.all(np.isfinite(result))
    assert np.allclose(result, [1.0, 1.0, 1.0])


def test_steady_state_matches_old_formula_away_from_the_singularity():
    # Regression check: the safe-input rewrite must not change the answer
    # anywhere except exactly at the removable singularity.
    for nsub, npch in [(6.5, 14.0), (10.0, 20.0), (24.0, 30.0)]:
        lam0 = nsub / npch
        rho_e0 = 1.0 / (1.0 + npch * (1.0 - lam0))
        r0 = 1.0 / rho_e0
        expected_m0 = lam0 + (1.0 - lam0) * np.log(r0) / (r0 - 1.0)
        result = np.array(steady_state(nsub, npch))
        assert abs(result[1] - expected_m0) < 1e-10


def test_steady_state_gradient_is_finite_near_the_singularity():
    # The whole point of the safe-input trick, not just the forward value.
    grad_fn = jax.grad(lambda npch: steady_state(npch, npch)[1])
    g = float(grad_fn(8.0))
    assert np.isfinite(g)


def test_state_derivative_general_rejects_odd_n1():
    x = jnp.zeros(5)  # arbitrary shape, error should fire before using it
    with pytest.raises(ValueError, match="even N1"):
        state_derivative_general(x, params={}, N1=3)


def test_steady_state_general_rejects_odd_n1():
    with pytest.raises(ValueError, match="even N1"):
        steady_state_general(6.5, 14.0, N1=5)


def test_state_derivative_general_still_accepts_even_n1():
    Nsub, Npch, Fr, Lam, ki, ke = 6.5, 14.0, 0.035, 5.90, 6.55, 2.03
    Eu = euler_number(Npch, Nsub, Fr, Lam, ki, ke)
    params = dict(Nsub=Nsub, Npch=Npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)
    x = steady_state_general(Nsub, Npch, N1=4)  # a genuine, valid state
    result = state_derivative_general(x, params, N1=4)
    assert np.all(np.isfinite(np.array(result)))
