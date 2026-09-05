"""Validates the general even-N1 Clausse-Lahey model (PROJECT_LOG.md Sec. 21)
against the already-proven N1=2 special case, and locks in the realistic-parameter
codim-2 point found with N1=4.
"""

import jax
import jax.numpy as jnp
import numpy as np

import tide  # noqa: F401
from tide.physics.clausse_lahey import (
    state_derivative_general,
    state_derivative_n1_2,
    steady_state_general,
    steady_state_n1_2,
)
from tide.physics.ledinegg_curve import euler_number
from tide.continuation.curves import fold_npch
from tide.continuation.fast_gap_scan import gap_scan, gap_scan_general


def test_general_n1_2_matches_dedicated_n1_2_implementation():
    Nsub, Npch, Fr, Lam, ki, ke = 8.0, 14.0, 5.0, 3.0, 6.0, 2.0
    Eu = float(euler_number(Npch, Nsub, Fr, Lam, ki, ke))
    params = dict(Nsub=Nsub, Npch=Npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)

    x0_dedicated = steady_state_n1_2(Nsub, Npch)
    x0_general = steady_state_general(Nsub, Npch, 2)
    assert np.allclose(x0_dedicated, x0_general)

    f_dedicated = state_derivative_n1_2(x0_dedicated, params)
    f_general = state_derivative_general(x0_general, params, 2)
    assert np.allclose(f_dedicated, f_general, atol=1e-12)

    # also check away from steady state, not just at it
    x_pert = x0_dedicated + jnp.array([0.01, -0.02, 0.01, 0.005])
    assert np.allclose(
        state_derivative_n1_2(x_pert, params),
        state_derivative_general(x_pert, params, 2),
        atol=1e-10,
    )


def test_n1_4_steady_state_satisfies_equations():
    Nsub, Npch = 6.5, 14.0  # the paper's validated Hopf example point
    Fr, Lam, ki, ke = 1.0, 3.0, 6.0, 2.0
    Eu = float(euler_number(Npch, Nsub, Fr, Lam, ki, ke))
    params = dict(Nsub=Nsub, Npch=Npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)

    x0 = steady_state_general(Nsub, Npch, 4)
    f0 = np.array(state_derivative_general(x0, params, 4))
    assert np.max(np.abs(f0)) < 1e-10


def test_n1_4_reproduces_the_known_hopf_instability():
    Nsub, Npch = 6.5, 14.0
    Fr, Lam, ki, ke = 1.0, 3.0, 6.0, 2.0
    Eu = float(euler_number(Npch, Nsub, Fr, Lam, ki, ke))
    params = dict(Nsub=Nsub, Npch=Npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)

    x0 = steady_state_general(Nsub, Npch, 4)
    f = lambda x: state_derivative_general(x, params, 4)
    J = np.array(jax.jacfwd(f)(x0))
    eigvals = np.linalg.eigvals(J)
    complex_eigs = eigvals[np.abs(eigvals.imag) > 1e-6]
    assert len(complex_eigs) >= 2
    assert np.any(complex_eigs.real > 0), "expected an unstable complex pair, as N1=2 also showed"


def test_gap_scan_general_n1_2_matches_dedicated_scanner():
    nsub_range = np.arange(6.0, 20.0, 1.0)
    r1 = gap_scan(5.0, 3.0, 6.0, 2.0, nsub_range)
    r2 = gap_scan_general(5.0, 3.0, 6.0, 2.0, nsub_range, N1=2)
    for (n1, f1, h1, g1), (n2, f2, h2, g2) in zip(r1, r2):
        assert n1 == n2
        if g1 is None:
            assert g2 is None
        else:
            assert abs(g1 - g2) < 1e-6


def test_realistic_parameter_codim2_point_at_n1_4():
    """The key finding: entirely real, documented facility values (Fr=0.035 and
    k_in=6.55, k_out=2.03 from Saha's facility; Lambda=5.90 from Ishii's) give a
    genuine sign change near Nsub~31.36 with N1=4 -- something N1=2 could not
    produce at any realistic parameter combination (Sec. 19)."""
    Fr, Lam, ki, ke = 0.035, 5.90, 6.55, 2.03
    N1 = 4

    def leading_re(nsub, npch):
        Eu = float(euler_number(npch, nsub, Fr, Lam, ki, ke))
        x0 = steady_state_general(nsub, npch, N1)
        params = dict(Nsub=nsub, Npch=npch, Fr=Fr, Lam=Lam, ki=ki, ke=ke, Eu=Eu)
        f = lambda x: state_derivative_general(x, params, N1)
        J = np.array(jax.jacfwd(f)(x0))
        ce = np.linalg.eigvals(J)
        ce = ce[np.abs(ce.imag) > 1e-9]
        return np.max(ce.real) if len(ce) else -np.inf

    def gap_at(nsub):
        # matches the manually-validated methodology (PROJECT_LOG Sec. 21): a
        # coarse scan to bracket the sign change, then full bisection (not just
        # linear interpolation) -- needed because the gap here is O(1e-3), too
        # small for a 1-step linear estimate to reliably get the sign right.
        fp = fold_npch(nsub, Fr, Lam, ki, ke)
        npch_scan = np.linspace(nsub * 1.1, nsub * 3.0, 30)
        re_scan = [leading_re(nsub, p) for p in npch_scan]
        for i in range(len(npch_scan) - 1):
            a, b = re_scan[i], re_scan[i + 1]
            if np.isfinite(a) and np.isfinite(b) and a < 0 < b:
                lo, hi = npch_scan[i], npch_scan[i + 1]
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    val = leading_re(nsub, mid)
                    if val < 0:
                        lo = mid
                    else:
                        hi = mid
                return fp, 0.5 * (lo + hi)
        return fp, None

    gaps = []
    for nsub in [31.35, 31.36, 31.37, 31.38]:
        fp, hp = gap_at(nsub)
        assert fp is not None and hp is not None
        gaps.append(hp - fp)

    assert gaps[0] > 0, f"expected positive gap at Nsub=31.35, got {gaps[0]}"
    assert gaps[-1] < 0, f"expected negative gap at Nsub=31.38, got {gaps[-1]}"
    assert all(b < a for a, b in zip(gaps, gaps[1:])), "gap should decrease monotonically here"
