"""Validates the node-count convergence study (PROJECT_LOG.md Sec. 24): does the
N1=4 codim-2 point (Sec. 21, Nsub*~31.36) hold up as N1 increases?

Answer found: no -- it keeps moving, converging toward Nsub*~29.83-29.85 as N1
grows, an ~5% shift. These tests lock in that finding so a future change to the
physics model or the fold/Hopf machinery can't silently regress it unnoticed.
"""

import jax.numpy as jnp

from tide.continuation.codim2_convergence import _fold, find_codim2_point
from tide.continuation.curves import fold_npch
from tide.physics.clausse_lahey import state_derivative_general, steady_state_general
from tide.physics.ledinegg_curve import euler_number

FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03


def test_fast_fold_matches_slow_curves_fold_npch():
    # codim2_convergence uses the jitted grid-search fold (fast_gap_scan's),
    # not curves.fold_npch's unjitted 8000-point Python loop -- confirm they
    # agree to well within the ~1 unit shifts this module is measuring.
    for nsub in (15.0, 20.0, 31.36):
        slow = fold_npch(nsub, FR, LAM, KI, KE)
        fast = _fold(nsub, FR, LAM, KI, KE)
        assert abs(slow - fast) < 0.01


def test_n1_4_codim2_point_matches_sec21_manual_value():
    # Cross-check against the independently, manually refined value from Sec. 21.
    result = find_codim2_point(FR, LAM, KI, KE, N1=4, nsub_bracket=(31.0, 31.7))
    assert result is not None
    nsub_star, npch_star = result
    assert abs(nsub_star - 31.36) < 0.05
    assert abs(npch_star - 53.12) < 0.05


def test_codim2_point_shifts_substantially_with_node_count():
    # The core finding: N1=4 is NOT converged. Nsub* should decrease
    # monotonically and substantially as N1 increases from 4 to 16.
    n1_values = [4, 6, 8, 12, 16]
    windows = {4: (31.0, 31.7), 6: (30.0, 31.0), 8: (29.8, 30.5),
               12: (29.0, 30.5), 16: (28.0, 30.5)}
    nsub_stars = []
    for N1 in n1_values:
        result = find_codim2_point(FR, LAM, KI, KE, N1, windows[N1])
        assert result is not None, f"no codim-2 point found for N1={N1}"
        nsub_stars.append(result[0])

    # strictly decreasing
    assert all(b < a for a, b in zip(nsub_stars, nsub_stars[1:]))
    # the total shift from N1=4 to N1=16 is large (~1.5 in Nsub, ~5% relative) --
    # not noise. Assert it's at least 1.0, well above the ~0.01 grid-fold noise
    # floor established in test_fast_fold_matches_slow_curves_fold_npch.
    assert (nsub_stars[0] - nsub_stars[-1]) > 1.0


def test_convergence_is_slowing_down_not_diverging():
    # Successive differences should shrink in magnitude (genuine convergence,
    # not a runaway drift) -- checked over N1=8,12,16,20 where the earlier
    # (N1=4->6->8) transient has already settled down.
    windows = {8: (29.8, 30.5), 12: (29.0, 30.5), 16: (28.0, 30.5), 20: (27.0, 30.5)}
    nsub_stars = [find_codim2_point(FR, LAM, KI, KE, N1, w)[0] for N1, w in windows.items()]
    diffs = [abs(b - a) for a, b in zip(nsub_stars, nsub_stars[1:])]
    assert diffs[-1] < diffs[0]


def test_general_model_steady_state_residual_stays_machine_precision_at_high_n1():
    # The same structural cross-check Sec. 21 ran for N1=4, repeated at higher N1
    # to confirm the general-N1 recurrence in clausse_lahey.py doesn't develop
    # numerical trouble as N1 grows -- this convergence study is only trustworthy
    # if the underlying model stays exact at every N1 tested.
    Nsub, Npch = 29.87971, 49.69491
    for N1 in (4, 8, 16, 20, 24):
        Eu = float(euler_number(Npch, Nsub, FR, LAM, KI, KE))
        x0 = steady_state_general(Nsub, Npch, N1)
        params = dict(Nsub=Nsub, Npch=Npch, Fr=FR, Lam=LAM, ki=KI, ke=KE, Eu=Eu)
        resid = state_derivative_general(x0, params, N1)
        assert float(jnp.max(jnp.abs(resid))) < 1e-10
