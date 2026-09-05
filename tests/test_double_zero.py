"""Validates the corrected double-zero (Bogdanov-Takens) point solver
against an external audit's independently-computed reference values
(PROJECT_LOG.md, post-manuscript review) -- these values were reproduced
here from scratch via 2D Newton on the sum/product of the coalescing
eigenvalue pair, not copied from the audit.
"""
import numpy as np

from tide.continuation.double_zero import _coalescing_pair, find_double_zero_point

FR_BT, LAM_BT, KI_BT, KE_BT = 0.5, 0.001, 11.0, 3.0
FR_C, LAM_C, KI_C, KE_C = 0.3, 0.001, 6.0, 5.0
N1 = 16


def test_point_b_converges_to_audited_double_zero_location():
    nsub, npch, converged, residual, it = find_double_zero_point(
        [14.14278, 20.59478], FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    assert converged
    assert np.max(np.abs(residual)) < 1e-9
    assert abs(nsub - 14.142794816) < 1e-6
    assert abs(npch - 20.597778032) < 1e-6


def test_point_c_converges_to_audited_double_zero_location():
    nsub, npch, converged, residual, it = find_double_zero_point(
        [7.630091, 10.914522], FR_C, LAM_C, KI_C, KE_C, N1)
    assert converged
    assert np.max(np.abs(residual)) < 1e-9
    assert abs(nsub - 7.630101792) < 1e-6
    assert abs(npch - 10.913202566) < 1e-6


def test_converged_point_is_actually_a_repeated_zero_eigenvalue():
    # Direct spectral check, not just residual=0 on the sum/product proxy --
    # the whole point of this fix is that the OLD coordinates fail exactly
    # this check (two distinct real eigenvalues, not a repeated zero).
    nsub, npch, converged, residual, it = find_double_zero_point(
        [14.14278, 20.59478], FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    pair = _coalescing_pair(nsub, npch, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    assert np.max(np.abs(pair)) < 1e-4


def test_old_recorded_point_is_confirmed_not_a_double_zero():
    # Regression-locks the audit finding itself: at the ORIGINAL recorded
    # (NSUB_BT, NPCH_BT), the two smallest-magnitude eigenvalues are
    # distinct and O(1e-2), not a repeated zero.
    pair = _coalescing_pair(14.14278, 20.59478, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    assert abs(pair[0] - pair[1]) > 1e-3


def test_double_zero_point_is_n1_independent():
    # Same claim as the original (flawed) discovery -- re-verified against
    # the CORRECTED coordinates, since the original N1-independence check
    # was performed on the wrong point.
    results = {}
    for n1 in (2, 4, 8, 16):
        nsub, npch, converged, residual, it = find_double_zero_point(
            [14.14278, 20.59478], FR_BT, LAM_BT, KI_BT, KE_BT, n1)
        assert converged
        results[n1] = (nsub, npch)
    values = list(results.values())
    for v in values[1:]:
        assert abs(v[0] - values[0][0]) < 1e-4
        assert abs(v[1] - values[0][1]) < 1e-4
