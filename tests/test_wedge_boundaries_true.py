"""Validates wedge_boundaries_true, the corrected replacement for
wedge_boundaries (PROJECT_LOG.md, post-manuscript-draft external audit:
the old function's `lower` was the Eq. 26 fold value, which sits measurably
OUTSIDE the true stable window near the BT tip, not on its edge).
"""
import numpy as np

from tide.surrogates.bt_point_data import (FR_BT, KE_BT, KI_BT, LAM_BT,
                                            _g, wedge_boundaries,
                                            wedge_boundaries_true)

N1 = 16


def test_matches_audited_reference_values_near_bt_tip():
    lower, upper = wedge_boundaries_true(14.14, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    assert abs(lower - 20.592275971) < 1e-6
    assert abs(upper - 20.594954087) < 1e-6


def test_true_lower_boundary_differs_from_the_superseded_fold_value():
    # Regression-locks the audit finding itself: the old function's lower
    # boundary (the fold) is NOT the same point as the true g=0 root.
    old_lower, _ = wedge_boundaries(14.14, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    true_lower, _ = wedge_boundaries_true(14.14, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    assert abs(true_lower - old_lower) > 1e-3


def test_both_boundaries_are_genuine_g_zero_crossings():
    lower, upper = wedge_boundaries_true(13.0, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    assert abs(_g(13.0, lower, FR_BT, LAM_BT, KI_BT, KE_BT, N1)) < 1e-6
    assert abs(_g(13.0, upper, FR_BT, LAM_BT, KI_BT, KE_BT, N1)) < 1e-6


def test_window_is_stable_inside_and_unstable_just_outside():
    lower, upper = wedge_boundaries_true(13.0, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    mid = 0.5 * (lower + upper)
    assert _g(13.0, mid, FR_BT, LAM_BT, KI_BT, KE_BT, N1) < 0
    assert _g(13.0, lower - 0.01, FR_BT, LAM_BT, KI_BT, KE_BT, N1) > 0
    assert _g(13.0, upper + 0.01, FR_BT, LAM_BT, KI_BT, KE_BT, N1) > 0


def test_width_grows_monotonically_moving_away_from_bt_point():
    widths = []
    for nsub in [14.10, 13.5, 13.0, 12.0, 11.0]:
        lower, upper = wedge_boundaries_true(nsub, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
        widths.append(upper - lower)
    assert np.all(np.diff(widths) > 0)
