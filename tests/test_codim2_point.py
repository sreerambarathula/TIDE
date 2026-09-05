"""Locks in the first genuine codimension-2 point found in this project
(PROJECT_LOG.md Sec. 18): Fr=0.5, k_in=11, k_out=3, Lambda->0, where the fold and
Hopf boundary curves meet almost exactly near Nsub=14.16, Npch=20.63.

Found via a joint search over (Fr, k_in, k_out) after Sec. 17 showed Lambda alone
(at Fr=5, ki=6, ke=2) cannot produce a crossing -- the gap there plateaus at a
nonzero floor (~0.275) as Lambda->0 rather than vanishing.
"""

import numpy as np

import tide  # noqa: F401
from tide.continuation.curves import fold_npch, hopf_npch, _leading_complex_real_part

FR, KI, KE = 0.5, 11.0, 3.0
LAM = 1e-5
NSUB_STAR = 14.16


def _gap_at(nsub):
    fp = fold_npch(nsub, FR, LAM, KI, KE)
    npch_scan = np.linspace(nsub * 1.2, nsub * 2.0, 25)
    re_scan = [_leading_complex_real_part(nsub, p, FR, LAM, KI, KE) for p in npch_scan]
    hp = None
    for i in range(len(npch_scan) - 1):
        a, b = re_scan[i], re_scan[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a < 0 < b:
            hp = hopf_npch(nsub, FR, LAM, KI, KE, bracket=(npch_scan[i], npch_scan[i + 1]))
            break
    return fp, hp


def test_gap_is_near_zero_at_the_codim2_point():
    fp, hp = _gap_at(NSUB_STAR)
    assert fp is not None and hp is not None
    gap = hp - fp
    assert abs(gap) < 1e-3, f"expected near-zero gap at the codim-2 point, got {gap}"


def test_gap_grows_on_both_sides_confirming_a_true_minimum():
    # Not a screener artifact: the gap must increase moving away from Nsub_star in
    # BOTH directions, confirming a genuine local minimum rather than a fluke.
    fp_below, hp_below = _gap_at(NSUB_STAR - 0.06)
    fp_at, hp_at = _gap_at(NSUB_STAR)
    fp_above, hp_above = _gap_at(NSUB_STAR + 0.06)

    gap_below = hp_below - fp_below
    gap_at = hp_at - fp_at
    gap_above = hp_above - fp_above

    assert gap_below > gap_at
    assert gap_above > gap_at
