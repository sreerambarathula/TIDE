"""Validates the fold-Hopf crossing tangency check (PROJECT_LOG.md Sec. 35),
which explains WHY Phase 3 found no decoupling signal: the codim-2 crossing
is structurally transversal, not tangent, across an extensive parameter
sweep -- locking in the canonical point's specific value here.
"""

from tide.continuation.tangency import crossing_slopes

FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
N1 = 16
NSUB_STAR = 29.886


def test_canonical_crossing_is_transversal_not_tangent():
    result = crossing_slopes(FR, LAM, KI, KE, N1, NSUB_STAR)
    assert result is not None
    slope_fold, slope_hopf, diff = result
    # both slopes are real, finite, positive, and clearly distinct --
    # a genuine tangency would have diff close to 0.
    assert diff > 0.5
    assert slope_fold > 0
    assert slope_hopf > 0
