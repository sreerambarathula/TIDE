"""Validates the real-mode instability threshold finder (PROJECT_LOG.md Sec.
27) -- a persistent, non-oscillatory unstable eigenvalue found while building
Phase 3 data, distinct from both the fold (Sec. 16) and the Hopf (Sec. 13-14).
"""

from tide.continuation.real_mode_boundary import best_case_stability_margin, find_nsub_crit

N1 = 4


def test_nsub_crit_matches_realistic_saha_ishii_params():
    FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
    crit = find_nsub_crit(FR, LAM, KI, KE, N1, nsub_bracket=(15.0, 22.0))
    assert crit is not None
    assert abs(crit - 18.43) < 0.05


def test_nsub_crit_matches_sec18_combo():
    FR, LAM, KI, KE = 0.5, 0.001, 11.0, 3.0
    crit = find_nsub_crit(FR, LAM, KI, KE, N1, nsub_bracket=(12.0, 16.0))
    assert crit is not None
    assert abs(crit - 14.08) < 0.05


def test_below_threshold_some_npch_is_stable_above_it_none_is():
    FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
    below = best_case_stability_margin(15.0, FR, LAM, KI, KE, N1)
    above = best_case_stability_margin(25.0, FR, LAM, KI, KE, N1)
    assert below < 0
    assert above > 0
