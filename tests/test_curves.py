"""Validate the fold and Hopf boundary-tracing curves (src/tide/continuation/curves.py).

Locks in the corrected fold-finder (grid-search-seeded, not a caller-supplied
bracket -- see PROJECT_LOG.md Sec. 15 for the bug this replaced: a badly-seeded
bracket converged to a spurious root, producing a non-monotonic, implausible fold
curve) and the finding that fold and Hopf approach but do not cross for these
channel parameters.
"""

import numpy as np

import tide  # noqa: F401
from tide.continuation.curves import fold_npch, hopf_npch, _leading_complex_real_part

FR, LAM, KI, KE = 5.0, 3.0, 6.0, 2.0


def test_fold_matches_ledinegg_curve_validation():
    # Cross-check against the independently-validated fold from test_ledinegg_curve.py
    # (Nsub=8, Fr=5, Lam=3, ki=6, ke=2 -> fold at Npch~10.01)
    fp = fold_npch(8.0, FR, LAM, KI, KE)
    assert abs(fp - 10.0118) < 1e-3


def test_fold_curve_is_monotonically_increasing():
    # Documents the bug found and fixed: an earlier bracket-based finder gave a
    # non-monotonic curve (12.53 at Nsub=7.5, dropping to 10.01 at Nsub=8) that
    # turned out to be a search artifact, not real structure.
    nsubs = [7.5, 8.0, 9.0, 10.0, 11.0, 12.0]
    folds = [fold_npch(ns, FR, LAM, KI, KE) for ns in nsubs]
    assert all(f is not None for f in folds)
    assert all(b > a for a, b in zip(folds, folds[1:])), (
        f"fold curve should increase monotonically with Nsub, got {folds}"
    )


def test_no_fold_below_threshold_nsub():
    # Confirmed via wide dense-grid scan (not just this narrow check) that the fold
    # genuinely doesn't exist for Nsub <~ 7.2 -- Eu(Npch) is monotonically
    # decreasing there, no local maximum at all.
    assert fold_npch(6.0, FR, LAM, KI, KE) is None
    assert fold_npch(7.0, FR, LAM, KI, KE) is None


def test_fold_and_hopf_approach_but_do_not_cross():
    # The actual Phase 2 finding: for this channel's friction parameters, the two
    # curves get closest (gap ~2.9) near Nsub=10 but never meet -- no codim-2 point
    # exists here. Don't assume one exists without checking.
    nsub = 10.0
    fp = fold_npch(nsub, FR, LAM, KI, KE)

    npch_scan = np.linspace(nsub * 1.3, nsub * 4.0, 25)
    re_scan = [_leading_complex_real_part(nsub, p, FR, LAM, KI, KE) for p in npch_scan]
    bracket = None
    for i in range(len(npch_scan) - 1):
        a, b = re_scan[i], re_scan[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a < 0 < b:
            bracket = (npch_scan[i], npch_scan[i + 1])
            break
    assert bracket is not None
    hp = hopf_npch(nsub, FR, LAM, KI, KE, bracket=bracket)

    gap = hp - fp
    assert gap > 1.0, f"expected fold and Hopf to remain clearly separated, gap={gap}"
