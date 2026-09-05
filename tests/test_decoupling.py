"""Validates the decoupling analysis (PROJECT_LOG.md Sec. 31). Locks in the
two fixes that mattered: Hopf extraction must use the same narrow band as
the ground-truth tool (a wide grid picks up a different, real sign change
unrelated to the Hopf boundary), and field error must be measured over a
local neighborhood, not exactly at the root itself.
"""

import numpy as np

from tide.eval.decoupling import _extract_surrogate_hopf, run_decoupling_analysis
from tide.surrogates.data_generation import _boundaries_at, generate_stability_dataset
from tide.surrogates.mlp import train_surrogate, train_test_split

FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
N1 = 16
NSUB_STAR = 29.886


def _quick_surrogate(seed=0):
    data = generate_stability_dataset(FR, LAM, KI, KE, N1, nsub_range=(24.0, 36.0), n_nsub=30)
    train, _ = train_test_split(data, seed=seed)
    surrogate, _ = train_surrogate(train, n_epochs=1500, seed=seed)
    return surrogate


def test_hopf_extraction_matches_true_value_when_restricted_to_narrow_band():
    surrogate = _quick_surrogate()
    nsub = 28.0
    true_fold, true_hopf = _boundaries_at(nsub, FR, LAM, KI, KE, N1)
    npch_band = np.linspace(true_fold * 0.85, true_fold * 1.4, 2000)
    surr_hopf = _extract_surrogate_hopf(surrogate, nsub, npch_band)
    assert surr_hopf is not None
    assert abs(surr_hopf - true_hopf) < 1.0


def test_wide_band_extraction_finds_a_different_earlier_crossing():
    # Regression test for the bug itself: a WIDE band starting near nsub
    # should NOT match the true Hopf point, confirming the narrow-band
    # requirement is load-bearing, not incidental.
    surrogate = _quick_surrogate()
    nsub = 28.0
    _, true_hopf = _boundaries_at(nsub, FR, LAM, KI, KE, N1)
    wide_band = np.linspace(nsub * 1.02, nsub * 3.0, 2000)
    surr_hopf_wide = _extract_surrogate_hopf(surrogate, nsub, wide_band)
    assert surr_hopf_wide is not None
    assert abs(surr_hopf_wide - true_hopf) > 5.0


def test_decoupling_analysis_runs_and_produces_near_far_groups():
    surrogate = _quick_surrogate()
    nsub_grid = np.linspace(24.5, 35.5, 20)
    data, summary = run_decoupling_analysis(surrogate, FR, LAM, KI, KE, N1, NSUB_STAR, nsub_grid)
    assert len(data["nsub"]) > 10
    assert summary["n_near"] > 0
    assert summary["n_far"] > 0
    assert np.all(np.isfinite(data["fold_error"]))
    assert np.all(np.isfinite(data["hopf_error"]))
