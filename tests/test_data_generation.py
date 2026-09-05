"""Validates the Phase 3 Stage 1 training-data generator (PROJECT_LOG.md Sec.
27-28). Locks in the fix that mattered: g must be the leading COMPLEX-pair
real part, not max over all eigenvalues -- the latter is dominated everywhere
in this region by the (real, documented) excursive-instability mode and never
changes sign, which would silently make the fold/Hopf boundary invisible.
"""

import numpy as np

from tide.surrogates.data_generation import generate_stability_dataset

FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
N1 = 16


def test_dataset_has_no_nans_and_reasonable_size():
    data = generate_stability_dataset(FR, LAM, KI, KE, N1, nsub_range=(24.0, 36.0), n_nsub=20)
    assert len(data["nsub"]) > 500
    for v in data.values():
        assert np.all(np.isfinite(v))


def test_g_straddles_zero_not_dominated_by_excursive_mode():
    # This is the core regression: an earlier version used max(Re(all
    # eigenvalues)) and every single point came back positive (unstable) --
    # the fold/Hopf boundary was completely invisible to the dataset.
    data = generate_stability_dataset(FR, LAM, KI, KE, N1, nsub_range=(24.0, 36.0), n_nsub=40)
    n_stable = (data["g"] < 0).sum()
    n_unstable = (data["g"] > 0).sum()
    assert n_stable > 100
    assert n_unstable > 100


def test_npch_always_exceeds_nsub():
    data = generate_stability_dataset(FR, LAM, KI, KE, N1, nsub_range=(24.0, 36.0), n_nsub=20)
    assert np.all(data["npch"] > data["nsub"])


def test_hopf_and_fold_labels_bracket_the_codim2_crossover():
    # Below Nsub*~29.886 Hopf sits ABOVE fold (Hopf is the operative
    # boundary); above it, Hopf sits BELOW fold (fold is operative) -- Sec.
    # 26. Confirm the per-row labels reflect this crossover.
    data = generate_stability_dataset(FR, LAM, KI, KE, N1, nsub_range=(24.0, 36.0), n_nsub=40)
    low = data["nsub"] < 27.0
    high = data["nsub"] > 33.0
    assert np.all(data["hopf_at_nsub"][low] > data["fold_at_nsub"][low])
    assert np.all(data["hopf_at_nsub"][high] < data["fold_at_nsub"][high])
