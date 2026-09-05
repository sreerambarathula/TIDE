"""Validates the BT-point adaptive sampling module (PROJECT_LOG.md Sec. 41),
Phase 4's first fix candidate (later found to only modestly help -- data
density was ruled out as the limiting factor there; this just checks the
module itself works correctly).
"""

import numpy as np

from tide.surrogates.bt_adaptive_sampling import measure_misfit_and_augment_bt
from tide.surrogates.bt_point_data import FR_BT, KE_BT, KI_BT, LAM_BT, generate_bt_dataset

N1 = 16


def test_augmented_bt_dataset_is_larger_and_finite():
    base = generate_bt_dataset(FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.05), n_nsub=20)
    augmented, diag = measure_misfit_and_augment_bt(
        base, FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.05),
        n_candidates=500, n_add=100, seed=0, n_epochs=800,
    )
    assert len(augmented["nsub"]) > len(base["nsub"])
    assert diag["n_added"] > 0
    for v in augmented.values():
        assert np.all(np.isfinite(v))


def test_added_points_stay_in_valid_domain():
    base = generate_bt_dataset(FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.05), n_nsub=20)
    _, diag = measure_misfit_and_augment_bt(
        base, FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.05),
        n_candidates=500, n_add=100, seed=0, n_epochs=800,
    )
    assert np.all(diag["added_nsub"] >= 11.0)
    assert np.all(diag["added_nsub"] <= 14.05)
