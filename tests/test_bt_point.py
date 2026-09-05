"""Validates the genuine Bogdanov-Takens point (PROJECT_LOG.md Sec. 37) and
the positive-counterpart decoupling result at it (Sec. 38).
"""

import numpy as np

from tide.eval.bt_decoupling import (_extract_surrogate_upper,
                                      run_bt_decoupling_analysis,
                                      true_gradient_norm_at)
from tide.surrogates.bt_point_data import (FR_BT, KE_BT, KI_BT, LAM_BT,
                                            NPCH_BT, NSUB_BT,
                                            generate_bt_dataset,
                                            wedge_boundaries)
from tide.surrogates.mlp import train_surrogate, train_test_split

N1 = 16


def test_wedge_closes_at_the_bt_point():
    # window width should shrink monotonically to (near) zero approaching
    # Nsub_BT -- the defining cusp signature.
    nsubs = [12.0, 13.0, 13.8, 14.0, 14.1]
    widths = []
    for nsub in nsubs:
        lower, upper = wedge_boundaries(nsub, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
        assert lower is not None and upper is not None
        widths.append(upper - lower)
    assert all(b < a for a, b in zip(widths, widths[1:]))
    assert widths[-1] < 0.1


def test_no_window_at_or_above_bt_point():
    lower, upper = wedge_boundaries(NSUB_BT + 0.5, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    assert upper is None


def test_bt_dataset_generates_finite_data_spanning_the_window():
    data = generate_bt_dataset(FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.0), n_nsub=20)
    assert len(data["nsub"]) > 200
    for v in data.values():
        assert np.all(np.isfinite(v))
    assert (data["g"] < 0).sum() > 10  # some points inside the stable window
    assert (data["g"] > 0).sum() > 10


def test_boundary_error_spikes_near_bt_point_while_field_error_stays_flat():
    # Single-seed smoke test of the core positive result (fast settings) --
    # the full 20-seed statistical confirmation is in PROJECT_LOG.md Sec. 38,
    # not re-run here for speed. This just locks in the qualitative pattern.
    data = generate_bt_dataset(FR_BT, LAM_BT, KI_BT, KE_BT, N1, nsub_range=(11.0, 14.05), n_nsub=30)
    train, _ = train_test_split(data, seed=0)
    surrogate, _ = train_surrogate(train, n_epochs=2000, seed=0)

    nsub_grid = np.linspace(11.5, 14.0, 20)
    per_point, summary = run_bt_decoupling_analysis(surrogate, FR_BT, LAM_BT, KI_BT, KE_BT, N1, NSUB_BT, nsub_grid)
    assert summary["n_near"] > 0 and summary["n_far"] > 0
    assert summary["upper_error_near"] > summary["upper_error_far"]


def test_gradient_does_not_vanish_but_window_width_does():
    # Sec. 39's mechanism finding: ||grad g|| at the upper boundary stays
    # roughly constant (does NOT vanish) approaching the BT point, while the
    # window width shrinks to near zero -- the actual driver of boundary
    # error here is feature-scale shrinkage, not a vanishing gradient.
    nsubs = [11.0, 12.5, 14.0, 14.1]
    grad_norms, widths = [], []
    for nsub in nsubs:
        lower, upper = wedge_boundaries(nsub, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
        assert lower is not None and upper is not None
        grad_norms.append(true_gradient_norm_at(nsub, upper, FR_BT, LAM_BT, KI_BT, KE_BT, N1))
        widths.append(upper - lower)

    # gradient stays within a narrow band (does not collapse toward zero)
    assert min(grad_norms) > 0.5 * max(grad_norms)
    # width shrinks dramatically (~70x here; the earlier PROJECT_LOG check
    # at Nsub=14.14 found ~750x -- this test uses Nsub=14.1 for reliability
    # margin against the bracket-search precision limits near the exact tip)
    assert widths[-1] < 0.02 * widths[0]


class _MockSurrogateWithSpuriousRoot:
    """Fakes a surrogate whose predicted g has a spurious extra root far from
    the true boundary -- reproduces the exact failure mode found in Sec. 40:
    a wide, uninformed bracket picks up the spurious root instead of the real
    one. Used to lock in that _extract_surrogate_upper now searches near the
    true value instead of taking the first sign change encountered."""

    def predict(self, nsub, npch):
        npch = np.asarray(npch)
        # true root at npch=10; spurious extra root at npch=6 that a wide
        # (fold, 1.6*fold)-style sweep starting well below 10 would hit first.
        g = (npch - 10.0) * (npch - 6.0) / 10.0
        return np.zeros_like(npch), g

    def predict_scalar(self, nsub, npch):
        g = (npch - 10.0) * (npch - 6.0) / 10.0
        return 0.0, g


def test_extraction_finds_root_near_true_value_not_spurious_one():
    surrogate = _MockSurrogateWithSpuriousRoot()
    # true value is 10.0; a wide blind sweep from ~7 upward would hit the
    # spurious root's neighborhood (the parabola is negative between 6 and
    # 10, so within (7, 10) there's no sign change, but a genuine wide sweep
    # starting near 6 would cross zero at both 6 and 10 -- picking the first
    # one found is exactly Sec. 40's bug).
    result = _extract_surrogate_upper(surrogate, nsub=1.0, true_upper=10.0)
    assert result is not None
    assert abs(result - 10.0) < 0.01
