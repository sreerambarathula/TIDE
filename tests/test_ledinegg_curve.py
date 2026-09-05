"""Validate the Ledinegg internal characteristic curve (Theler, Clausse & Bonetto
2010, Eq. 26) against the paper's own published Figure 3 (Nsub=8, Fr=5, Lambda=3,
k_in=6, k_out=2): the three labeled Euler-number curves should cross zero (of
Eq. 25's left member) -- equivalently, Eu(Npch) should equal 10, 10.5, 11 -- at the
N_pch values readable from that figure, and the curve must show a genuine fold
(local extremum), which is the Ledinegg instability signature.
"""

import numpy as np

import tide  # noqa: F401
from tide.physics.ledinegg_curve import euler_number

NSUB, FR, LAM, KIN, KOUT = 8.0, 5.0, 3.0, 6.0, 2.0


def _npch_at_eu(target_eu, n_pch_grid):
    eu = np.asarray(euler_number(n_pch_grid, NSUB, FR, LAM, KIN, KOUT))
    return n_pch_grid[np.argmin(np.abs(eu - target_eu))]


def test_matches_published_figure_3_crossings():
    # Figure 3 zero-crossings read off the plot (Eu=10 curve crosses near Npch~17.7,
    # Eu=10.5 near ~15.6, Eu=11 near ~13.5). Reading a printed figure by eye has
    # maybe +-0.5 precision, so allow a generous tolerance.
    n_pch_grid = np.linspace(8.001, 25.0, 200000)
    expected = {10.0: 17.7, 10.5: 15.6, 11.0: 13.5}
    for eu_target, expected_npch in expected.items():
        got = _npch_at_eu(eu_target, n_pch_grid)
        assert abs(got - expected_npch) < 0.5, (
            f"Eu={eu_target}: expected Npch~{expected_npch}, got {got:.3f}"
        )


def test_curve_has_exactly_one_fold_ledinegg_signature():
    n_pch_grid = np.linspace(8.001, 30.0, 200000)
    eu = np.asarray(euler_number(n_pch_grid, NSUB, FR, LAM, KIN, KOUT))
    d = np.diff(eu)
    sign_changes = np.sum(np.diff(np.sign(d)) != 0)
    assert sign_changes == 1, f"expected exactly one fold, found {sign_changes}"

    fold_idx = np.argmax(eu)
    # Eu must rise then fall around the fold -- the multivalued-flow-rate signature
    assert eu[fold_idx] > eu[0]
    assert eu[fold_idx] > eu[-1]


def test_mathematical_vs_physical_validity_boundary():
    # The paper states Eq. 26 is only *physically* meaningful for Npch > Nsub (below
    # that there is no two-phase flow at all). The formula's actual mathematical
    # singularity is looser: log(1 + Npch - Nsub) blows up at Npch = Nsub - 1, and
    # is undefined (nan) only below that -- confirmed by direct evaluation, not
    # assumed. Eq. 26 is numerically finite (but physically meaningless) on
    # (Nsub - 1, Nsub); don't mistake that range for validity in any later use.
    with np.errstate(invalid="ignore"):
        at_singularity = float(euler_number(NSUB - 1.0, NSUB, FR, LAM, KIN, KOUT))
        below_singularity = float(euler_number(NSUB - 1.5, NSUB, FR, LAM, KIN, KOUT))
        in_unphysical_gap = float(euler_number(NSUB - 0.5, NSUB, FR, LAM, KIN, KOUT))

    assert at_singularity == float("-inf")
    assert np.isnan(below_singularity)
    assert np.isfinite(in_unphysical_gap)  # numerically fine, physically meaningless
