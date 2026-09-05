"""Validates the seed-level (not per-Nsub-point) statistics module
(PROJECT_LOG.md Sec. 32) that fixed the pseudo-replication bug in the first
decoupling analysis (Sec. 31).
"""

import numpy as np

from tide.eval.stats import paired_near_vs_far_test, run_and_summarize
from tide.surrogates.data_generation import generate_stability_dataset

FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
N1 = 16
NSUB_STAR = 29.886


def test_paired_test_detects_a_clear_synthetic_effect():
    # Synthetic near/far pairs with an obvious, consistent elevation --
    # confirms the test mechanics themselves (Wilcoxon direction, ratios)
    # before trusting them on real data.
    rng = np.random.default_rng(0)
    near = 2.0 + 0.1 * rng.standard_normal(15)
    far = 1.0 + 0.1 * rng.standard_normal(15)
    result = paired_near_vs_far_test({"near": near, "far": far})
    assert result["ratio_mean"] > 1.5
    assert result["p_value"] < 0.01


def test_paired_test_finds_no_effect_when_none_exists():
    rng = np.random.default_rng(1)
    near = 1.0 + 0.2 * rng.standard_normal(15)
    far = 1.0 + 0.2 * rng.standard_normal(15)
    result = paired_near_vs_far_test({"near": near, "far": far})
    assert result["p_value"] > 0.05


def test_run_and_summarize_uses_seed_as_replicate_not_per_point():
    # Small, fast smoke test (few seeds, few epochs) -- just confirms the
    # pipeline runs end to end and returns one (near, far) pair PER SEED,
    # which is the actual fix (Sec. 31's bug was treating every (seed, Nsub)
    # point as independent).
    data = generate_stability_dataset(FR, LAM, KI, KE, N1, nsub_range=(24.0, 36.0), n_nsub=30)
    nsub_grid = np.linspace(24.5, 35.5, 15)
    results, per_seed = run_and_summarize(
        data, FR, LAM, KI, KE, N1, NSUB_STAR, nsub_grid, n_seeds=4, n_epochs=800,
    )
    for metric in ("fold_error", "hopf_error", "eu_field_err", "g_field_err"):
        assert len(per_seed[metric]["near"]) == 4
        assert len(per_seed[metric]["far"]) == 4
        assert results[metric]["n"] <= 4
