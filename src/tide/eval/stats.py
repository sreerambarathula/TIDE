"""Seed-level statistics for the decoupling analysis (PROJECT_LOG.md Sec. 32).

Fixes a real pseudo-replication bug in the first version of this analysis
(Sec. 31): treating every (seed, Nsub) evaluation point as an independent
sample for a Mann-Whitney U test, when the 20 "seeds" all trained on the SAME
underlying dataset (only the train/test split and network init differed),
and within one trained network the near/far evaluation points are spatially
correlated (a smooth surrogate's error at one Nsub predicts its error at a
nearby Nsub). That inflated significance -- p-values like 1e-18 were treating
~400 correlated points as if they were 400 independent draws.

The fix: SEED is the unit of independent replication, not (seed, Nsub). For
each seed, collapse to one near-mean and one far-mean per metric (exactly
what run_decoupling_analysis.summary already returns), then run a PAIRED
test (Wilcoxon signed-rank, appropriate for n~20 paired, non-normal
differences) across seeds on the (near - far) difference. This is a much
smaller, much more honest effective sample size.
"""

import numpy as np
from scipy import stats

from tide.eval.decoupling import run_decoupling_analysis
from tide.surrogates.mlp import train_surrogate, train_test_split

_METRICS = ("fold_error", "hopf_error", "eu_field_err", "g_field_err")


def run_seeded_decoupling_study(data, Fr, Lam, ki, ke, N1, nsub_star, nsub_grid,
                                 n_seeds=20, n_epochs=3000, hidden_sizes=(64, 64)):
    """Trains n_seeds independent surrogates (independent train/test split
    AND independent network init per seed) on the same dataset, runs the
    decoupling analysis for each, and returns per-seed near/far means for
    every metric -- SEED is the replicate unit, not (seed, Nsub)."""
    per_seed = {m: {"near": [], "far": []} for m in _METRICS}
    for seed in range(n_seeds):
        train, _ = train_test_split(data, seed=seed)
        surrogate, _ = train_surrogate(train, hidden_sizes=hidden_sizes, n_epochs=n_epochs, seed=seed)
        _, summary = run_decoupling_analysis(surrogate, Fr, Lam, ki, ke, N1, nsub_star,
                                              nsub_grid, train_data=train)
        for m in _METRICS:
            per_seed[m]["near"].append(summary[f"{m}_near"])
            per_seed[m]["far"].append(summary[f"{m}_far"])
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in per_seed.items()}


def paired_near_vs_far_test(per_seed_metric):
    """Wilcoxon signed-rank test (one-sided, near > far) on paired per-seed
    near/far means for one metric. Returns ratio-of-medians, ratio-of-means,
    and the p-value -- report both ratios since a mean/median disagreement is
    itself informative (Sec. 31 found exactly this for hopf_error: a real,
    modest effect, not a dramatic one)."""
    near, far = per_seed_metric["near"], per_seed_metric["far"]
    valid = np.isfinite(near) & np.isfinite(far)
    near, far = near[valid], far[valid]
    if len(near) == 0:
        # All pairs were invalid (NaN/inf) -- this is a data problem, not a
        # "no difference found" result, and must not be conflated with one
        # (found via external audit: np.all(diff==0) is vacuously True on
        # an empty array, so this used to silently return a clean null).
        raise ValueError("paired_near_vs_far_test: no valid (finite) near/far "
                          "pairs after filtering -- cannot compute a p-value.")
    diff = near - far
    if np.all(diff == 0):
        return dict(n=len(diff), ratio_mean=1.0, ratio_median=1.0, p_value=1.0)
    stat, p = stats.wilcoxon(diff, alternative="greater")
    return dict(
        n=len(diff),
        ratio_mean=float(np.mean(near) / np.mean(far)),
        ratio_median=float(np.median(near) / np.median(far)),
        p_value=float(p),
    )


def run_and_summarize(data, Fr, Lam, ki, ke, N1, nsub_star, nsub_grid,
                       n_seeds=20, n_epochs=3000, hidden_sizes=(64, 64)):
    """Convenience wrapper: run the seeded study and the paired test for
    every metric in one call, returning a dict of results ready to print or
    log."""
    per_seed = run_seeded_decoupling_study(
        data, Fr, Lam, ki, ke, N1, nsub_star, nsub_grid,
        n_seeds=n_seeds, n_epochs=n_epochs, hidden_sizes=hidden_sizes,
    )
    return {m: paired_near_vs_far_test(per_seed[m]) for m in _METRICS}, per_seed


def run_mechanism_study(data, Fr, Lam, ki, ke, N1, nsub_grid,
                         n_seeds=20, n_epochs=3000, hidden_sizes=(64, 64)):
    """Tests two candidate mechanisms for hopf_error, per-seed (not pooled --
    Sec. 31's pooled Spearman test suffered the same pseudo-replication
    problem as the primary test, since it treated (seed, Nsub) pairs as
    independent): (1) the classical ill-conditioning hypothesis, 1/||grad
    g||; (2) the ML-standard alternative, distance to the nearest training
    point. For each seed, computes both Spearman correlations across that
    seed's own Nsub grid, then reports how consistent the sign and
    significance are ACROSS seeds -- a mechanism should show up in most
    seeds, not just some."""
    from scipy import stats as _stats

    from tide.eval.decoupling import run_decoupling_analysis
    from tide.surrogates.mlp import train_surrogate, train_test_split

    grad_rhos, grad_ps = [], []
    dist_rhos, dist_ps = [], []
    for seed in range(n_seeds):
        train, _ = train_test_split(data, seed=seed)
        surrogate, _ = train_surrogate(train, hidden_sizes=hidden_sizes, n_epochs=n_epochs, seed=seed)
        per_point, _ = run_decoupling_analysis(
            surrogate, Fr, Lam, ki, ke, N1, nsub_star=0.0,  # near/far unused here
            nsub_grid=nsub_grid, train_data=train,
        )
        finite = np.isfinite(per_point["hopf_error"]) & np.isfinite(per_point["grad_norm_at_hopf"]) & (per_point["grad_norm_at_hopf"] > 0)
        if finite.sum() > 5:
            rho, p = _stats.spearmanr(per_point["hopf_error"][finite], 1.0 / per_point["grad_norm_at_hopf"][finite])
            grad_rhos.append(rho)
            grad_ps.append(p)
        finite2 = np.isfinite(per_point["hopf_error"]) & np.isfinite(per_point["train_dist_at_hopf"])
        if finite2.sum() > 5:
            rho2, p2 = _stats.spearmanr(per_point["hopf_error"][finite2], per_point["train_dist_at_hopf"][finite2])
            dist_rhos.append(rho2)
            dist_ps.append(p2)

    def _summarize(rhos, ps):
        rhos, ps = np.array(rhos), np.array(ps)
        return dict(
            n_seeds=len(rhos), median_rho=float(np.median(rhos)),
            frac_positive_significant=float(np.mean((rhos > 0) & (ps < 0.05))),
            frac_negative_significant=float(np.mean((rhos < 0) & (ps < 0.05))),
        )

    return dict(
        gradient_mechanism=_summarize(grad_rhos, grad_ps),
        data_density_mechanism=_summarize(dist_rhos, dist_ps),
    )
