"""Misfit-driven adaptive sampling (PROJECT_LOG.md Sec. 34), built to resolve
the confound §32 found: the original "boundary-band" data generator
(data_generation.py) assumed a priori that the codim-2 region needs denser
training coverage, which made it impossible to tell whether elevated
boundary error near codim-2 was a real surrogate-fitting phenomenon or an
artifact of that assumption. Removing the assumption (plain uniform
sampling, §32) made the whole effect vanish -- but that doesn't prove the
effect was fake either, since uniform sampling might just under-serve the
hard region in a different, uncontrolled way.

This module follows the two-stage template from a live literature search
(Sec. 33): "Goal-Driven Adaptive Sampling Strategies for Machine Learning
Models Predicting Fields" (arXiv 2601.21832, Jan 2026) -- start with uniform
coverage, THEN add points where a trained model's MEASURED misfit (actual
error against ground truth) is largest, not where a human assumes it will be
largest. If the misfit-driven process naturally clusters extra points near
the codim-2 point, that is itself independent evidence the region is
genuinely harder to fit -- decided by measured error, not geometric
assumption.
"""

import numpy as np

from tide.continuation.codim2_convergence import _fold
from tide.physics.ledinegg_curve import euler_number
from tide.surrogates.data_generation import _leading_real_part
from tide.surrogates.mlp import train_surrogate


def _sample_candidate_pool(nsub_range, n_candidates, Fr, Lam, ki, ke, npch_mult=1.8, seed=0):
    """Plain uniform candidate pool over the domain -- no boundary
    assumption. Npch is sampled relative to a locally-appropriate upper
    bound (npch_mult * fold(nsub)) purely to stay in a physically sensible
    operating range, not to bias toward the boundary specifically."""
    rng = np.random.default_rng(seed)
    nsub = rng.uniform(nsub_range[0], nsub_range[1], n_candidates)
    npch = np.empty_like(nsub)
    for i, n in enumerate(nsub):
        fold = _fold(n, Fr, Lam, ki, ke)
        hi = npch_mult * fold if fold is not None else n * 3.0
        npch[i] = rng.uniform(n * 1.02, hi)
    return nsub, npch


def measure_misfit_and_augment(base_data, Fr, Lam, ki, ke, N1, nsub_range,
                                n_candidates=2000, n_add=600, seed=0, n_epochs=3000):
    """Stage 2 of the adaptive scheme: train a surrogate on base_data (Stage
    1, plain uniform), measure its misfit against ground truth on a large
    uniform candidate pool, and add the n_add highest-misfit candidates to
    the dataset. Returns (augmented_data, diagnostics) where diagnostics
    records the Nsub distribution of added points -- the thing that answers
    "does the codim-2 region attract disproportionate misfit-driven density"
    without assuming the answer."""
    surrogate, _ = train_surrogate(base_data, n_epochs=n_epochs, seed=seed)

    cand_nsub, cand_npch = _sample_candidate_pool(nsub_range, n_candidates, Fr, Lam, ki, ke, seed=seed)
    eu_true = np.array([euler_number(p, n, Fr, Lam, ki, ke) for n, p in zip(cand_nsub, cand_npch)])
    g_true_list = [_leading_real_part(n, p, Fr, Lam, ki, ke, N1) for n, p in zip(cand_nsub, cand_npch)]
    valid = np.array([v is not None for v in g_true_list])
    g_true = np.array([v if v is not None else np.nan for v in g_true_list])

    eu_pred, g_pred = surrogate.predict(cand_nsub, cand_npch)
    eu_pred, g_pred = np.asarray(eu_pred), np.asarray(g_pred)

    # Normalized combined misfit (Eu and g differ ~20x in scale, Sec. 30) --
    # using the surrogate's own training-set y_std so the two error terms
    # contribute comparably, same normalization logic as the training loss.
    eu_err_norm = (eu_pred - eu_true) / float(surrogate.y_std[0])
    g_err_norm = np.where(valid, (g_pred - g_true) / float(surrogate.y_std[1]), 0.0)
    misfit = eu_err_norm**2 + g_err_norm**2

    top_idx = np.argsort(-misfit)[:n_add]
    added_nsub, added_npch = cand_nsub[top_idx], cand_npch[top_idx]
    added_eu = eu_true[top_idx]
    added_g = np.array([g_true_list[i] if g_true_list[i] is not None else np.nan for i in top_idx])

    valid_added = np.isfinite(added_g)
    augmented = {
        "nsub": np.concatenate([base_data["nsub"], added_nsub[valid_added]]),
        "npch": np.concatenate([base_data["npch"], added_npch[valid_added]]),
        "Eu": np.concatenate([base_data["Eu"], added_eu[valid_added]]),
        "g": np.concatenate([base_data["g"], added_g[valid_added]]),
        "fold_at_nsub": np.concatenate([base_data["fold_at_nsub"],
                                         np.full(valid_added.sum(), np.nan)]),
        "hopf_at_nsub": np.concatenate([base_data["hopf_at_nsub"],
                                         np.full(valid_added.sum(), np.nan)]),
    }

    diagnostics = dict(
        added_nsub=added_nsub[valid_added],
        added_misfit=misfit[top_idx][valid_added],
        n_added=int(valid_added.sum()),
    )
    return augmented, diagnostics
