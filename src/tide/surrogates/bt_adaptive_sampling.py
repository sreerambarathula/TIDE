"""Phase 4 candidate fix (PROJECT_LOG.md Sec. 41): misfit-driven adaptive
sampling applied to a genuine Bogdanov-Takens point, testing whether it
closes the decoupling gap Phase 3 found there.

Why this candidate, not the originally-planned boundary-targeted loss
(PROJECT_LOG.md Sec. 3/7): Sec. 39 found the mechanism behind BT-point
decoupling is a REPRESENTABILITY limit (the stable window narrows below
what a fixed-capacity smooth regressor can resolve), not the originally
hypothesized vanishing-gradient ill-conditioning. A loss reweighting doesn't
add resolution where the network lacks it; targeted data density might.
This reuses the exact two-stage misfit-driven scheme already validated in
Sec. 34 (adaptive_sampling.py), adapted to bt_point_data.py's dataset format
and g-definition (max over ALL eigenvalues, not complex-pair-only).
"""

import numpy as np

from tide.continuation.codim2_convergence import _fold
from tide.physics.ledinegg_curve import euler_number
from tide.surrogates.bt_point_data import _g
from tide.surrogates.mlp import train_surrogate


def _sample_candidate_pool_bt(nsub_range, n_candidates, Fr, Lam, ki, ke, npch_mult=1.6, seed=0):
    """Plain uniform candidate pool over the domain, same spirit as
    adaptive_sampling.py's version -- no boundary assumption baked in."""
    rng = np.random.default_rng(seed)
    nsub = rng.uniform(nsub_range[0], nsub_range[1], n_candidates)
    npch = np.empty_like(nsub)
    for i, n in enumerate(nsub):
        fold = _fold(n, Fr, Lam, ki, ke)
        hi = npch_mult * fold if fold is not None else n * 3.0
        npch[i] = rng.uniform(n * 1.02, hi)
    return nsub, npch


def measure_misfit_and_augment_bt(base_data, Fr, Lam, ki, ke, N1, nsub_range,
                                   n_candidates=2000, n_add=600, seed=0, n_epochs=3000):
    """Stage 2 of the adaptive scheme for a BT-point dataset: train on
    base_data, measure misfit on a uniform candidate pool, add the
    highest-misfit points. Returns (augmented_data, diagnostics)."""
    surrogate, _ = train_surrogate(base_data, n_epochs=n_epochs, seed=seed)

    cand_nsub, cand_npch = _sample_candidate_pool_bt(nsub_range, n_candidates, Fr, Lam, ki, ke, seed=seed)
    eu_true = np.array([euler_number(p, n, Fr, Lam, ki, ke) for n, p in zip(cand_nsub, cand_npch)])
    g_true = np.array([_g(n, p, Fr, Lam, ki, ke, N1) for n, p in zip(cand_nsub, cand_npch)])
    valid = np.array([v is not None for v in g_true])
    g_true = np.array([v if v is not None else np.nan for v in g_true])

    eu_pred, g_pred = surrogate.predict(cand_nsub, cand_npch)
    eu_pred, g_pred = np.asarray(eu_pred), np.asarray(g_pred)

    eu_err_norm = (eu_pred - eu_true) / float(surrogate.y_std[0])
    g_err_norm = np.where(valid, (g_pred - g_true) / float(surrogate.y_std[1]), 0.0)
    misfit = eu_err_norm**2 + g_err_norm**2

    top_idx = np.argsort(-misfit)[:n_add]
    added_nsub, added_npch = cand_nsub[top_idx], cand_npch[top_idx]
    added_eu = eu_true[top_idx]
    added_g = g_true[top_idx]

    valid_added = np.isfinite(added_g)
    augmented = {
        "nsub": np.concatenate([base_data["nsub"], added_nsub[valid_added]]),
        "npch": np.concatenate([base_data["npch"], added_npch[valid_added]]),
        "Eu": np.concatenate([base_data["Eu"], added_eu[valid_added]]),
        "g": np.concatenate([base_data["g"], added_g[valid_added]]),
    }
    diagnostics = dict(
        added_nsub=added_nsub[valid_added],
        added_misfit=misfit[top_idx][valid_added],
        n_added=int(valid_added.sum()),
    )
    return augmented, diagnostics
