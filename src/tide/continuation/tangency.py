"""Checks whether a fold-Hopf codim-2 crossing is genuinely degenerate
(tangent, the classical Bogdanov-Takens signature) or merely transversal
(two independently-smooth curves crossing at an angle) -- built to explain
Phase 3's negative decoupling result (PROJECT_LOG.md Sec. 31-34).

Why this matters: the theoretical expectation that ML surrogates should
struggle near a codim-2 point relies on genuine geometric degeneracy there
(a cusp, a tangency, a true double-zero eigenvalue) -- a scalar field like
Eu or g is not inherently hard for a smooth regressor to fit near an
ORDINARY (transversal) intersection of two of its level/extremal sets. This
module checks which case applies here, rather than assuming either.
"""

import numpy as np

from tide.continuation.codim2_convergence import _fold, hopf_npch_general


def crossing_slopes(Fr, Lam, ki, ke, N1, nsub_crossing, h=0.05):
    """At a known fold-Hopf crossing Nsub, returns (slope_fold, slope_hopf,
    |slope_fold - slope_hopf|) -- dNpch/dNsub for each curve via central
    finite differences. A near-zero slope difference indicates tangency
    (degenerate, BT-like); a large one indicates a transversal (ordinary,
    non-degenerate) crossing."""
    fold_lo = _fold(nsub_crossing - h, Fr, Lam, ki, ke)
    fold_hi = _fold(nsub_crossing + h, Fr, Lam, ki, ke)
    if fold_lo is None or fold_hi is None:
        return None
    hopf_lo = hopf_npch_general(nsub_crossing - h, Fr, Lam, ki, ke, N1,
                                 bracket=(fold_lo * 0.85, fold_lo * 1.4))
    hopf_hi = hopf_npch_general(nsub_crossing + h, Fr, Lam, ki, ke, N1,
                                 bracket=(fold_hi * 0.85, fold_hi * 1.4))
    if hopf_lo is None or hopf_hi is None:
        return None
    slope_fold = (fold_hi - fold_lo) / (2 * h)
    slope_hopf = (hopf_hi - hopf_lo) / (2 * h)
    return slope_fold, slope_hopf, abs(slope_fold - slope_hopf)
