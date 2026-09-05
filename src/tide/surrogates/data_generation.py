"""Phase 3 Stage 1 training data: (Nsub, Npch) -> (Eu, leading eigenvalue real
part), at fixed channel/friction parameters (Fr, Lam, ki, ke) and the canonical
N1=16 model (PROJECT_LOG.md Sec. 22/26).

Why these two targets, not the steady state directly: the steady state
(lambda0, m0, ui0, node positions) is CLOSED-FORM in (Nsub, Npch) -- see
steady_state_general -- so a surrogate "learning" it would just be fitting an
exactly-known algebraic formula, not a meaningful stand-in for the kind of
expensive-to-evaluate field a real surrogate replaces. The two genuinely
non-trivial scalar fields a surrogate has to earn are:
  - Eu(Nsub, Npch): the internal characteristic curve (Eq. 26). Its Nsub-wise
    local maximum over Npch IS the fold boundary (curves.fold_npch finds it by
    dEu/dNpch=0). A surrogate can have tiny pointwise error on Eu everywhere
    and still badly mislocate the fold if it doesn't reproduce the curve's
    flatness near the extremum correctly -- exactly the ill-conditioning this
    project's thesis is about.
  - g(Nsub, Npch) = max(Re(eigenvalues)) among the leading COMPLEX-CONJUGATE
    PAIR only (NOT all eigenvalues -- see Sec. 27/28 below) of the N1=16
    Jacobian at the steady state: the linear stability margin for the
    oscillatory (DWO) mode specifically. Its Nsub-wise zero-crossing over
    Npch IS the Hopf boundary.
Critically (established in Sec. 16, re-confirmed here): the fold is NOT a
zero-eigenvalue event of this Jacobian at fixed (Nsub, Npch) -- g does not
cross zero at the fold. So Eu and g are both needed; neither one alone
captures the full fold+Hopf+codim-2 structure a surrogate must get right.

**Why g is restricted to the complex pair, not max over ALL eigenvalues
(Sec. 27-28 finding, discovered while first building this module):** this
model also carries a persistent REAL eigenvalue ("excursive instability," a
documented Ledinegg-type static instability that Clausse & Lahey's own work
describes as co-existing with DWO in this model family -- confirmed via live
literature search, not assumed) that dominates `max(Re(all eigenvalues))`
throughout this entire (Nsub, Npch) region and never changes sign here at all
-- using it as the surrogate target would make the fold/Hopf boundary
invisible to the learning task entirely (confirmed directly: an earlier
version of this module used the all-eigenvalue max and every single one of
4800 generated points came back "unstable," rather than straddling the true
boundary). Restricting to the leading complex pair recovers the correct
fold/Hopf-only signal this project's Phase 3 experiment is actually scoped to
test -- excursive instability is real and co-occurs here, but it is out of
scope for this specific experiment (stated explicitly in the manuscript per
Sec. 28's decision, not silently ignored).

Sampling deliberately straddles both boundaries and their crossing (the
codim-2 point), not just background coverage -- per PROJECT_LOG.md Sec. 5's
explicit warning that sampling only the deep-stable region makes any negative
result trivial. Below Nsub* (~29.886 at N1=16, Sec. 26) Hopf sits below fold
(Hopf is the operative boundary); above Nsub* it flips (fold is operative) --
confirmed directly by evaluating both at a few Nsub values before designing
the sampler, not assumed from the codim-2 point's existence alone.
"""

import numpy as np
from scipy.stats import qmc

from tide.continuation.codim2_convergence import _fold, hopf_npch_general
from tide.continuation.pseudo_arclength import _eigvals_jit
from tide.physics.ledinegg_curve import euler_number


def _leading_real_part(nsub, npch, Fr, Lam, ki, ke, N1):
    """Leading real part among the COMPLEX-conjugate eigenvalue pair only --
    NOT max over all eigenvalues (see this module's docstring, Sec. 27-28):
    the persistent real "excursive instability" eigenvalue dominates the raw
    max everywhere in this region and never changes sign, which would make
    the actual fold/Hopf boundary invisible to a surrogate trained on it.
    Returns None if no complex pair exists or the eigendecomposition fails."""
    eigvals = np.array(_eigvals_jit(nsub, npch, Fr, Lam, ki, ke, N1))
    if not np.all(np.isfinite(eigvals)):
        return None
    complex_eigs = eigvals[np.abs(eigvals.imag) > 1e-9]
    if len(complex_eigs) == 0:
        return None
    return float(np.max(complex_eigs.real))


def _boundaries_at(nsub, Fr, Lam, ki, ke, N1):
    """(fold, hopf) at this Nsub, or (fold, None) if no Hopf bracket is found
    (can happen far from the codim-2 region where the fast fold estimate and
    the true Hopf location are no longer close).

    Bracket width matters here: a wider (fold*0.7, fold*1.6) bracket was tried
    first and silently returned None everywhere in the Nsub~24-36 range,
    because the leading-COMPLEX-pair real part isn't monotonic that far out --
    it has a second, unrelated sign change (near fold*0.8, a real/complex
    eigenvalue reshuffling unrelated to the Hopf boundary itself, see
    PROJECT_LOG.md Sec. 27's investigation into this model's eigenvalue
    structure), so a bracket wide enough to include it has both endpoints on
    the same sign and bisection refuses to run. Narrowing to (0.85, 1.4)
    excludes that second crossing while still safely bracketing the true Hopf
    point at every Nsub checked (24-36)."""
    fold = _fold(nsub, Fr, Lam, ki, ke)
    if fold is None:
        return None, None
    try:
        hopf = hopf_npch_general(nsub, Fr, Lam, ki, ke, N1, bracket=(fold * 0.85, fold * 1.4))
    except Exception:
        hopf = None
    return fold, hopf


def generate_stability_dataset(Fr, Lam, ki, ke, N1, nsub_range, n_nsub=60,
                                n_boundary_per_nsub=30, n_background_per_nsub=20,
                                band_frac=0.20, background_frac=(1.05, 1.8),
                                seed=0):
    """Generate the (Nsub, Npch) -> (Eu, g) training set.

    For each of n_nsub Nsub values (evenly spaced over nsub_range):
      - locate fold(Nsub) and hopf(Nsub) (may be None if out of bracket range)
      - "boundary band" samples: Latin-hypercube Npch within
        +/-band_frac of EACH located boundary (both fold and Hopf, so the
        codim-2 crossing region -- where they nearly coincide -- gets doubled
        density, not accidentally under-sampled)
      - "background" samples: Latin-hypercube Npch across
        [background_frac[0]*Nsub, background_frac[1]*max(fold,hopf)], for
        global coverage (deep-stable and deep-unstable regions)

    Returns a dict of numpy arrays: nsub, npch, Eu, g, fold_at_nsub, hopf_at_nsub
    (the latter two repeated per-row, for convenience when checking boundary
    error later -- NOT used as training targets).
    """
    rng = np.random.default_rng(seed)
    nsub_values = np.linspace(nsub_range[0], nsub_range[1], n_nsub)

    rows_nsub, rows_npch, rows_eu, rows_g, rows_fold, rows_hopf = [], [], [], [], [], []

    for nsub in nsub_values:
        fold, hopf = _boundaries_at(nsub, Fr, Lam, ki, ke, N1)
        if fold is None:
            continue

        npch_samples = []
        for boundary in (fold, hopf):
            if boundary is None:
                continue
            lo, hi = boundary * (1 - band_frac), boundary * (1 + band_frac)
            sampler = qmc.LatinHypercube(d=1, seed=rng.integers(0, 2**32 - 1))
            u = sampler.random(n_boundary_per_nsub).ravel()
            npch_samples.append(lo + u * (hi - lo))

        hi_ref = max(fold, hopf) if hopf is not None else fold
        lo_bg, hi_bg = background_frac[0] * nsub, background_frac[1] * hi_ref
        sampler = qmc.LatinHypercube(d=1, seed=rng.integers(0, 2**32 - 1))
        u = sampler.random(n_background_per_nsub).ravel()
        npch_samples.append(lo_bg + u * (hi_bg - lo_bg))

        npch_values = np.concatenate(npch_samples)
        npch_values = npch_values[npch_values > nsub * 1.001]  # stay in valid domain

        for npch in npch_values:
            g = _leading_real_part(nsub, npch, Fr, Lam, ki, ke, N1)
            if g is None:
                continue
            eu = float(euler_number(npch, nsub, Fr, Lam, ki, ke))
            rows_nsub.append(nsub)
            rows_npch.append(npch)
            rows_eu.append(eu)
            rows_g.append(g)
            rows_fold.append(fold)
            rows_hopf.append(hopf if hopf is not None else np.nan)

    return dict(
        nsub=np.array(rows_nsub), npch=np.array(rows_npch),
        Eu=np.array(rows_eu), g=np.array(rows_g),
        fold_at_nsub=np.array(rows_fold), hopf_at_nsub=np.array(rows_hopf),
    )
