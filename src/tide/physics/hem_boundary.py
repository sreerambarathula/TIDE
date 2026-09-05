"""Closed-form homogeneous-equilibrium-model (HEM) DWO neutral-stability boundary.

Source (verified by direct transcription from the rendered PDF page, not OCR text
extraction — see PROJECT_LOG.md Sec. 11): P. Hurley, "Density-Wave Instability
Characterization in Boiling Water Reactors under MELLLA+ Domain during ATWS,"
Ph.D. dissertation, Virginia Tech, 2023, Eq. (3.30)-(3.31), p. 41. Hurley attributes
the underlying formulation to Guido et al. [70] (citation not independently
verified — only the transcription from Hurley is).

This closed-form relation gives the density-wave-oscillation (Hopf) neutral
stability boundary N_pch(N_sub; eps) directly -- it is a validation anchor for
Phase 1, not the dynamical system itself. It says nothing about Ledinegg (fold)
instability; that requires the pressure-drop characteristic curve (Hurley Eqs.
3.15-3.18), not yet implemented -- see PROJECT_LOG.md open items.
"""

import jax.numpy as jnp


def epsilon(k_in: float, k_out: float) -> float:
    """Friction-factor coefficient combining inlet/outlet loss (k) factors. Eq. (3.31)."""
    return 2.0 * (k_in + k_out) / (k_out + 1.0)


def npch_boundary(n_sub, eps):
    """Neutral-stability phase-change number N_pch as a function of subcooling
    number N_sub and the k-factor coefficient eps. Eq. (3.30).

    N_pch = N_sub + (eps/2)(1 + 2/N_sub) - 5/2
            + sqrt{ [ (eps/2)(1 + 2/N_sub) - 5/2 ]^2 + eps }
    """
    n_sub = jnp.asarray(n_sub, dtype=jnp.float64)
    inner = (eps / 2.0) * (1.0 + 2.0 / n_sub) - 5.0 / 2.0
    return n_sub + inner + jnp.sqrt(inner**2 + eps)
