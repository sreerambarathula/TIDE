"""Steady-state internal characteristic curve (Euler number vs. phase-change number)
for a uniformly heated vertical boiling channel, homogeneous-equilibrium, single-node
(no spatial discretization needed -- this is the exact steady-state solution).

Source, verified by direct page-image transcription (term-by-term, twice): G. Theler,
A. Clausse, F. Bonetto, "The moving boiling-boundary model of a vertical two-phase
flow channel revisited," Mecanica Computacional Vol. XXIX, pp. 3949-3976 (2010),
Eq. (26), p. 3963. This paper re-derives the Clausse & Lahey (1991) model from the
continuous conservation equations with every intermediate step shown -- no undefined
symbols, unlike the Hurley (2023) and Verma & Iyer (2023) sources checked earlier
(see PROJECT_LOG.md Sec. 12).

Eu = (1/Npch)(Nsub^2 + 0.5*Lambda*Nsub^2 + ke*Nsub^2)
   + (1/Npch^2)(-Nsub^3 + Lambda*Nsub^2 - Lambda*Nsub^3 + ki*Nsub^2 + ke*Nsub^2 - ke*Nsub^3)
   + (Nsub/Npch)*(1/Fr)*(1 + ln(1 + Npch - Nsub)/Nsub)
   + 0.5*(Nsub^4/Npch^3)*Lambda

Eu (Euler number) is the non-dimensional external pressure drop; Npch is the
phase-change number, inversely proportional to the steady inlet velocity at fixed
heat input -- so sweeping Npch at fixed Nsub, Fr, Lambda, ki, ke traces exactly the
classical Ledinegg internal characteristic curve (pressure drop vs. flow rate at
fixed power). Non-monotonicity in this curve is the fold/Ledinegg condition.

Valid only for Npch > Nsub (otherwise there is no two-phase flow at all).
"""

import jax.numpy as jnp


def euler_number(n_pch, n_sub, Fr, Lam, k_in, k_out):
    """Eq. (26): Eu(Npch) at fixed Nsub and channel/friction parameters."""
    n_pch = jnp.asarray(n_pch, dtype=jnp.float64)

    term1 = (1.0 / n_pch) * (n_sub**2 + 0.5 * Lam * n_sub**2 + k_out * n_sub**2)
    term2 = (1.0 / n_pch**2) * (
        -n_sub**3 + Lam * n_sub**2 - Lam * n_sub**3
        + k_in * n_sub**2 + k_out * n_sub**2 - k_out * n_sub**3
    )
    term3 = (n_sub / n_pch) * (1.0 / Fr) * (1.0 + jnp.log(1.0 + n_pch - n_sub) / n_sub)
    term4 = 0.5 * (n_sub**4 / n_pch**3) * Lam

    return term1 + term2 + term3 + term4
