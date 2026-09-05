"""Validate the transcribed HEM boundary (Hurley 2023, Eq. 3.30-3.31) against the
two qualitative claims stated in the source text (p. 40-42):

1. The N_pch-vs-N_sub boundary has positive slope at high subcooling and negative
   slope at low subcooling (text says the crossover is near N_sub ~ 2, discussed
   relative to N_sub < 2 for the Ishii/Saha models specifically -- the HEM curve's
   own crossover is checked here on its own terms, not assumed to be at exactly 2).
2. Increasing the inlet k-factor is stabilizing (boundary N_pch shifts up: more
   power needed to destabilize at fixed subcooling); increasing the outlet k-factor
   is destabilizing (boundary N_pch shifts down).
"""

import numpy as np

import tide  # noqa: F401  (enables float64)
from tide.physics.hem_boundary import epsilon, npch_boundary


def test_slope_changes_sign():
    n_sub = np.linspace(0.3, 6.0, 400)
    eps = epsilon(k_in=1.0, k_out=1.0)
    n_pch = np.asarray(npch_boundary(n_sub, eps))
    slope = np.gradient(n_pch, n_sub)

    assert slope[-1] > 0, "expected positive slope at high subcooling"
    assert slope[0] < 0, "expected negative slope at low subcooling"
    sign_changes = np.sum(np.diff(np.sign(slope)) != 0)
    assert sign_changes == 1, f"expected exactly one slope sign change, found {sign_changes}"


def test_inlet_restriction_is_stabilizing():
    n_sub = np.linspace(0.5, 5.0, 50)
    eps_low = epsilon(k_in=1.0, k_out=1.0)
    eps_high_kin = epsilon(k_in=4.0, k_out=1.0)

    n_pch_low = np.asarray(npch_boundary(n_sub, eps_low))
    n_pch_high_kin = np.asarray(npch_boundary(n_sub, eps_high_kin))

    assert np.all(n_pch_high_kin > n_pch_low), (
        "increasing inlet k-factor should raise the boundary N_pch "
        "(more power needed to destabilize) -- stabilizing, per Hurley Fig. 3.7"
    )


def test_outlet_restriction_is_destabilizing():
    # NOTE: with this closed-form eps = 2(k_in+k_out)/(k_out+1), d(eps)/d(k_out)
    # has sign (1 - k_in) -- the destabilizing-outlet trend only holds for
    # k_in > 1. At exactly k_in = 1 the two curves are algebraically identical
    # (degenerate edge case, confirmed numerically, see PROJECT_LOG.md Sec. 11).
    # k_in = 3.0 sits in the realistic range the dissertation validates against
    # (cited Saha data: inlet k-factors 2.85-6.55).
    n_sub = np.linspace(0.5, 5.0, 50)
    eps_low = epsilon(k_in=3.0, k_out=1.0)
    eps_high_kout = epsilon(k_in=3.0, k_out=4.0)

    n_pch_low = np.asarray(npch_boundary(n_sub, eps_low))
    n_pch_high_kout = np.asarray(npch_boundary(n_sub, eps_high_kout))

    assert np.all(n_pch_high_kout < n_pch_low), (
        "increasing outlet k-factor should lower the boundary N_pch "
        "(less power needed to destabilize) -- destabilizing, per Hurley Fig. 3.7"
    )


def test_outlet_effect_sign_depends_on_inlet_k_factor():
    """Documents a real property of the closed-form correlation, not a test of
    physical correctness: d(eps)/d(k_out) has sign (1 - k_in), so the
    "outlet restriction is destabilizing" trend is conditional, not universal,
    under this simplified HEM formula. k_in = 1 is the exact degenerate point."""
    n_sub = np.linspace(0.5, 5.0, 20)

    eps_below = epsilon(k_in=0.5, k_out=1.0)
    eps_below_more_kout = epsilon(k_in=0.5, k_out=4.0)
    below = np.asarray(npch_boundary(n_sub, eps_below))
    below_more = np.asarray(npch_boundary(n_sub, eps_below_more_kout))
    assert np.all(below_more > below), "k_in < 1: outlet restriction flips to stabilizing here"

    eps_at1 = epsilon(k_in=1.0, k_out=1.0)
    eps_at1_more_kout = epsilon(k_in=1.0, k_out=4.0)
    assert eps_at1 == eps_at1_more_kout, "k_in == 1 is the exact degenerate case"
