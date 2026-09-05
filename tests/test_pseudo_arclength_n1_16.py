"""Validates the Hopf curve tracer at N1=16 -- the corrected canonical
reference (PROJECT_LOG.md Sec. 25-26), replacing N1=4 (Sec. 21/23) after the
node-count convergence study found N1=4 was not converged (~5% off in Nsub*).

Same three-way validation standard as test_pseudo_arclength.py's N1=4 tests,
re-run here for N1=16. Kept small (few steps) for the same reason -- each step
now costs ~15ms after Sec. 24's JIT fix, but there's no need for more than a
handful of points to confirm correctness.
"""

import numpy as np

import tide  # noqa: F401
from tide.continuation.pseudo_arclength import trace_hopf_curve

# Real facility parameters, N1=16 -- the codim-2 point is at Nsub~29.886 here
# (PROJECT_LOG Sec. 26), not Nsub~31.36 (that was the N1=4 location, Sec. 21).
FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
N1 = 16


def test_first_traced_point_matches_independent_newton_bisection():
    # Cross-check against the value found independently in Sec. 26's precision
    # refinement: at Nsub=30.0, hopf Npch = 49.8707 (from a completely separate
    # Newton bisection via codim2_convergence.hopf_npch_general, not this module).
    trace = trace_hopf_curve(30.0, 50.2, FR, LAM, KI, KE, N1, dnsub=0.1, n_steps=0)
    assert len(trace) == 1
    assert abs(trace[0]["npch"] - 49.8707) < 0.01


def test_curve_is_smooth_and_monotonic_over_a_short_stretch():
    trace = trace_hopf_curve(30.0, 50.2, FR, LAM, KI, KE, N1, dnsub=0.1, n_steps=5)
    assert len(trace) == 6
    npchs = [pt["npch"] for pt in trace]
    omegas = [pt["omega"] for pt in trace]
    assert all(b > a for a, b in zip(npchs, npchs[1:]))
    assert all(b < a for a, b in zip(omegas, omegas[1:]))
    for a, b in zip(npchs, npchs[1:]):
        assert abs(b - a) < 2.0


def test_traced_toward_codim2_point_matches_independent_fold_value():
    # Tracing backward (decreasing Nsub) toward the codim-2 point should land
    # near the precision-refined codim-2 value at Nsub*~29.886, Npch*~49.702
    # (Sec. 26).
    trace = trace_hopf_curve(30.0, 50.2, FR, LAM, KI, KE, N1, dnsub=0.02, n_steps=6, direction=-1.0)
    last = trace[-1]
    assert abs(last["nsub"] - 29.88) < 1e-6
    assert abs(last["npch"] - 49.6932) < 0.01
