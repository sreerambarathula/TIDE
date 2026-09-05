"""Validates the Hopf curve tracer (PROJECT_LOG.md Sec. 23) against independently
computed reference values -- not just "it runs without crashing."

Kept intentionally small (few steps) since each step costs ~5-8s unbatched
(bisection with continuous eigenvalue tracking, not yet JIT-optimized -- see the
module docstring's "not yet done" note).
"""

import numpy as np

import tide  # noqa: F401
from tide.continuation.pseudo_arclength import trace_hopf_curve

# Real facility parameters, N1=4 -- the codim-2 point is at Nsub~31.36 here
# (PROJECT_LOG Sec. 21).
FR, LAM, KI, KE = 0.035, 5.90, 6.55, 2.03
N1 = 4


def test_first_traced_point_matches_independent_newton_bisection():
    # Cross-check against the value found independently in Sec. 21's manual
    # precision refinement: at Nsub=31.5, hopf Npch = 53.325 (from a completely
    # separate Newton bisection, not this module).
    trace = trace_hopf_curve(31.5, 53.4, FR, LAM, KI, KE, N1, dnsub=0.1, n_steps=0)
    assert len(trace) == 1
    assert abs(trace[0]["npch"] - 53.325) < 0.01


def test_curve_is_smooth_and_monotonic_over_a_short_stretch():
    trace = trace_hopf_curve(31.5, 53.4, FR, LAM, KI, KE, N1, dnsub=0.1, n_steps=5)
    assert len(trace) == 6
    npchs = [pt["npch"] for pt in trace]
    omegas = [pt["omega"] for pt in trace]
    # Npch should increase monotonically with Nsub here, and omega decrease --
    # both smoothly, not jumping around (the failure mode two earlier attempts
    # had before being fixed).
    assert all(b > a for a, b in zip(npchs, npchs[1:]))
    assert all(b < a for a, b in zip(omegas, omegas[1:]))
    # no single step should move Npch by an implausible amount for dnsub=0.1
    for a, b in zip(npchs, npchs[1:]):
        assert abs(b - a) < 2.0


def test_traced_toward_codim2_point_matches_independent_fold_value():
    # Tracing backward (decreasing Nsub) toward the codim-2 point should land
    # near the independently-found fold value at Nsub=31.35 (~53.099, Sec. 21).
    trace = trace_hopf_curve(31.5, 53.4, FR, LAM, KI, KE, N1, dnsub=0.05, n_steps=3, direction=-1.0)
    last = trace[-1]
    assert abs(last["nsub"] - 31.35) < 1e-6
    assert abs(last["npch"] - 53.1096) < 0.01
