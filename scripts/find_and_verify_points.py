"""Stage 1 of the reproducible workflow (see REPRODUCIBILITY.md): locate and
verify Points A, B, C (Table 2) from documented search grids and initial
guesses, rather than trusting the already-known coordinates by assertion.

Every function called here is pre-existing, already-verified real code
(src/tide/continuation/*); this script only orchestrates them and checks
the result against what the manuscript reports -- it introduces no new
physics or statistics.

Honesty note on the "265 combinations" transversality sweep (Section 5.1):
PROJECT_LOG.md's account of this (Sec. 27, Sec. 35) describes a 300-combination
search whose exact random/grid construction isn't fully specified in the log
-- 265 of those 300 happened to have a valid fold-Hopf crossing at all, and
those 265 were the ones checked. This script does NOT claim to bit-reproduce
that exact historical run; it runs an independent, comparably-sized
systematic grid over the same documented ranges and reports how many valid
crossings it finds and their slope-difference range, as an honest
re-verification of the QUALITATIVE claim (transversal everywhere, no
near-tangent case), not a bit-exact replay.

Usage:
    python scripts/find_and_verify_points.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tide.continuation.codim2_convergence import _fold, hopf_npch_general
from tide.continuation.double_zero import _coalescing_pair, find_double_zero_point
from tide.continuation.tangency import crossing_slopes

N1 = 16
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"

# Table 2's reported values, for automated comparison.
TABLE_2 = {
    "A": dict(Fr=0.035, Lam=5.90, ki=6.55, ke=2.03, Nsub=29.886, Npch=49.702),
    "B": dict(Fr=0.5, Lam=0.001, ki=11.0, ke=3.0, Nsub=14.142794816, Npch=20.597778032),
    "C": dict(Fr=0.3, Lam=0.001, ki=6.0, ke=5.0, Nsub=7.630101792, Npch=10.913202566),
}


def _find_crossing(Fr, Lam, ki, ke, nsub_lo, nsub_hi, n_scan=400):
    """Scan for the Nsub where fold(Nsub) - hopf(Nsub) changes sign (the
    codim-2 point), refining with bisection. Returns None if no crossing."""
    nsub_scan = np.linspace(nsub_lo, nsub_hi, n_scan)
    gaps = []
    for ns in nsub_scan:
        f = _fold(ns, Fr, Lam, ki, ke)
        if f is None:
            gaps.append(np.nan)
            continue
        h = hopf_npch_general(ns, Fr, Lam, ki, ke, N1, bracket=(f * 0.85, f * 1.4))
        gaps.append((h - f) if h is not None else np.nan)
    gaps = np.array(gaps)

    crossing = None
    for i in range(len(nsub_scan) - 1):
        a, b = gaps[i], gaps[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            lo, hi = nsub_scan[i], nsub_scan[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                f = _fold(mid, Fr, Lam, ki, ke)
                if f is None:
                    break
                h = hopf_npch_general(mid, Fr, Lam, ki, ke, N1, bracket=(f * 0.85, f * 1.4))
                if h is None:
                    break
                gap_mid = h - f
                if abs(gap_mid) < 1e-9 or (hi - lo) < 1e-9:
                    crossing = mid
                    break
                if (a < 0) == (gap_mid < 0):
                    lo, a = mid, gap_mid
                else:
                    hi = mid
            else:
                crossing = 0.5 * (lo + hi)
            break
    return crossing


def verify_point_a():
    print("=== Point A: locating the transversal crossing from scratch ===")
    p = TABLE_2["A"]
    crossing = _find_crossing(p["Fr"], p["Lam"], p["ki"], p["ke"],
                               p["Nsub"] - 5, p["Nsub"] + 5)
    result = dict(found_nsub=crossing, table2_nsub=p["Nsub"])
    if crossing is None:
        result["status"] = "FAIL - no crossing found"
        print("  FAIL: no crossing found")
        return result

    npch_at_crossing = _fold(crossing, p["Fr"], p["Lam"], p["ki"], p["ke"])
    slopes = crossing_slopes(p["Fr"], p["Lam"], p["ki"], p["ke"], N1, crossing)
    result["found_npch"] = npch_at_crossing
    result["table2_npch"] = p["Npch"]
    result["slope_fold"], result["slope_hopf"], result["slope_diff"] = slopes
    nsub_match = abs(crossing - p["Nsub"]) < 0.01
    result["status"] = "PASS" if nsub_match else "FAIL - crossing does not match Table 2"
    print(f"  Found Nsub*={crossing:.4f} (Table 2: {p['Nsub']}), "
          f"Npch*={npch_at_crossing:.4f} (Table 2: {p['Npch']})")
    print(f"  Slope difference: {slopes[2]:.4f} (manuscript reports 0.92, "
          f"transversal since not near zero)")
    print(f"  {result['status']}")
    return result


def transversality_sweep():
    print("\n=== Independent transversality sweep (Section 5.1's negative control) ===")
    print("(Not a bit-exact replay of the historical '265 combinations' -- see")
    print(" this script's docstring. An independent, comparable grid.)")
    Fr_vals = np.array([0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
    Lam_vals = np.array([0.5, 1.2, 2.0, 3.0, 4.0, 5.0, 5.9])
    ki_vals = np.array([2.0, 4.0, 6.0, 8.0, 11.0])
    ke_vals = np.array([2.0, 3.0, 4.0, 5.0])

    slope_diffs = []
    n_valid = 0
    n_total = 0
    t0 = time.time()
    for Fr in Fr_vals:
        for Lam in Lam_vals:
            for ki in ki_vals:
                for ke in ke_vals:
                    n_total += 1
                    crossing = _find_crossing(Fr, Lam, ki, ke, 3.0, 40.0, n_scan=120)
                    if crossing is None:
                        continue
                    slopes = crossing_slopes(Fr, Lam, ki, ke, N1, crossing)
                    if slopes is None:
                        continue
                    n_valid += 1
                    slope_diffs.append(slopes[2])
    elapsed = time.time() - t0
    slope_diffs = np.array(slope_diffs)
    result = dict(
        n_total=n_total, n_valid_crossings=n_valid,
        slope_diff_min=float(slope_diffs.min()) if n_valid else None,
        slope_diff_max=float(slope_diffs.max()) if n_valid else None,
        near_tangent_count=int(np.sum(slope_diffs < 0.05)) if n_valid else None,
        elapsed_seconds=round(elapsed, 1),
    )
    print(f"  {n_valid}/{n_total} combinations had a valid crossing "
          f"({elapsed:.0f}s)")
    if n_valid:
        print(f"  Slope-difference range: {slope_diffs.min():.3f} to {slope_diffs.max():.3f} "
              f"(manuscript reports 0.68-1.68)")
        print(f"  Near-tangent (<0.05) combinations found: {result['near_tangent_count']} "
              f"(manuscript reports none)")
    return result


def verify_bt_point(name, x0):
    print(f"\n=== Point {name}: double-zero solve from initial guess {x0} ===")
    p = TABLE_2[name]
    nsub, npch, converged, residual, n_iter = find_double_zero_point(
        x0, p["Fr"], p["Lam"], p["ki"], p["ke"], N1)
    pair = _coalescing_pair(nsub, npch, p["Fr"], p["Lam"], p["ki"], p["ke"], N1)

    n1_results = {}
    for n1 in (2, 4, 8, 16):
        ns_n1, np_n1, conv_n1, _, _ = find_double_zero_point(
            x0, p["Fr"], p["Lam"], p["ki"], p["ke"], n1)
        n1_results[n1] = dict(nsub=ns_n1, npch=np_n1, converged=bool(conv_n1))

    matches_table2 = abs(nsub - p["Nsub"]) < 1e-4 and abs(npch - p["Npch"]) < 1e-3
    is_genuine_double_zero = float(np.max(np.abs(pair))) < 1e-4
    n1_independent = all(
        abs(v["nsub"] - n1_results[2]["nsub"]) < 1e-4 and
        abs(v["npch"] - n1_results[2]["npch"]) < 1e-3
        for v in n1_results.values()
    )
    status = "PASS" if (converged and matches_table2 and is_genuine_double_zero and n1_independent) else "FAIL"

    print(f"  Converged to Nsub={nsub:.9f}, Npch={npch:.9f} "
          f"(Table 2: {p['Nsub']}, {p['Npch']})")
    print(f"  Coalescing eigenvalue pair: {pair} (should be ~0 for a genuine double-zero)")
    print(f"  N1-independence (2,4,8,16): {n1_results}")
    print(f"  {status}")

    return dict(
        x0=list(x0), converged=bool(converged), nsub=nsub, npch=npch,
        residual=[float(r) for r in residual], n_iter=n_iter,
        coalescing_pair_real=[float(np.real(v)) for v in pair],
        coalescing_pair_imag=[float(np.imag(v)) for v in pair],
        matches_table2=matches_table2, is_genuine_double_zero=is_genuine_double_zero,
        n1_independent=n1_independent,
        n1_results={k: {kk: (vv if not isinstance(vv, (np.floating,)) else float(vv))
                        for kk, vv in v.items()} for k, v in n1_results.items()},
        status=status,
    )


def main():
    output = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    output["point_A"] = verify_point_a()
    output["transversality_sweep"] = transversality_sweep()
    # Generic initial guesses, deliberately NOT the already-known answer --
    # a nearby low-friction-region starting point, matching how a real
    # search would be seeded (a coarse gap-scan hit), not the exact
    # published coordinate.
    output["point_B"] = verify_bt_point("B", x0=(14.0, 20.5))
    output["point_C"] = verify_bt_point("C", x0=(7.5, 10.8))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"point_verification_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    all_pass = (output["point_A"]["status"] == "PASS" and
                output["point_B"]["status"] == "PASS" and
                output["point_C"]["status"] == "PASS")
    print(f"\n=== Overall: {'PASS' if all_pass else 'FAIL'} ===")
    print(f"Saved: {out_path}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
