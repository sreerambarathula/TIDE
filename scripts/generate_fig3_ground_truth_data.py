"""Compute and cache machine-precision ground-truth continuation datasets for Figure 3.
Reads directly from src/tide/ physics and continuation solvers.
"""
import os
import time
import numpy as np

from tide.continuation.codim2_convergence import _fold, hopf_npch_general
from tide.surrogates.bt_point_data import (
    FR_BT, LAM_BT, KI_BT, KE_BT, NSUB_BT, NPCH_BT,
    wedge_boundaries, wedge_boundaries_true, _g, _bisect_g
)

def compute_all_fig3_data():
    t0 = time.time()
    print("=" * 70)
    print("COMPUTING EXACT GROUND-TRUTH CONTINUATION FOR FIGURE 3")
    print("=" * 70)
    
    N1 = 16
    NSUB_BT_CORRECTED = 14.142794816
    
    # --------------------------------------------------------------------------
    # 1. Point B High-Resolution Stable Wedge (Panel a & Panel b)
    # --------------------------------------------------------------------------
    print("\n[1/5] Computing Point B Knife-Edge Wedge (80 points)...")
    nsub_b = np.linspace(10.6, NSUB_BT, 80)
    lower_b, upper_b = [], []
    for ns in nsub_b:
        lo, hi = wedge_boundaries(ns, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
        lower_b.append(lo)
        upper_b.append(hi)
    
    lower_b = np.array(lower_b, dtype=np.float64)
    upper_b = np.array(upper_b, dtype=np.float64)
    
    # --------------------------------------------------------------------------
    # 2. Point A Transversal Crossing (Panel b Inset)
    # --------------------------------------------------------------------------
    print("[2/5] Computing Point A Transversal Crossing (40 points)...")
    FR_A, LAM_A, KI_A, KE_A = 0.035, 5.90, 6.55, 2.03
    NSUB_A = 29.886
    nsub_a = np.linspace(NSUB_A - 4.0, NSUB_A + 4.0, 40)
    fold_a, hopf_a = [], []
    for ns in nsub_a:
        f = _fold(ns, FR_A, LAM_A, KI_A, KE_A)
        fold_a.append(f)
        if f is not None:
            h = hopf_npch_general(ns, FR_A, LAM_A, KI_A, KE_A, N1, bracket=(f * 0.85, f * 1.4))
            hopf_a.append(h)
        else:
            hopf_a.append(np.nan)
    fold_a = np.array(fold_a, dtype=np.float64)
    hopf_a = np.array(hopf_a, dtype=np.float64)
    
    # --------------------------------------------------------------------------
    # 3. Global Macro Domain for Inset (Nsub in [1.0, 35.0])
    # --------------------------------------------------------------------------
    print("[3/5] Computing Global Macro Domain (35 points)...")
    nsub_macro = np.linspace(1.0, 35.0, 35)
    fold_macro = []
    for ns in nsub_macro:
        f = _fold(ns, FR_BT, LAM_BT, KI_BT, KE_BT)
        fold_macro.append(f if f is not None else np.nan)
    fold_macro = np.array(fold_macro, dtype=np.float64)
    
    # --------------------------------------------------------------------------
    # 4. Multi-Decade Power Law Scaling (Panel c)
    # --------------------------------------------------------------------------
    print("[4/5] Computing Multi-Decade Window Scaling (40 points across 4 decades)...")
    deltas = np.geomspace(1e-4, 2.0, 40)
    widths, valid_deltas = [], []
    for d in deltas:
        ns = NSUB_BT_CORRECTED - d
        center = NPCH_BT - 0.77 * d
        half_span = max(1.4 * d, 0.04)
        grid = np.linspace(center - half_span, center + half_span, 160)
        g_vals = [_g(ns, p, FR_BT, LAM_BT, KI_BT, KE_BT, N1) for p in grid]
        crossings = []
        for i in range(len(grid) - 1):
            if g_vals[i] is not None and g_vals[i+1] is not None and g_vals[i] * g_vals[i+1] < 0:
                crossings.append((grid[i], grid[i+1]))
        if len(crossings) >= 2:
            l = _bisect_g(ns, crossings[0][0], crossings[0][1], FR_BT, LAM_BT, KI_BT, KE_BT, N1, tol=1e-10)
            u = _bisect_g(ns, crossings[1][0], crossings[1][1], FR_BT, LAM_BT, KI_BT, KE_BT, N1, tol=1e-10)
            if l is not None and u is not None and u > l:
                widths.append(u - l)
                valid_deltas.append(d)
    
    valid_deltas = np.array(valid_deltas, dtype=np.float64)
    widths = np.array(widths, dtype=np.float64)
    
    logd, logw = np.log(valid_deltas), np.log(widths)
    slope, intercept = np.polyfit(logd, logw, 1)
    resid = logw - (slope * logd + intercept)
    r2 = 1.0 - np.sum(resid**2) / np.sum((logw - logw.mean())**2)
    print(f"      Measured Power-Law: width = {np.exp(intercept):.4f} * delta^{slope:.4f} (R^2 = {r2:.6f})")

    # --------------------------------------------------------------------------
    # 5. Continuous Stability Margin Field Profiles g(x) (Panel d)
    # --------------------------------------------------------------------------
    print("[5/5] Computing Exact Transverse Stability Margin Field Cross-Sections...")
    target_deltas = [1.00, 0.50, 0.20, 0.05]
    y_norm = np.linspace(-1.3, 1.3, 80)
    margin_profiles = {}
    
    for td in target_deltas:
        ns = NSUB_BT_CORRECTED - td
        center = NPCH_BT - 0.77 * td
        half_span = max(1.4 * td, 0.04)
        grid = np.linspace(center - half_span, center + half_span, 160)
        g_vals = [_g(ns, p, FR_BT, LAM_BT, KI_BT, KE_BT, N1) for p in grid]
        crossings = []
        for i in range(len(grid) - 1):
            if g_vals[i] is not None and g_vals[i+1] is not None and g_vals[i] * g_vals[i+1] < 0:
                crossings.append((grid[i], grid[i+1]))
        if len(crossings) >= 2:
            lo = _bisect_g(ns, crossings[0][0], crossings[0][1], FR_BT, LAM_BT, KI_BT, KE_BT, N1, tol=1e-10)
            hi = _bisect_g(ns, crossings[1][0], crossings[1][1], FR_BT, LAM_BT, KI_BT, KE_BT, N1, tol=1e-10)
        else:
            lo, hi = wedge_boundaries_true(ns, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
        w = hi - lo
        mid = 0.5 * (lo + hi)
        
        # Sample margin field g along the transverse cut
        g_slice = []
        for y in y_norm:
            p = mid + y * (w / 2.0)
            val = _g(ns, p, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
            # True stability margin: g_margin = -val (positive inside stable basin)
            g_slice.append(-val if val is not None else np.nan)
        margin_profiles[f"d_{td:.2f}"] = np.array(g_slice, dtype=np.float64)

    # --------------------------------------------------------------------------
    # Save to Cache NPZ
    # --------------------------------------------------------------------------
    out_dir = os.path.normpath(r"D:\AGravity\Tide_Tutor\data\generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig3_continuation_data.npz")
    
    save_dict = {
        "nsub_b": nsub_b,
        "lower_b": lower_b,
        "upper_b": upper_b,
        "NSUB_BT": NSUB_BT_CORRECTED,
        "NPCH_BT": NPCH_BT,
        "nsub_a": nsub_a,
        "fold_a": fold_a,
        "hopf_a": hopf_a,
        "NSUB_A": NSUB_A,
        "nsub_macro": nsub_macro,
        "fold_macro": fold_macro,
        "deltas": valid_deltas,
        "widths": widths,
        "scaling_slope": slope,
        "scaling_intercept": intercept,
        "scaling_r2": r2,
        "y_norm": y_norm,
    }
    for k, v in margin_profiles.items():
        save_dict[k] = v
        
    np.savez_compressed(out_path, **save_dict)
    print(f"\n[SUCCESS] Exact ground-truth saved in {time.time()-t0:.2f}s to:\n  {out_path}")

if __name__ == "__main__":
    compute_all_fig3_data()
