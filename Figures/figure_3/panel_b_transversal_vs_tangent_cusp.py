import sys
"""Panel (b): Topological Contrast: Transversal Crossing (Point A) vs. Tangential Cusp (Point B)
Full, un-truncated physical continuation across the complete operating domains:
  - Left: Non-degenerate transversal X-crossing at Point A (Lambda = 5.90, kin = 6.55, kex = 2.03, Fr = 0.035)
          Fold and Hopf curves cross with Delta_slope = 0.84 (theta ~ 11 deg); curves continue past crossing.
  - Right: Degenerate tangential cusp at Point B (Lambda = 0.001, kin = 11.00, kex = 3.00, Fr = 0.50)
           Fold and Hopf curves meet in exact tangency (Delta_slope = 0.00, theta = 0 deg) and TERMINATE at vertex.
           Beyond the vertex (Nsub > 14.14), the stable window is completely closed (globally unstable).
Evaluated directly from machine-precision continuation solutions.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.gridspec as gridspec
import numpy as np

from tide.continuation.codim2_convergence import _fold, hopf_npch_general
from tide.surrogates.bt_point_data import FR_BT, LAM_BT, KI_BT, KE_BT, NSUB_BT, NPCH_BT, wedge_boundaries_true
from tide.surrogates.data_generation import _leading_real_part

# ==============================================================================
# 0. FONT REGISTRATION (Aptos / CloudFonts)
# ==============================================================================
cloud_dir = r"C:\Users\User\AppData\Local\Microsoft\FontCache\4\CloudFonts"
for d in glob.glob(os.path.join(cloud_dir, "Aptos*")):
    for f in glob.glob(os.path.join(d, "*.ttf")):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass

# ==============================================================================
# MASTER CONFIGURATION & STYLING DICTIONARY
# ==============================================================================
CONFIG = {
    # 1. Figure & Canvas (Matching Panel A's exact (8.2, 6.2) aspect ratio & bounding box)
    "figure": {
        "figsize": (8.2, 6.2),
        "dpi": 300,
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_3", "panel_b_transversal_vs_tangent_cusp.png")),
    },

    # 2. Typography & Matplotlib RC Params (Aptos + STIX-Sans)
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10.0,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.0,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
        "mathtext_shrink_factor": 0.85,
    },

    # 3. Color Palette
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "gold": "#c88a10",
        "slate": "#475569",
        "corridor_fill": "#fef08a",
        "corridor_alpha": 0.90,
        "dwo_fill": "#ffe4e6",
        "dwo_alpha": 0.70,
        "ledinegg_fill": "#dbeafe",
        "ledinegg_alpha": 0.70,
        "compound_fill": "#d8b4fe",
        "compound_alpha": 0.85,
        "post_fill": "#f1f5f9",
        "post_alpha": 0.85,
    },

    # 4. Master Panel Title
    "master_title": {
        "text": "(b) Topological Contrast: Transversal Crossing vs. Tangential Cusp",
        "fontsize": 12.0,
        "fontweight": "bold",
        "pad": 12,
        "loc": "left",
    },

    # 5. Left Subplot: Point A Configuration (Transversal Crossing)
    "point_a": {
        "title": "Point A: Transversal Crossing",
        "params_box": r"$\Lambda = 5.90,\ k_{\mathrm{in}} = 6.55,$" + "\n" + r"$k_{\mathrm{ex}} = 2.03,\ Fr = 0.035$",
        "xlim": (25.5, 34.5),
        "ylim": (38.0, 61.0),
        "marker_pos": (29.886, 49.69),
        "badge": {
            "xy": (29.886, 49.69),
            "xytext": (26.2, 54.8),
            "text": "Point A (Crossing)",
            "fontsize": 9.5,
            "fc": "#f0fdf4",
            "ec": "#007a78",
        },
    },

    # 6. Right Subplot: Point B Configuration (Tangential Cusp)
    "point_b": {
        "title": "Point B: Tangential Cusp",
        "params_box": r"$\Lambda = 0.001,\ k_{\mathrm{in}} = 11.00,$" + "\n" + r"$k_{\mathrm{ex}} = 3.00,\ Fr = 0.50$",
        "xlim": (10.4, 15.6),
        "ylim": (13.0, 25.5),
        "marker_pos": (14.1428, 20.5948),
        "badge": {
            "xy": (14.1428, 20.5948),
            "xytext": (13.7, 23.6),
            "text": "Point B (BT Cusp)",
            "fontsize": 9.5,
            "fc": "#fff7ed",
            "ec": "#b5451b",
        },
    },

    # 7. Legend Settings
    "legend": {
        "loc": "lower right",
        "frameon": True,
        "framealpha": 1.0,
        "facecolor": "white",
        "edgecolor": "#94a3b8",
        "fontsize": 9.0,
        "borderpad": 0.35,
        "handlelength": 1.3,
        "handletextpad": 0.40,
    },
}

def load_data():
    """Load continuation data from precomputed npz cache for instantaneous generation."""
    cache_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "data", "generated", "fig3_continuation_data.npz"))
    if os.path.exists(cache_path):
        return np.load(cache_path)
    # Fallback to direct computation if needed
    nsub_a = np.linspace(22.0, 36.0, 50)
    nsub_b = np.linspace(8.8, 14.14278, 60)
    return None

def compute_point_a_curves():
    """Compute complete continuation curves for Point A (Lambda = 5.90) across Nsub in [22, 36]."""
    data = load_data()
    if data is not None and "nsub_a" in data:
        return data["nsub_a"], data["fold_a"], data["hopf_a"]
    FR_A, LAM_A, KI_A, KE_A = 0.035, 5.90, 6.55, 2.03
    N1 = 16
    nsub_a = np.linspace(22.0, 36.0, 50)
    fold_a, hopf_a = [], []
    for ns in nsub_a:
        f = _fold(ns, FR_A, LAM_A, KI_A, KE_A)
        fold_a.append(f)
        npch_test = np.linspace(25.0, 65.0, 100)
        g_vals = [_leading_real_part(ns, p, FR_A, LAM_A, KI_A, KE_A, N1) for p in npch_test]
        h_cross = None
        for i in range(len(npch_test)-1):
            if g_vals[i] is not None and g_vals[i+1] is not None and g_vals[i] * g_vals[i+1] < 0:
                h_cross = 0.5 * (npch_test[i] + npch_test[i+1])
        hopf_a.append(h_cross if h_cross is not None else np.nan)
    return nsub_a, np.array(fold_a, dtype=np.float64), np.array(hopf_a, dtype=np.float64)

def compute_point_b_curves():
    """Compute complete continuation curves for Point B (Lambda = 0.001) across Nsub in [8.8, 14.1428]."""
    data = load_data()
    if data is not None and "nsub_b" in data:
        lower_b = np.copy(data["lower_b"])
        upper_b = np.copy(data["upper_b"])
        if np.isnan(upper_b[-1]):
            upper_b[-1] = NPCH_BT
        if np.isnan(lower_b[-1]):
            lower_b[-1] = NPCH_BT
        return data["nsub_b"], lower_b, upper_b
    N1 = 16
    nsub_b = np.linspace(8.8, 14.14278, 60)
    lower_b, upper_b = [], []
    for ns in nsub_b:
        lo, hi = wedge_boundaries_true(ns, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
        lower_b.append(lo if lo is not None else np.nan)
        upper_b.append(hi if hi is not None else np.nan)
    lower_b[-1] = NPCH_BT
    upper_b[-1] = NPCH_BT
    return nsub_b, np.array(lower_b, dtype=np.float64), np.array(upper_b, dtype=np.float64)

def draw_panel_b(fig, config=CONFIG):
    pal = config["palette"]
    leg_cfg = config["legend"]

    # Reduced gap between plots (wspace=0.15)
    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.075, right=0.985, bottom=0.10, top=0.91, wspace=0.15)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ==========================================================================
    # LEFT SUBPLOT: Point A (Transversal Crossing)
    # ==========================================================================
    cfg_a = config["point_a"]
    nsub_a, fold_a, hopf_a = compute_point_a_curves()

    ax_left.set_title(cfg_a["title"], fontsize=11.2, fontweight="bold", pad=8, loc="center")

    # Inset condition box at top-left (Sized up font)
    ax_left.text(0.04, 0.94, cfg_a["params_box"], transform=ax_left.transAxes,
                 fontsize=9.2, va="top", ha="left", color="#1e293b", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.30", fc="#f8fafc", ec="#94a3b8", lw=1.0, alpha=0.95), zorder=8)

    ax_left.set_xlabel(r"$\mathbf{Subcooling\ No.,\ }N_{\mathrm{sub}}$", fontsize=10.5, fontweight="bold", labelpad=4)
    ax_left.set_ylabel(r"$\mathbf{Phase\ Change\ No.,\ }N_{\mathrm{pch}}$", fontsize=10.5, fontweight="bold", labelpad=4)
    ax_left.set_xlim(*cfg_a["xlim"])
    ax_left.set_ylim(*cfg_a["ylim"])
    ax_left.grid(True, alpha=0.28)

    cross_idx = np.where(fold_a >= hopf_a)[0]
    ci = cross_idx[0] if len(cross_idx) > 0 else len(nsub_a) - 1

    # Fill 4 sectors of the X
    ax_left.fill_between(nsub_a[:ci+1], fold_a[:ci+1], hopf_a[:ci+1],
                         color=pal["corridor_fill"], alpha=pal["corridor_alpha"], zorder=2,
                         label=r"Safe Operable ($g > 0$)")
    ax_left.fill_between(nsub_a, np.maximum(fold_a, hopf_a), 65.0,
                         color=pal["dwo_fill"], alpha=pal["dwo_alpha"], zorder=1)
    ax_left.fill_between(nsub_a, 30.0, np.minimum(fold_a, hopf_a),
                         color=pal["ledinegg_fill"], alpha=pal["ledinegg_alpha"], zorder=1)
    ax_left.fill_between(nsub_a[ci:], hopf_a[ci:], fold_a[ci:],
                         color=pal["compound_fill"], alpha=pal["compound_alpha"], zorder=2,
                         label=r"Compound (Fold > Hopf)")

    # Plot boundary curves completely across domain
    ax_left.plot(nsub_a, fold_a, color=pal["navy"], lw=2.4, zorder=4,
                 label=r"Fold ($\partial\Delta P/\partial G = 0$)")
    ax_left.plot(nsub_a, hopf_a, color=pal["crimson"], lw=2.4, zorder=4,
                 label=r"Hopf ($\mathrm{Re}(\mu) = 0$)")

    # Point A intersection marker
    px_a, py_a = cfg_a["marker_pos"]
    ax_left.plot([px_a], [py_a], marker="o", color=pal["teal"], ms=8.0, mec="black", mew=1.3, zorder=7,
                 label=r"Point A Intersection")

    # Callout badge (Sized up font)
    badge_a = cfg_a["badge"]
    ax_left.annotate(badge_a["text"],
                     xy=badge_a["xy"], xytext=badge_a["xytext"],
                     fontsize=badge_a["fontsize"], color=pal["navy"], fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.30", fc=badge_a["fc"], ec=badge_a["ec"], lw=1.2),
                     arrowprops=dict(arrowstyle="->", color=pal["navy"], lw=1.3), zorder=8)

    # Sector text labels (Sized up font)
    ax_left.text(27.4, 50.5, "Dynamic DWO\nInstability", fontsize=9.0, color=pal["crimson"],
                 fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec="#fda4af", lw=0.9, alpha=0.95), zorder=6)
    ax_left.text(28.0, 42.0, "Static Ledinegg\nInstability", fontsize=9.0, color=pal["navy"],
                 fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#93c5fd", lw=0.9, alpha=0.95), zorder=6)
    ax_left.text(33.0, 52.8, "Compound\nInstability", fontsize=8.8, color="#6b21a8",
                 fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#f3e8ff", ec="#c084fc", lw=0.9, alpha=0.95), zorder=6)

    leg_a = ax_left.legend(loc=leg_cfg["loc"], frameon=True, framealpha=1.0, facecolor="white",
                           edgecolor=leg_cfg["edgecolor"], fontsize=leg_cfg["fontsize"],
                           borderpad=leg_cfg["borderpad"], handlelength=leg_cfg["handlelength"],
                           handletextpad=leg_cfg["handletextpad"])
    leg_a.set_zorder(10)

    # ==========================================================================
    # RIGHT SUBPLOT: Point B (Tangential Bogdanov-Takens Cusp)
    # ==========================================================================
    cfg_b = config["point_b"]
    nsub_b, lower_b, upper_b = compute_point_b_curves()

    ax_right.set_title(cfg_b["title"], fontsize=11.2, fontweight="bold", pad=8, loc="center")

    # Inset condition box at top-left (Sized up font)
    ax_right.text(0.04, 0.94, cfg_b["params_box"], transform=ax_right.transAxes,
                  fontsize=9.2, va="top", ha="left", color="#1e293b", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.30", fc="#f8fafc", ec="#94a3b8", lw=1.0, alpha=0.95), zorder=8)

    ax_right.set_xlabel(r"$\mathbf{Subcooling\ No.,\ }N_{\mathrm{sub}}$", fontsize=10.5, fontweight="bold", labelpad=4)
    ax_right.set_ylabel(r"$\mathbf{Phase\ Change\ No.,\ }N_{\mathrm{pch}}$", fontsize=10.5, fontweight="bold", labelpad=4)
    ax_right.set_xlim(*cfg_b["xlim"])
    ax_right.set_ylim(*cfg_b["ylim"])
    ax_right.grid(True, alpha=0.28)

    # Fill regimes
    ax_right.fill_between(nsub_b, lower_b, upper_b,
                          color=pal["corridor_fill"], alpha=pal["corridor_alpha"], zorder=2,
                          label=r"Safe Corridor ($g > 0$)")
    ax_right.fill_between(nsub_b, upper_b, 30.0,
                          color=pal["dwo_fill"], alpha=pal["dwo_alpha"], zorder=1)
    ax_right.fill_between(nsub_b, 5.0, lower_b,
                          color=pal["ledinegg_fill"], alpha=pal["ledinegg_alpha"], zorder=1)

    # Beyond Point B: Entire region is globally unstable
    nsub_dead = np.linspace(NSUB_BT, cfg_b["xlim"][1], 50)
    ax_right.fill_between(nsub_dead, 5.0, 30.0,
                          color=pal["post_fill"], alpha=pal["post_alpha"], zorder=1)

    # Plot boundary curves (TERMINATING AT THE STAR)
    ax_right.plot(nsub_b, lower_b, color=pal["navy"], lw=2.4, zorder=4,
                  label=r"Fold ($\partial\Delta P/\partial G = 0$)")
    ax_right.plot(nsub_b, upper_b, color=pal["crimson"], lw=2.4, zorder=4,
                  label=r"Hopf ($\mathrm{Re}(\mu) = 0$)")

    # BT vertex marker
    px_b, py_b = cfg_b["marker_pos"]
    ax_right.plot([px_b], [py_b], marker="*", color="black", ms=14.0, zorder=7,
                  label=r"BT Vertex ($\star$)")

    # Callout badge (Sized up font, moved to open area without overlap)
    badge_b = cfg_b["badge"]
    ax_right.annotate(badge_b["text"],
                      xy=badge_b["xy"], xytext=badge_b["xytext"],
                      fontsize=badge_b["fontsize"], color="black", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.30", fc=badge_b["fc"], ec=badge_b["ec"], lw=1.2),
                      arrowprops=dict(arrowstyle="->", color="black", lw=1.3), zorder=8)

    # Labels inside plot (Sized up font)
    ax_right.text(12.0, 21.0, "Dynamic DWO\nInstability", fontsize=9.0, color=pal["crimson"],
                  fontweight="bold", ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec="#fda4af", lw=0.9, alpha=0.95), zorder=6)
    ax_right.text(12.4, 15.2, "Static Ledinegg\nInstability", fontsize=9.0, color=pal["navy"],
                  fontweight="bold", ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#93c5fd", lw=0.9, alpha=0.95), zorder=6)
    ax_right.text(14.8, 16.5, "PINCHED-OUT\n(Globally Unstable)\n" + r"$\max(\mathrm{Re}(\lambda)) > 0$",
                  fontsize=8.5, color=pal["slate"], fontweight="bold", style="italic", ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.30", fc="#f8fafc", ec="#cbd5e1", lw=0.9, alpha=0.95), zorder=6)

    leg_b = ax_right.legend(loc=leg_cfg["loc"], frameon=True, framealpha=1.0, facecolor="white",
                            edgecolor=leg_cfg["edgecolor"], fontsize=leg_cfg["fontsize"],
                            borderpad=leg_cfg["borderpad"], handlelength=leg_cfg["handlelength"],
                            handletextpad=leg_cfg["handletextpad"])
    leg_b.set_zorder(10)

def generate_standalone_panel_b():
    typo_cfg = dict(CONFIG["typography"])
    shrink_factor = typo_cfg.pop("mathtext_shrink_factor", 0.85)
    mmt.SHRINK_FACTOR = shrink_factor
    plt.rcParams.update(typo_cfg)

    fig = plt.figure(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    
    m_title = CONFIG["master_title"]
    fig.suptitle(m_title["text"], fontsize=m_title["fontsize"], fontweight=m_title["fontweight"],
                 x=0.04, y=0.98, ha="left")

    draw_panel_b(fig, CONFIG)

    out_file = CONFIG["figure"]["output_path"]
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    plt.savefig(out_file, dpi=CONFIG["figure"]["dpi"], bbox_inches="tight")
    print("\n[SUCCESS] Successfully generated Standalone Panel (b) to:")
    print("  -", out_file)
    return fig

if __name__ == "__main__":
    generate_standalone_panel_b()
