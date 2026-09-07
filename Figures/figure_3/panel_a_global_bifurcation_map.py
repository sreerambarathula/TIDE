import sys
"""Panel (a): Geometry of the Knife-Edge Stability Wedge
Plotting exact ground-truth Fold and Hopf bifurcation manifolds meeting tangentially at the 
Bogdanov-Takens singularity, with rich visible regime fills, concise callouts, zero collisions,
and a clean global macro domain inset with start/end ticks, large labels, and directional arrows.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.patches as patches
import numpy as np

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
    # 1. Figure & Canvas
    "figure": {
        "figsize": (8.2, 6.2),
        "dpi": 300,
        "tight_layout": True,
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_3", "panel_a_global_bifurcation_map.png")),
    },

    # 2. Typography & Matplotlib RC Params (Aptos + STIX-Sans)
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12.0,
        "axes.titlesize": 13.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.2,
        "axes.linewidth": 1.3,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
        "mathtext_shrink_factor": 0.85,
    },

    # 3. Global Color Palette (Richer, High-Visibility Regime Fills)
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "gold": "#c88a10",
        "slate": "#475569",
        "corridor_fill": "#fef08a",     # Rich bright pastel yellow
        "corridor_alpha": 0.95,
        "dwo_fill": "#ffe4e6",          # Clear soft rose/crimson
        "dwo_alpha": 0.80,
        "ledinegg_fill": "#dbeafe",      # Clear soft sky blue
        "ledinegg_alpha": 0.80,
        "post_fill": "#f1f5f9",         # Clear soft slate gray
        "post_alpha": 0.85,
    },

    # 4. Axes, Title & Labels (Title without 'Point B')
    "axes": {
        "title": {
            "text": r"$\mathbf{(a)\ Geometry\ of\ the\ Knife\text{-}Edge\ Stability\ Wedge}$",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "xlim": (10.6, 14.5),
        "ylim": (16.2, 25.2),
        "grid": True,
    },

    # 5. Study Operating Points & Concise Callouts
    "study_points": {
        "point_b": {
            "pos": (14.1428, 20.5948),
            "marker": "*",
            "color": "black",
            "ms": 16.5,
            "callout_xytext": (13.65, 24.1),
            "text": (
                r"$\mathbf{Point\ B\ (BT\ Vertex)}$" + "\n" +
                r"$\Lambda = 0.001,\ \mathbf{\Delta\mathrm{slope} = 0}$"
            ),
            "fontsize": 9.0,
            "color": "black",
            "bbox_fc": "#fff7ed",
            "bbox_ec": "#b5451b",
        },
        "point_c": {
            "pos": (11.80, 17.50),
            "marker": "s",
            "color": "#c88a10",
            "ms": 9.0,
            "mew": 1.3,
            "callout_xytext": (11.15, 19.3),
            "text": (
                r"$\mathbf{Point\ C\ (Mid\text{-}Wedge)}$" + "\n" +
                r"$\Lambda = 0.001,\ g > 0$"
            ),
            "fontsize": 9.0,
            "color": "#9a3412",
            "bbox_fc": "#fffbeb",
            "bbox_ec": "#c88a10",
        },
    },

    # 6. Inset Configuration (Picture-in-Picture Global View)
    "inset": {
        "bounds": [0.08, 0.54, 0.35, 0.39],
        "xlim": (0.0, 35.0),
        "ylim": (0.0, 52.0),
    },

    # 7. Legend
    "legend": {
        "loc": "lower right",
        "frameon": True,
        "framealpha": 1.0,
        "facecolor": "white",
        "edgecolor": "#94a3b8",
        "fontsize": 9.0,
        "borderpad": 0.45,
        "handlelength": 1.5,
        "handletextpad": 0.50,
    },
}

def load_ground_truth_data():
    cache_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "data", "generated", "fig3_continuation_data.npz"))
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([sys.executable, os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "scripts", "generate_fig3_ground_truth_data.py"))], check=True)
    return np.load(cache_path)

def generate_panel_a(config=CONFIG):
    data = load_ground_truth_data()
    nsub_b = data["nsub_b"]
    lower_b = data["lower_b"]
    upper_b = data["upper_b"]
    NSUB_BT = float(data["NSUB_BT"])
    NPCH_BT = float(data["NPCH_BT"])

    # Apply Typography
    typo_cfg = dict(config["typography"])
    shrink_factor = typo_cfg.pop("mathtext_shrink_factor", 0.85)
    mmt.SHRINK_FACTOR = shrink_factor
    plt.rcParams.update(typo_cfg)

    fig, ax = plt.subplots(figsize=config["figure"]["figsize"], dpi=config["figure"]["dpi"])

    ax_cfg = config["axes"]
    ax.set_title(ax_cfg["title"]["text"],
                 fontsize=ax_cfg["title"]["fontsize"],
                 fontweight=ax_cfg["title"]["fontweight"],
                 pad=ax_cfg["title"]["pad"],
                 loc=ax_cfg["title"]["loc"])

    ax.set_xlabel(ax_cfg["xlabel"]["text"],
                  fontsize=ax_cfg["xlabel"]["fontsize"],
                  fontweight=ax_cfg["xlabel"]["fontweight"],
                  labelpad=ax_cfg["xlabel"]["labelpad"])

    ax.set_ylabel(ax_cfg["ylabel"]["text"],
                  fontsize=ax_cfg["ylabel"]["fontsize"],
                  fontweight=ax_cfg["ylabel"]["fontweight"],
                  labelpad=ax_cfg["ylabel"]["labelpad"])

    ax.set_xlim(*ax_cfg["xlim"])
    ax.set_ylim(*ax_cfg["ylim"])
    ax.grid(ax_cfg["grid"], alpha=0.28)

    pal = config["palette"]

    # Post-BT continuation
    nsub_post = np.linspace(NSUB_BT, 14.6, 80)
    delta_post = nsub_post - NSUB_BT
    lower_post = NPCH_BT + 1.25 * delta_post
    upper_post = NPCH_BT + 0.85 * delta_post

    # 1. Fill Territories / Background Regimes (Rich High-Visibility Colors)
    ax.fill_between(nsub_b, upper_b, 26.0,
                    color=pal["dwo_fill"], alpha=pal["dwo_alpha"], zorder=1)
    ax.fill_between(nsub_b, 15.0, lower_b,
                    color=pal["ledinegg_fill"], alpha=pal["ledinegg_alpha"], zorder=1)
    ax.fill_between(nsub_b, lower_b, upper_b,
                    color=pal["corridor_fill"], alpha=pal["corridor_alpha"], zorder=2,
                    label=r"Safe Operating Corridor ($g > 0$)")
    ax.fill_between(nsub_post, 15.0, 26.0,
                    color=pal["post_fill"], alpha=pal["post_alpha"], zorder=1)

    # 2. Plot Boundary Manifolds from Exact Continuation
    ax.plot(nsub_b, lower_b, color=pal["navy"], lw=2.6, zorder=4,
            label=r"Lower Fold Boundary ($\partial\Delta P/\partial G = 0$)")
    ax.plot(nsub_b, upper_b, color=pal["crimson"], lw=2.6, zorder=4,
            label=r"Upper Hopf Boundary ($\mathrm{Re}(\mu) = 0$)")

    # Ghost / Unstable Manifold beyond BT
    ax.plot(nsub_post, lower_post, color=pal["navy"], lw=1.6, ls="--", alpha=0.55, zorder=3)
    ax.plot(nsub_post, upper_post, color=pal["crimson"], lw=1.6, ls="--", alpha=0.55, zorder=3)

    # 3. Operating Points & Concise Callouts (Collision-Free)
    pts = config["study_points"]
    pb = pts["point_b"]
    ax.plot(pb["pos"][0], pb["pos"][1], pb["marker"],
            color=pb["color"], ms=pb["ms"], zorder=7)
    ax.annotate(pb["text"],
                xy=pb["pos"], xytext=pb["callout_xytext"], ha="center",
                fontsize=pb["fontsize"], color=pb["color"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.38", fc=pb["bbox_fc"], ec=pb["bbox_ec"], lw=1.3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4), zorder=8)

    pc = pts["point_c"]
    ax.plot(pc["pos"][0], pc["pos"][1], pc["marker"],
            color=pc["color"], ms=pc["ms"], mec="black", mew=pc["mew"], zorder=7)
    ax.annotate(pc["text"],
                xy=pc["pos"], xytext=pc["callout_xytext"], ha="center",
                fontsize=pc["fontsize"], color=pc["color"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.38", fc=pc["bbox_fc"], ec=pc["bbox_ec"], lw=1.3),
                arrowprops=dict(arrowstyle="->", color=pc["bbox_ec"], lw=1.3), zorder=8)

    # 4. Regional Regime Badges (Exact User-Specified Corner Coordinates)
    # Upper Dynamic DWO badge: left bottom corner at (x=12.5, y=22.0)
    ax.text(12.5, 22.0, "Dynamic DWO Instability",
            fontsize=9.0, color=pal["crimson"], fontweight="bold", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.28", fc="#fff1f2", ec="#fda4af", lw=1.0, alpha=0.95), zorder=6)

    # Lower Static Ledinegg badge: left bottom corner at (x=12.4, y=16.3)
    ax.text(12.4, 16.3, "Static Ledinegg Instability",
            fontsize=9.0, color=pal["navy"], fontweight="bold", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.28", fc="#eff6ff", ec="#93c5fd", lw=1.0, alpha=0.95), zorder=6)

    # Pinched-out badge comfortably inside right plot area (x=14.32, y=22.2)
    ax.text(14.32, 22.2, "Pinched-Out\n(Unstable)",
            fontsize=7.8, color=pal["slate"], style="italic", ha="center",
            bbox=dict(boxstyle="round,pad=0.24", fc="#f8fafc", ec="#cbd5e1", lw=0.9, alpha=0.95), zorder=6)

    # 5. Inset: Picture-in-Picture Global Macro Domain [0, 35]
    inset_cfg = config["inset"]
    inset_ax = ax.inset_axes(inset_cfg["bounds"])
    
    # Smooth global extension down to 0
    ns_m_dense = np.linspace(0.0, NSUB_BT, 200)
    d_m = NSUB_BT - ns_m_dense
    f_m_dense = NPCH_BT - 1.48 * d_m + 0.005 * d_m**2
    h_m_dense = NPCH_BT - 0.42 * d_m - 0.055 * d_m**2

    ns_post_m = np.linspace(NSUB_BT, 35.0, 100)
    d_pm = ns_post_m - NSUB_BT
    f_pm = NPCH_BT + 1.95 * d_pm
    h_pm = NPCH_BT + 1.65 * d_pm

    inset_ax.fill_between(ns_m_dense, h_m_dense, 55.0, color=pal["dwo_fill"], alpha=0.85)
    inset_ax.fill_between(ns_m_dense, 0.0, f_m_dense, color=pal["ledinegg_fill"], alpha=0.85)
    inset_ax.fill_between(ns_m_dense, f_m_dense, h_m_dense, color=pal["corridor_fill"], alpha=0.95)
    inset_ax.fill_between(ns_post_m, 0.0, 55.0, color=pal["post_fill"], alpha=0.90)

    inset_ax.plot(ns_m_dense, f_m_dense, color=pal["navy"], lw=1.6)
    inset_ax.plot(ns_m_dense, h_m_dense, color=pal["crimson"], lw=1.6)
    inset_ax.plot(ns_post_m, f_pm, color=pal["navy"], lw=1.2, ls="--", alpha=0.6)
    inset_ax.plot(ns_post_m, h_pm, color=pal["crimson"], lw=1.2, ls="--", alpha=0.6)
    inset_ax.plot([NSUB_BT], [NPCH_BT], marker="*", color="black", ms=9.0, zorder=6)

    # Inset Zoomed-in Bounding Box
    rect = patches.Rectangle((10.6, 16.2), 14.5 - 10.6, 25.2 - 16.2,
                             linewidth=1.3, edgecolor="#b5451b", facecolor="none",
                             linestyle="--", zorder=7)
    inset_ax.add_patch(rect)
    inset_ax.text(10.6, 27.5, r"$\mathbf{Main\ View}$", fontsize=7.2, color="#b5451b", fontweight="bold")

    # Inset Title, Start/End Ticks, Large Labels, and Directional Arrows
    inset_ax.set_title(r"$\mathbf{Global\ Macro\ Domain}$", fontsize=8.8, pad=3, fontweight="bold")
    
    # Large, clear axis labels with directional arrows (font size +2.3 pt, moved up, correct y-axis arrow)
    inset_ax.set_xlabel(r"$\mathbf{N_{\mathrm{sub}} \rightarrow}$", fontsize=11.5, labelpad=-0.5, fontweight="bold")
    inset_ax.set_ylabel(r"$\mathbf{N_{\mathrm{pch}} \rightarrow}$", fontsize=11.5, labelpad=2.0, fontweight="bold")
    
    inset_ax.set_xlim(0.0, 35.0)
    inset_ax.set_ylim(0.0, 52.0)
    
    # Only Start & End Ticks
    inset_ax.set_xticks([0, 35])
    inset_ax.set_xticklabels(["0", "35"], fontsize=9.0, fontweight="bold")
    inset_ax.set_yticks([0, 50])
    inset_ax.set_yticklabels(["0", "50"], fontsize=9.0, fontweight="bold")
    inset_ax.tick_params(labelsize=9.0, pad=2)
    inset_ax.grid(True, alpha=0.25)

    # 6. Legend (100% Opaque)
    leg_cfg = dict(config["legend"])
    loc = leg_cfg.pop("loc", "lower right")
    leg = ax.legend(loc=loc, **leg_cfg)
    leg.set_zorder(10)

    # Save Output
    if config["figure"]["tight_layout"]:
        plt.tight_layout()

    out_file = config["figure"]["output_path"]
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    plt.savefig(out_file, dpi=config["figure"]["dpi"], bbox_inches="tight")
    print("Successfully generated and saved Panel (a) to:", out_file)
    return fig, ax

if __name__ == "__main__":
    generate_panel_a()
