import sys
"""Panel (c): Hydrodynamic Force Decomposition & Internal S-Curve Genesis
Modular, Fully-Configurable Python Script styled with modern Aptos typography.

Features:
- Aptos font family with STIX-Sans mathematical fontset.
- Complete `CONFIG` styling dictionary controlling all typography, colors, geometries, and texts.
- Detailed decomposition of internal pressure drop: Single-phase friction (∝ G²),
  Two-phase boiling loss (∝ 1/ρ_m), and Hydrostatic gravity head (∝ ρ_m g).
- Highlighted Ledinegg Unstable Zone (∂ΔP/∂G < 0) with turning point bifurcation marker.
- Governing force balance equation card.
- Single-column journal layout.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
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
        "figsize": (6.8, 5.6),
        "dpi": 300,
        "tight_layout": True,
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_1", "panel_c_force_decomposition.png")),
    },

    # 2. Typography & Matplotlib RC Params (Aptos + stixsans)
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12.0,
        "axes.titlesize": 13.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.8,
        "axes.linewidth": 1.3,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
        "mathtext_shrink_factor": 0.85,  # Subscript & superscript scaling (default 0.7)
    },

    # 3. Global Color Palette
    "palette": {
        "total_dp": "#1a3c6e",        # Deep Navy
        "friction_1p": "#007a78",     # Teal
        "expansion_2p": "#c88a10",    # Gold / Amber
        "gravity": "#475569",         # Slate Grey
        "instability_zone": "#fee2e2",# Light Crimson / Red
        "instability_border": "#ef4444",
        "card_bg": "#f8fafc",
        "card_border": "#94a3b8",
        "callout_bg": "#fffaf5",
        "callout_border": "#ea580c",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(c)}$ Hydrodynamic Force Decomposition & S-Curve",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Coolant\ Mass\ Flux,\ G\ (kg/(m^2 s))}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Internal\ Pressure\ Drop,\ \Delta P\ (bar)}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 2,
        },
        "xlim": (0.05, 3.25),
        "ylim": (-0.05, 3.35),
        "grid": True,
    },

    # 5. Pressure Drop Component Curves
    "curves": {
        "g_range": (0.08, 3.25, 600),
        "dp_1phi": {
            "label": r"$\mathbf{Single\text{-}Phase\ Friction\ (\Delta P_{\mathrm{single}} \propto G^2)}$",
            "color": "#007a78",
            "linestyle": "-.",
            "linewidth": 2.0,
            "zorder": 3,
        },
        "dp_2phi": {
            "label": r"$\mathbf{Two\text{-}Phase\ Boiling\ Loss\ (\Delta P_{\mathrm{two\text{-}phase}} \propto 1/\rho_m)}$",
            "color": "#c88a10",
            "linestyle": "--",
            "linewidth": 2.0,
            "zorder": 3,
        },
        "dp_grav": {
            "label": r"$\mathbf{Hydrostatic\ Gravity\ Head\ (\Delta P_{\mathrm{grav}} \propto \rho_m g)}$",
            "color": "#475569",
            "linestyle": ":",
            "linewidth": 1.8,
            "zorder": 2,
        },
        "dp_total": {
            "label": r"$\mathbf{Total\ Internal\ \Delta P_{\mathrm{int}}(G)\ (S\text{-}Curve)}$",
            "color": "#1a3c6e",
            "linestyle": "-",
            "linewidth": 3.0,
            "zorder": 5,
        },
    },

    # 6. Negative Slope Instability Zone
    "instability_zone": {
        "color": "#fee2e2",
        "alpha": 0.55,
        "border_color": "#ef4444",
        "border_linestyle": "--",
        "border_linewidth": 1.2,
        "label": r"$\mathbf{Negative\ Slope\ Zone\ (\partial \Delta P / \partial G < 0)}$",
    },

    # 7. Local Minimum Bifurcation Turning Point
    "turning_point": {
        "marker": {
            "style": "o",
            "size": 8.5,
            "color": "#dc2626",
            "edgecolor": "black",
            "edgewidth": 1.3,
            "zorder": 7,
        },
        "guide_line": {
            "color": "#dc2626",
            "linestyle": ":",
            "linewidth": 1.5,
            "zorder": 6,
        },
        "annotation": {
            "xytext": (1.0, 1.8),
            "text": (
                r"$\mathbf{Ledinegg\ Turning\ Point}$" + "\n"
                r"$\mathbf{\left(\frac{\partial \Delta P}{\partial G} = 0\right)\ \rightarrow\ Neutral\ Limit}$"
            ),
            "fontsize": 9.5,
            "fontweight": "bold",
            "color": "#991b1b",
            "bbox": {
                "boxstyle": "round,pad=0.42",
                "fc": "#fff5f5",
                "ec": "#f87171",
                "lw": 1.1,
                "alpha": 0.96,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#dc2626",
                "lw": 1.5,
            },
        },
    },

    # 8. Governing Model Equation Card
    "equation_card": {
        "coordinate_system": "data",
        "x": 0.15,
        "y": 2.95,
        "ha": "left",
        "va": "top",
        "text": (
            r"$\mathbf{Internal\ Force\ Balance:}$" + "\n"
            r"$\mathbf{\Delta P_{\mathrm{int}}(G) = \Delta P_{\mathrm{single}} + \Delta P_{\mathrm{two\text{-}phase}} + \Delta P_{\mathrm{grav}}}$"
        ),
        "fontsize": 9.8,
        "fontweight": "bold",
        "color": "#1a3c6e",
        "bbox": {
            "boxstyle": "round,pad=0.42",
            "fc": "#f8fafc",
            "ec": "#94a3b8",
            "lw": 1.1,
            "alpha": 0.96,
        },
    },

    # 9. Legend
    "legend": {
        "loc": "upper right",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 9.8,
    },
}


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_c(config=CONFIG):
    # Apply Typography & RC Params
    typo_cfg = dict(config["typography"])
    shrink_factor = typo_cfg.pop("mathtext_shrink_factor", 0.85)
    mmt.SHRINK_FACTOR = shrink_factor
    plt.rcParams.update(typo_cfg)

    # Create Canvas
    fig, ax = plt.subplots(figsize=config["figure"]["figsize"], dpi=config["figure"]["dpi"])

    # Set Title & Axes Labels
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
    ax.grid(ax_cfg["grid"])

    # Generate Mass Flux Domain
    g_min, g_max, g_pts = config["curves"]["g_range"]
    G = np.linspace(g_min, g_max, g_pts)

    # Physical Component Formulations
    # Single-phase liquid friction: ~ G^2
    dp_1phi = 0.22 * G**2 + 0.05 * G
    # Two-phase boiling loss: ~ G^2 / rho_m ~ 1 / G
    dp_2phi = 2.4 / (1.0 + 3.0 * G**1.8)
    # Hydrostatic elevation head: ~ rho_m * g
    dp_grav = 0.55 * (1.0 - 0.75 / (1.0 + 2.0 * G**1.5))
    # Total internal characteristic
    dp_total = dp_1phi + dp_2phi + dp_grav

    # Identify Negative Slope Region (∂ΔP/∂G < 0)
    d_dp = np.gradient(dp_total, G)
    idx_neg = np.where(d_dp < 0)[0]
    g_neg_start = G[idx_neg[0]]
    g_neg_end = G[idx_neg[-1]]

    # 1. Shaded Instability Zone
    iz_cfg = config["instability_zone"]
    ax.axvspan(ax_cfg["xlim"][0], g_neg_end,
               color=iz_cfg["color"],
               alpha=iz_cfg["alpha"],
               label=iz_cfg["label"],
               zorder=1)
    ax.axvline(g_neg_end, color=iz_cfg["border_color"],
               ls=iz_cfg["border_linestyle"], lw=iz_cfg["border_linewidth"], zorder=2)

    # 2. Plot Component Curves
    c_cfg = config["curves"]
    ax.plot(G, dp_1phi, color=c_cfg["dp_1phi"]["color"], lw=c_cfg["dp_1phi"]["linewidth"],
            ls=c_cfg["dp_1phi"]["linestyle"], label=c_cfg["dp_1phi"]["label"], zorder=c_cfg["dp_1phi"]["zorder"])
    ax.plot(G, dp_2phi, color=c_cfg["dp_2phi"]["color"], lw=c_cfg["dp_2phi"]["linewidth"],
            ls=c_cfg["dp_2phi"]["linestyle"], label=c_cfg["dp_2phi"]["label"], zorder=c_cfg["dp_2phi"]["zorder"])
    ax.plot(G, dp_grav, color=c_cfg["dp_grav"]["color"], lw=c_cfg["dp_grav"]["linewidth"],
            ls=c_cfg["dp_grav"]["linestyle"], label=c_cfg["dp_grav"]["label"], zorder=c_cfg["dp_grav"]["zorder"])
    ax.plot(G, dp_total, color=c_cfg["dp_total"]["color"], lw=c_cfg["dp_total"]["linewidth"],
            ls=c_cfg["dp_total"]["linestyle"], label=c_cfg["dp_total"]["label"], zorder=c_cfg["dp_total"]["zorder"])

    # 3. Turning Point (Local Minimum)
    tp_cfg = config["turning_point"]
    g_min_val = g_neg_end
    dp_min_val = dp_total[idx_neg[-1]]

    mk = tp_cfg["marker"]
    ax.plot(g_min_val, dp_min_val, marker=mk["style"], markersize=mk["size"],
            color=mk["color"], mec=mk["edgecolor"], mew=mk["edgewidth"], zorder=mk["zorder"])

    # Guide line down to x-axis
    gl = tp_cfg["guide_line"]
    ax.plot([g_min_val, g_min_val], [0, dp_min_val], color=gl["color"], ls=gl["linestyle"], lw=gl["linewidth"], zorder=gl["zorder"])

    # Turning Point Annotation
    tp_ann = tp_cfg["annotation"]
    ax.annotate(tp_ann["text"],
                xy=(g_min_val, dp_min_val),
                xytext=tp_ann["xytext"],
                fontsize=tp_ann["fontsize"],
                color=tp_ann["color"],
                bbox=dict(**tp_ann["bbox"]),
                arrowprops=dict(**tp_ann["arrowprops"]))

    # 4. Governing Force Balance Card
    eq_cfg = config["equation_card"]
    eq_trans = ax.transAxes if eq_cfg.get("coordinate_system", "data") == "axes" else ax.transData
    ax.text(eq_cfg["x"], eq_cfg["y"], eq_cfg["text"],
            transform=eq_trans,
            ha=eq_cfg["ha"], va=eq_cfg["va"],
            fontsize=eq_cfg["fontsize"], color=eq_cfg["color"],
            bbox=dict(**eq_cfg["bbox"]))

    # 5. Add Legend
    leg_cfg = config["legend"]
    ax.legend(loc=leg_cfg["loc"], frameon=leg_cfg["frameon"],
              framealpha=leg_cfg["framealpha"], facecolor=leg_cfg["facecolor"],
              edgecolor=leg_cfg["edgecolor"], fontsize=leg_cfg["fontsize"])

    # Save Output
    if config["figure"]["tight_layout"]:
        plt.tight_layout()

    out_file = config["figure"]["output_path"]
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    if os.path.exists(out_file):
        try:
            os.remove(out_file)
        except Exception:
            pass

    plt.savefig(out_file, dpi=config["figure"]["dpi"], bbox_inches="tight")
    print("Successfully generated and saved Panel (c) with Aptos to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_c()
