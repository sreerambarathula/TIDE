"""Panel (d): Static Ledinegg Instability & Dynamic Excursion Runaway Jump
Modular, Fully-Configurable Python Script styled with modern Aptos typography.

Features:
- Aptos font family with STIX-Sans mathematical fontset.
- Complete `CONFIG` styling dictionary controlling all typography, colors, geometries, and texts.
- Physical S-curve with 3 intersection operating points:
    * Point A (Stable Low-Flow / Vapor-Rich Branch)
    * Point B (Unstable Saddle / Negative Slope Branch)
    * Point C (Stable High-Flow / Liquid-Rich Branch)
- Shaded Ledinegg Negative Slope Instability Zone (∂ΔP/∂G < 0).
- Dynamic Flow Excursion / Runaway Jump trajectory (B -> A CHF Burnout collapse).
- Governing Ledinegg Stability Criterion equation card.
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
        "output_path": "d:/AGravity/Tide_Tutor/Figures/figure_1/panel_d_ledinegg_excursion.png",
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
        "channel_curve": "#1a3c6e",    # Deep Navy
        "pump_supply": "#dc2626",      # Bright Crimson
        "stable_pt": "#007a78",        # Teal / Emerald
        "unstable_pt": "#dc2626",      # Crimson
        "excursion_arrow": "#dc2626",  # Red
        "instability_zone": "#fee2e2", # Soft Red Shading
        "instability_border": "#f87171",
        "card_bg": "#f8fafc",
        "card_border": "#94a3b8",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(d)}$ Static Ledinegg Instability & Flow Excursion",
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
        "xlim": (0.1, 3.15),
        "ylim": (0.35, 2.65),
        "grid": True,
    },

    # 5. Physics Curves
    "curves": {
        "g_range": (0.18, 3.08, 600),
        "pump_head": 1.45,
        "s_curve": {
            "label": r"$\mathbf{Channel\ Characteristic\ \Delta P_{\mathrm{int}}(G)}$",
            "color": "#1a3c6e",
            "linestyle": "-",
            "linewidth": 2.8,
            "zorder": 4,
        },
        "pump_line": {
            "label": r"$\mathbf{External\ Supply\ \Delta P_{\mathrm{ext}}\ (Plenum\ \Delta P_0)}$",
            "color": "#dc2626",
            "linestyle": "--",
            "linewidth": 2.0,
            "zorder": 3,
        },
    },

    # 6. Instability Zone Shading
    "instability_zone": {
        "color": "#fee2e2",
        "alpha": 0.50,
        "border_color": "#f87171",
        "border_linestyle": ":",
        "border_linewidth": 1.2,
        "label": r"$\mathbf{Negative\ Slope\ Zone\ (\partial \Delta P_{\mathrm{int}} / \partial G < 0)}$",
    },

    # 7. Operating Points Annotations
    "operating_points": {
        "marker_size": 8.5,
        "edge_color": "black",
        "edge_width": 1.3,
        "zorder": 6,
        "pt_A": {
            "name": "A",
            "color": "#007a78",
            "xytext": (0.53, 0.65),
            "text": (
                r"$\mathbf{Point\ A\ (Stable)}$" + "\n"
                r"$\mathbf{Vapor\text{-}Rich\ (\frac{\partial \Delta P}{\partial G} > 0)}$"
            ),
            "fontsize": 9.2,
            "fontweight": "bold",
            "color_text": "#005a58",
            "bbox": {
                "boxstyle": "round,pad=0.38",
                "fc": "#f0fdfa",
                "ec": "#5eead4",
                "lw": 1.1,
                "alpha": 0.96,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#007a78",
                "lw": 1.3,
            },
        },
        "pt_B": {
            "name": "B",
            "color": "#dc2626",
            "xytext": (1.60, 2.22),
            "text": (
                r"$\mathbf{Point\ B\ (Unstable\ Saddle)}$" + "\n"
                r"$\mathbf{Negative\ Slope\ (\frac{\partial \Delta P}{\partial G} < 0)}$"
            ),
            "fontsize": 9.2,
            "fontweight": "bold",
            "color_text": "#991b1b",
            "bbox": {
                "boxstyle": "round,pad=0.38",
                "fc": "#fff5f5",
                "ec": "#fca5a5",
                "lw": 1.1,
                "alpha": 0.96,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#dc2626",
                "lw": 1.3,
            },
        },
        "pt_C": {
            "name": "C",
            "color": "#007a78",
            "xytext": (2.67, 2.15),
            "text": (
                r"$\mathbf{Point\ C\ (Stable)}$" + "\n"
                r"$\mathbf{Liquid\text{-}Rich\ (\frac{\partial \Delta P}{\partial G} > 0)}$"
            ),
            "fontsize": 9.2,
            "fontweight": "bold",
            "color_text": "#005a58",
            "bbox": {
                "boxstyle": "round,pad=0.38",
                "fc": "#f0fdfa",
                "ec": "#5eead4",
                "lw": 1.1,
                "alpha": 0.96,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#007a78",
                "lw": 1.3,
            },
        },
    },

    # 8. Dynamic Excursion Jump Trajectory
    "excursion_jump": {
        "start": (1.52, 1.40),
        "end": (0.60, 1.40),
        "arrowprops": {
            "arrowstyle": "-|>",
            "color": "#dc2626",
            "lw": 2.2,
            "ls": "-",
            "mutation_scale": 16,
            "connectionstyle": "arc3,rad=-0.20",
        },
        "text": (
            r"$\mathbf{Runaway\ Flow\ Collapse}$" + "\n"
            r"$\mathbf{B \rightarrow A\ (Burnout\ /\ CHF)}$"
        ),
        "x": 1.06,
        "y": 1.12,
        "fontsize": 9.0,
        "color": "#991b1b",
        "ha": "center",
        "va": "top",
        "bbox": {
            "boxstyle": "round,pad=0.32",
            "fc": "#fff5f5",
            "ec": "#f87171",
            "lw": 1.0,
            "alpha": 0.94,
        },
    },

    # 9. Stability Criterion Card
    "stability_card": {
        "coordinate_system": "data",
        "x": 0.16,
        "y": 2.54,
        "ha": "left",
        "va": "top",
        "text": (
            r"$\mathbf{Ledinegg\ Stability\ Criterion:}$" + "\n"
            r"$\mathbf{\frac{\partial \Delta P_{\mathrm{ext}}}{\partial G} < \frac{\partial \Delta P_{\mathrm{int}}}{\partial G}\ \rightarrow\ Stable\ (A,\ C)}$" + "\n"
            r"$\mathbf{\frac{\partial \Delta P_{\mathrm{ext}}}{\partial G} > \frac{\partial \Delta P_{\mathrm{int}}}{\partial G}\ \rightarrow\ Excursion\ (B)}$"
        ),
        "fontsize": 9.6,
        "fontweight": "bold",
        "color": "#1a3c6e",
        "bbox": {
            "boxstyle": "round,pad=0.45",
            "fc": "#f8fafc",
            "ec": "#94a3b8",
            "lw": 1.1,
            "alpha": 0.96,
        },
    },

    # 10. Legend
    "legend": {
        "loc": "lower right",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 9.6,
    },
}


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_d(config=CONFIG):
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

    # Generate Mass Flux Domain & Physics S-Curve
    g_min, g_max, g_pts = config["curves"]["g_range"]
    G = np.linspace(g_min, g_max, g_pts)
    
    def s_curve_func(g):
        x = g - 1.6
        return x**3 - 1.15 * x + 1.45

    dp_int = s_curve_func(G)
    pump_head = config["curves"]["pump_head"]
    pump_line = np.full_like(G, pump_head)

    # Negative Slope Turning Points
    # d(dP)/dG = 3*(G - 1.6)^2 - 1.15 = 0 => G = 1.6 +/- sqrt(1.15/3)
    delta_turn = np.sqrt(1.15 / 3.0)
    g_turn_max = 1.6 - delta_turn  # ~0.981
    g_turn_min = 1.6 + delta_turn  # ~2.219

    # 1. Shaded Instability Zone
    iz_cfg = config["instability_zone"]
    ax.axvspan(g_turn_max, g_turn_min,
               color=iz_cfg["color"],
               alpha=iz_cfg["alpha"],
               label=iz_cfg["label"],
               zorder=1)
    ax.axvline(g_turn_max, color=iz_cfg["border_color"], ls=iz_cfg["border_linestyle"], lw=iz_cfg["border_linewidth"], zorder=2)
    ax.axvline(g_turn_min, color=iz_cfg["border_color"], ls=iz_cfg["border_linestyle"], lw=iz_cfg["border_linewidth"], zorder=2)

    # 2. Plot Characteristics
    c_cfg = config["curves"]
    ax.plot(G, dp_int, color=c_cfg["s_curve"]["color"], lw=c_cfg["s_curve"]["linewidth"],
            ls=c_cfg["s_curve"]["linestyle"], label=c_cfg["s_curve"]["label"], zorder=c_cfg["s_curve"]["zorder"])
    ax.plot(G, pump_line, color=c_cfg["pump_line"]["color"], lw=c_cfg["pump_line"]["linewidth"],
            ls=c_cfg["pump_line"]["linestyle"], label=c_cfg["pump_line"]["label"], zorder=c_cfg["pump_line"]["zorder"])

    # 3. Intersection Roots (Operating Points A, B, C)
    # (G - 1.6)^3 - 1.15*(G - 1.6) = 0 => G = 1.6 - sqrt(1.15), 1.6, 1.6 + sqrt(1.15)
    delta_root = np.sqrt(1.15)
    g_roots = {
        "pt_A": (1.6 - delta_root, pump_head),
        "pt_B": (1.6, pump_head),
        "pt_C": (1.6 + delta_root, pump_head),
    }

    op_cfg = config["operating_points"]
    for pt_key, (gx, gy) in g_roots.items():
        pt_data = op_cfg[pt_key]
        # Plot marker
        ax.plot(gx, gy, "o", color=pt_data["color"],
                markersize=op_cfg["marker_size"],
                markeredgecolor=op_cfg["edge_color"],
                markeredgewidth=op_cfg["edge_width"],
                zorder=op_cfg["zorder"])
        # Annotate
        ax.annotate(pt_data["text"],
                    xy=(gx, gy),
                    xytext=pt_data["xytext"],
                    fontsize=pt_data["fontsize"],
                    fontweight=pt_data["fontweight"],
                    color=pt_data["color_text"],
                    ha="center",
                    bbox=dict(**pt_data["bbox"]),
                    arrowprops=dict(**pt_data["arrowprops"]),
                    zorder=7)

    # 4. Dynamic Flow Excursion Jump Arrow & Text
    exc_cfg = config["excursion_jump"]
    ax.annotate("", xy=exc_cfg["end"], xytext=exc_cfg["start"],
                arrowprops=dict(**exc_cfg["arrowprops"]), zorder=6)
    ax.text(exc_cfg["x"], exc_cfg["y"], exc_cfg["text"],
            fontsize=exc_cfg["fontsize"],
            color=exc_cfg["color"],
            ha=exc_cfg["ha"],
            va=exc_cfg["va"],
            bbox=dict(**exc_cfg["bbox"]),
            zorder=7)

    # 5. Stability Criterion Card
    sc_cfg = config["stability_card"]
    sc_trans = ax.transAxes if sc_cfg.get("coordinate_system", "data") == "axes" else ax.transData
    ax.text(sc_cfg["x"], sc_cfg["y"], sc_cfg["text"],
            transform=sc_trans,
            ha=sc_cfg["ha"], va=sc_cfg["va"],
            fontsize=sc_cfg["fontsize"], color=sc_cfg["color"],
            bbox=dict(**sc_cfg["bbox"]), zorder=7)

    # 6. Add Legend
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
    print("Successfully generated and saved Panel (d) with Aptos to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_d()

