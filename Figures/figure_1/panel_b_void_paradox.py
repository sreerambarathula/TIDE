import sys
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
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_1", "panel_b_void_paradox.png")),
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
        "legend.fontsize": 10.0,
        "axes.linewidth": 1.3,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
        "mathtext_shrink_factor": 0.85,  # Subscript & superscript scaling (default 0.7)
    },

    # 3. Global Color Palette
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "purple": "#6b21a8",
        "slate": "#64748b",
        "light_slate": "#94a3b8",
        "gold": "#b45309",
        "card_bg": "#f8fafc",
        "card_border": "#94a3b8",
        "callout_bg": "#fffaf5",
        "callout_border": "#ea580c",
        "trend_bg": "#f0fdfa",
        "trend_border": "#007a78",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(b)\ Thermodynamic\ Void\ Non\text{-}Linearity\ \alpha(x, P)}$",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Thermodynamic\ Flow\ Quality,\ x\ (\%)}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Cross\text{-}Sectional\ Void\ Fraction,\ \alpha\ (\%)}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 2,
        },
        "xlim": (-2, 102),
        "ylim": (-2, 102),
        "grid": True,
    },

    # 5. Thermodynamic Isobaric Curves (Water-Steam Database)
    "curves": [
        {
            "id": "1bar",
            "label": r"$\mathbf{1\,\mathrm{bar}\ (\rho_f/\rho_g \approx 1600)}$",
            "gamma": 0.598 / 958.4,
            "color": "#94a3b8",
            "linestyle": ":",
            "linewidth": 1.8,
            "zorder": 2,
        },
        {
            "id": "10bar",
            "label": r"$\mathbf{10\,\mathrm{bar}\ (\rho_f/\rho_g \approx 172)}$",
            "gamma": 5.15 / 887.0,
            "color": "#007a78",
            "linestyle": "-.",
            "linewidth": 1.8,
            "zorder": 3,
        },
        {
            "id": "70bar",
            "label": r"$\mathbf{70\,\mathrm{bar}\ (\rho_f/\rho_g \approx 20.5)\ [Ref]}$",
            "gamma": 36.1 / 740.0,
            "color": "#b5451b",
            "linestyle": "-",
            "linewidth": 2.8,
            "zorder": 4,
        },
        {
            "id": "150bar",
            "label": r"$\mathbf{150\,\mathrm{bar}\ (\rho_f/\rho_g \approx 6.0)}$",
            "gamma": 100.0 / 600.0,
            "color": "#6b21a8",
            "linestyle": "--",
            "linewidth": 2.0,
            "zorder": 3,
        },
        {
            "id": "critical",
            "label": r"$\mathbf{P \to P_c\ (\rho_f/\rho_g = 1.0)\ [Linear]}$",
            "gamma": 1.0,
            "color": "#64748b",
            "linestyle": "--",
            "linewidth": 1.3,
            "zorder": 2,
        },
    ],

    # 6. Reference Operating Point (70 bar, x = 5%)
    "reference_point": {
        "x_val": 0.05,
        "gamma_ref": 36.1 / 740.0,
        "marker": {
            "style": "o",
            "size": 9.0,
            "color": "#b5451b",
            "edgecolor": "black",
            "edgewidth": 1.3,
            "zorder": 6,
        },
        "guide_lines": {
            "color": "#b45309",
            "linestyle": ":",
            "linewidth": 1.8,
            "zorder": 5,
        },
    },

    # 7. Governing Equation Card (Positioned at x=25, y=5, bottom-left)
    "equation_card": {
        "coordinate_system": "data",
        "x": 25.0,
        "y": 5.0,
        "ha": "left",
        "va": "bottom",
        "text": (
            r"$\mathbf{Governing\ Void\ Model:}$" + "\n"
            r"$\mathbf{\alpha(x, P) = \frac{x}{x + (1 - x)\,(\rho_g / \rho_f)}}$"
        ),
        "fontsize": 11.5,
        "color": "#1a3c6e",
        "bbox": {
            "boxstyle": "round,pad=0.50",
            "fc": "#f8fafc",
            "ec": "#94a3b8",
            "lw": 1.2,
            "alpha": 0.96,
        },
    },

    # 8. 5% Void Paradox Callout Box (Positioned at x=7, y=32)
    "callout_box": {
        "xy": (5.0, None),  # Y computed dynamically from alpha_target
        "xytext": (7.0, 32.0),
        "text": (
            r"$\mathbf{5\%\ Mass\ Quality\ (x = 0.05)}$" + "\n"
            r"$\mathbf{\rightarrow\ \alpha = 51.9\%\ Vapor\ Volume!}$" + "\n"
            r"$\mathbf{(Extreme\ Volumetric\ Expansion)}$"
        ),
        "fontsize": 10.0,
        "color": "#9a3412",
        "bbox": {
            "boxstyle": "round,pad=0.50",
            "fc": "#fffaf5",
            "ec": "#ea580c",
            "lw": 1.4,
        },
        "arrowprops": {
            "arrowstyle": "->",
            "color": "#ea580c",
            "lw": 1.8,
        },
    },

    # 9. Pressure Evolution & Phase Slip Trend Annotation (Arrow from (0,100) to (60,60), text at arrow terminus)
    "trend_annotation": {
        "arrow": {
            "xy_start": (0.0, 100.0),
            "xy_end": (60.0, 60.0),
            "arrowstyle": "-|>",
            "color": "#007a78",
            "alpha": 0.5,
            "lw": 2.2,
            "mutation_scale": 14,
            "linestyle": "-",
        },
        "text_box": {
            "x": 61.0,
            "y": 62.0,
            "ha": "left",
            "va": "bottom",
            "text": (
                r"$\mathbf{Pressure\ P\uparrow \;\rightarrow\; \rho_f/\rho_g \downarrow}$" + "\n"
                "Phase Slip & Expansion Diminish"
            ),
            "fontsize": 9.5,
            "fontweight": "bold",
            "color": "#1a3c6e",
            "bbox": {
                "boxstyle": "round,pad=0.42",
                "fc": "#f0fdfa",
                "ec": "#007a78",
                "lw": 1.2,
                "alpha": 0.96,
            },
        },
    },

    # 10. Legend
    "legend": {
        "loc": "lower right",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 10.0,
    },
}


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_b(config=CONFIG):
    # Apply Matplotlib Typography & RC Params
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

    # Quality array
    x_vals = np.linspace(0.0, 1.0, 600)

    # Plot Isobaric Void Curves
    for c in config["curves"]:
        alpha_vals = x_vals / (x_vals + (1.0 - x_vals) * c["gamma"])
        ax.plot(x_vals * 100, alpha_vals * 100,
                color=c["color"],
                lw=c["linewidth"],
                ls=c["linestyle"],
                label=c["label"],
                zorder=c["zorder"])

    # Plot Reference Target Point (70 bar, x = 0.05)
    ref_cfg = config["reference_point"]
    x_tgt = ref_cfg["x_val"]
    alpha_tgt = (x_tgt / (x_tgt + (1.0 - x_tgt) * ref_cfg["gamma_ref"])) * 100.0

    gl = ref_cfg["guide_lines"]
    ax.plot([x_tgt * 100, x_tgt * 100], [0, alpha_tgt], color=gl["color"], ls=gl["linestyle"], lw=gl["linewidth"], zorder=gl["zorder"])
    ax.plot([0, x_tgt * 100], [alpha_tgt, alpha_tgt], color=gl["color"], ls=gl["linestyle"], lw=gl["linewidth"], zorder=gl["zorder"])

    mk = ref_cfg["marker"]
    ax.plot(x_tgt * 100, alpha_tgt, marker=mk["style"], markersize=mk["size"], color=mk["color"],
            mec=mk["edgecolor"], mew=mk["edgewidth"], zorder=mk["zorder"])

    # 1. Add Governing Equation Card
    eq_cfg = config["equation_card"]
    eq_trans = ax.transAxes if eq_cfg.get("coordinate_system", "data") == "axes" else ax.transData
    ax.text(eq_cfg["x"], eq_cfg["y"], eq_cfg["text"],
            transform=eq_trans,
            ha=eq_cfg["ha"], va=eq_cfg["va"],
            fontsize=eq_cfg["fontsize"], color=eq_cfg["color"],
            bbox=dict(**eq_cfg["bbox"]))

    # 2. Add Callout Box for Reference Point
    callout_cfg = config["callout_box"]
    ax.annotate(callout_cfg["text"],
                xy=(x_tgt * 100, alpha_tgt),
                xytext=callout_cfg["xytext"],
                fontsize=callout_cfg["fontsize"],
                color=callout_cfg["color"],
                bbox=dict(**callout_cfg["bbox"]),
                arrowprops=dict(**callout_cfg["arrowprops"]))

    # 3. Add Pressure Evolution / Phase Slip Trend Annotation
    trend_cfg = config["trend_annotation"]
    arr = trend_cfg["arrow"]
    ax.annotate("", xy=arr["xy_end"], xytext=arr["xy_start"],
                xycoords="data",
                arrowprops=dict(arrowstyle=arr["arrowstyle"], color=arr["color"],
                                lw=arr["lw"], ls=arr["linestyle"], mutation_scale=arr["mutation_scale"],
                                alpha=arr.get("alpha", 1.0)))

    tb = trend_cfg["text_box"]
    tb_trans = ax.transAxes if tb.get("coordinate_system", "data") == "axes" else ax.transData
    ax.text(tb["x"], tb["y"], tb["text"],
            transform=tb_trans,
            ha=tb["ha"], va=tb["va"],
            fontsize=tb["fontsize"], color=tb["color"], fontweight=tb["fontweight"],
            bbox=dict(**tb["bbox"]))

    # Add Legend
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
    print("Successfully generated and saved Panel (b) to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_b()
