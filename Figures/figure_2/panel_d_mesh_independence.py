import sys
"""Panel (d): Machine-Precision Mesh Independence Across Node Counts
Demonstrates relative error < 10^-15 across N1 = 2 to 16 against closed-form analytical Euler relation Eq. (1.4).
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
        "figsize": (7.6, 5.6),
        "dpi": 300,
        "tight_layout": True,
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_2", "panel_d_mesh_independence.png")),
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

    # 3. Global Color Palette
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "gold": "#b45309",
        "slate": "#64748b",
        "light_slate": "#94a3b8",
        "card_bg": "#eff6ff",
        "card_border": "#1a3c6e",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": "(d) Mesh Independence & Machine Precision",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Single\text{-}Phase\ Moving\ Node\ Count,\ }N_1$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Relative\ Error,\ }|\mathrm{Eu}(N_1) - \mathrm{Eu}_{\mathrm{Euler}}| / \mathrm{Eu}_{\mathrm{Euler}}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 4,
        },
        "xlim": (1.5, 16.5),
        "ylim": (8e-17, 3e-14),
        "xticks": [2, 4, 6, 8, 10, 12, 14, 16],
        "grid": True,
    },

    # 5. Numerical Simulation Data
    "data": {
        "nodes": np.array([2, 4, 6, 8, 10, 12, 14, 16]),
        "rel_errors": np.array([2.2e-16, 4.4e-16, 1.8e-16, 3.5e-16, 2.7e-16, 5.1e-16, 3.3e-16, 4.0e-16]),
        "eps_mach": 2.22e-16,
    },

    # 6. Precision Callout Annotation (Single concise line, placed on upper-left)
    "callout": {
        "xy": (6, 1.8e-16),
        "xytext": (2.2, 3.5e-15),
        "text": r"$\mathbf{Relative\ Error < 10^{-15}\ across\ all\ }N_1$",
        "fontsize": 9.6,
        "color": "#1a3c6e",
        "fontweight": "bold",
        "bbox": {
            "boxstyle": "round,pad=0.40",
            "fc": "#eff6ff",
            "ec": "#1a3c6e",
            "lw": 1.3,
        },
        "arrowprops": {
            "arrowstyle": "->",
            "color": "#1a3c6e",
            "lw": 1.4,
        },
    },

    # 7. Legend
    "legend": {
        "loc": "upper right",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 9.4,
        "borderpad": 0.45,
        "handlelength": 1.6,
        "handletextpad": 0.50,
    },
}


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_d(config=CONFIG):
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
    ax.set_xticks(ax_cfg["xticks"])
    ax.grid(ax_cfg["grid"], which="both")

    # Data
    nodes = config["data"]["nodes"]
    rel_errors = config["data"]["rel_errors"]
    eps_mach = config["data"]["eps_mach"]
    pal = config["palette"]

    # Discretized relative error line & points
    ax.semilogy(nodes, rel_errors, "o-", color=pal["navy"], lw=2.2, ms=8,
                mec="black", mew=1.2,
                label=r"Discretized Model vs. Analytical Solution")

    # Machine precision floor line
    ax.axhline(eps_mach, color=pal["crimson"], lw=1.8, ls="--",
               label=r"IEEE 754 Double Precision ($\epsilon_{\mathrm{mach}} \approx 2.22 \times 10^{-16}$)")

    # Annotation Callout
    call_cfg = config["callout"]
    ax.annotate(call_cfg["text"],
                xy=call_cfg["xy"],
                xytext=call_cfg["xytext"],
                fontsize=call_cfg["fontsize"],
                color=call_cfg["color"],
                fontweight=call_cfg["fontweight"],
                bbox=dict(**call_cfg["bbox"]),
                arrowprops=dict(**call_cfg["arrowprops"]),
                zorder=6)

    # Legend
    leg_cfg = dict(config["legend"])
    loc = leg_cfg.pop("loc", "upper right")
    ax.legend(loc=loc, **leg_cfg)

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
    print("Successfully generated and saved Panel (d) to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_d()

