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
        "figsize": (7.6, 5.6),
        "dpi": 300,
        "tight_layout": True,
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_2", "panel_b_enthalpy_evolution.png")),
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
        "legend.fontsize": 8.8,
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
        "gold": "#c88a10",
        "slate": "#64748b",
        "light_slate": "#94a3b8",
        "node_fill": ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa"],
        "twophase_fill": "#ffedd5",
        "callout_bg": "#fff7ed",
        "callout_border": "#b5451b",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(b)\ Multi\text{-}Node\ Enthalpy\ Field\ Evolution}$",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Dimensionless\ Time,\ } t$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Axial\ Channel\ Coordinate,\ } z$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 4,
        },
        "xlim": (0, 20),
        "ylim": (0, 1.05),
        "grid": True,
    },

    # 5. Boundary Callout Annotation
    "callout": {
        "target_t": 10.0,
        "xytext": (6.5, 0.82),
        "text": (
            r"$\mathbf{Moving\ Boiling\ Boundary\ \lambda(t)}$" + "\n"
            r"$(h = h_{\mathrm{sat},f}\ \mathrm{Saturation\ Enthalpy\ Reached})$"
        ),
        "fontsize": 10.2,
        "color": "#b5451b",
        "fontweight": "bold",
        "bbox": {
            "boxstyle": "round,pad=0.45",
            "fc": "#fff7ed",
            "ec": "#b5451b",
            "lw": 1.3,
        },
        "arrowprops": {
            "arrowstyle": "->",
            "color": "#b5451b",
            "lw": 1.5,
        },
    },

    # 6. Legend
    "legend": {
        "loc": "lower right",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 9.8,
        "borderpad": 0.40,
        "handlelength": 1.4,
        "handletextpad": 0.50,
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

    # Time domain
    t = np.linspace(0, 20, 600)

    # Moving node boundaries l_1(t), l_2(t), l_3(t), lambda(t) for N1=4
    lam = 0.55 + 0.12 * np.sin(1.83 * t) * (1.0 - np.exp(-0.35 * t))
    l3 = 0.75 * lam
    l2 = 0.50 * lam
    l1 = 0.25 * lam

    fills = config["palette"]["node_fill"]

    # Shaded enthalpy regions
    ax.fill_between(t, 0, l1, color=fills[0], alpha=0.9, label=r"Subcooled Node 1 ($h_{\mathrm{in}} \to h_1$)")
    ax.fill_between(t, l1, l2, color=fills[1], alpha=0.9, label=r"Subcooled Node 2 ($h_1 \to h_2$)")
    ax.fill_between(t, l2, l3, color=fills[2], alpha=0.9, label=r"Subcooled Node 3 ($h_2 \to h_3$)")
    ax.fill_between(t, l3, lam, color=fills[3], alpha=0.9, label=r"Subcooled Node 4 ($h_3 \to h_{\mathrm{sat},f}$)")
    ax.fill_between(t, lam, 1.0, color=config["palette"]["twophase_fill"], alpha=0.85, label=r"Two-Phase Mixture ($\rho_m < \rho_f$)")

    # Boundary lines
    ax.plot(t, l1, color=config["palette"]["slate"], lw=1.0, ls=":")
    ax.plot(t, l2, color=config["palette"]["slate"], lw=1.0, ls=":")
    ax.plot(t, l3, color=config["palette"]["slate"], lw=1.0, ls=":")
    ax.plot(t, lam, color=config["palette"]["crimson"], lw=2.4, label=r"Boiling Interface $z = \lambda(t)$")
    ax.axhline(1.0, color=config["palette"]["slate"], lw=1.8, ls="-")

    # Callout Annotation
    call_cfg = config["callout"]
    idx_target = np.argmin(np.abs(t - call_cfg["target_t"]))
    target_y = lam[idx_target]
    ax.annotate(call_cfg["text"],
                xy=(call_cfg["target_t"], target_y),
                xytext=call_cfg["xytext"],
                fontsize=call_cfg["fontsize"],
                color=call_cfg["color"],
                fontweight=call_cfg["fontweight"],
                bbox=dict(**call_cfg["bbox"]),
                arrowprops=dict(**call_cfg["arrowprops"]),
                zorder=6)

    # Legend
    leg_cfg = dict(config["legend"])
    loc = leg_cfg.pop("loc", "lower right")
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
    print("Successfully generated and saved Panel (b) to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_b()

