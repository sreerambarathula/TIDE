"""Panel (d): Spatial Compression & Collapse of Continuous Margin Profiles g(x)
Shows 1D transverse cross-sections of the limit-state function as delta -> 0, proving corridor compression.
Evaluated directly from machine-precision ODE stability margin fields.
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
    # 1. Figure & Canvas (Matching Panel A, B, C exact (8.2, 6.2) aspect ratio)
    "figure": {
        "figsize": (8.2, 6.2),
        "dpi": 300,
        "tight_layout": True,
        "output_path": os.path.normpath(r"D:\AGravity\Tide_Tutor\Figures\figure_3\panel_d_margin_profile_collapse.png"),
    },

    # 2. Typography & Matplotlib RC Params (Aptos + STIX-Sans)
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10.0,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 11.0,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
        "mathtext_shrink_factor": 0.85,
    },

    # 3. Color Palette
    "palette": {
        "navy": "#1a3c6e",
        "teal": "#007a78",
        "gold": "#c88a10",
        "crimson": "#b5451b",
        "slate": "#64748b",
    },

    # 4. Axes & Labels
    "axes": {
        "title": {
            "text": "(d) Transverse Stability Margin Field Compression",
            "fontsize": 12.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "params_box": r"$\Lambda = 0.001,\ k_{\mathrm{in}} = 11.00,$" + "\n" + r"$k_{\mathrm{ex}} = 3.00,\ Fr = 0.50$",
        "xlabel": {
            "text": r"$\mathbf{Normalized\ Transverse\ Coordinate,\ }y$",
            "fontsize": 11.5,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Continuous\ Stability\ Margin\ Field,\ }g(\mathbf{x})$",
            "fontsize": 11.5,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "xlim": (-1.35, 1.35),
        "ylim": (-0.25, 0.30),
        "grid": True,
    },

    # 5. Legend
    "legend": {
        "loc": "upper right",
        "frameon": True,
        "framealpha": 1.0,
        "facecolor": "white",
        "edgecolor": "#94a3b8",
        "fontsize": 11.0,
        "borderpad": 0.45,
        "handlelength": 1.4,
        "handletextpad": 0.45,
    },
}

def load_ground_truth_data():
    cache_path = os.path.normpath(r"D:\AGravity\Tide_Tutor\data\generated\fig3_continuation_data.npz")
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([r"d:\AGravity\Tide_Tutor\.venv\Scripts\python.exe", os.path.normpath(r"D:\AGravity\Tide_Tutor\scripts\generate_fig3_ground_truth_data.py")], check=True)
    return np.load(cache_path)

def generate_panel_d(config=CONFIG):
    data = load_ground_truth_data()
    y_norm = data["y_norm"]
    g_d10 = data["d_1.00"]
    g_d05 = data["d_0.50"]
    g_d02 = data["d_0.20"]
    g_d005 = data["d_0.05"]

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

    # Condition box at top-left (Matching Panel b & c standard, Sized Up Font)
    ax.text(0.04, 0.94, ax_cfg["params_box"], transform=ax.transAxes,
            fontsize=11.0, va="top", ha="left", color="#1e293b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.40", fc="#f8fafc", ec="#94a3b8", lw=1.1, alpha=0.95), zorder=8)

    ax.set_xlabel(ax_cfg["xlabel"]["text"],
                  fontsize=ax_cfg["xlabel"]["fontsize"],
                  fontweight=ax_cfg["xlabel"]["fontweight"],
                  labelpad=ax_cfg["xlabel"]["labelpad"])

    ax.set_ylabel(ax_cfg["ylabel"]["text"],
                  fontsize=ax_cfg["ylabel"]["fontsize"],
                  fontweight=ax_cfg["ylabel"]["fontweight"],
                  labelpad=ax_cfg["ylabel"]["labelpad"])

    pal = config["palette"]

    # 1. Plot Margin Field Curves at Decreasing delta
    ax.plot(y_norm, g_d10, color=pal["navy"], lw=2.4, label=r"$\delta = 1.00$")
    ax.plot(y_norm, g_d05, color=pal["teal"], lw=2.2, ls="--", label=r"$\delta = 0.50$")
    ax.plot(y_norm, g_d02, color=pal["gold"], lw=2.2, ls="-.", label=r"$\delta = 0.20$")
    ax.plot(y_norm, g_d005, color=pal["crimson"], lw=2.6, label=r"$\delta = 0.05$")

    # 2. Limit-State Zero Threshold Line
    ax.axhline(0.0, color=pal["slate"], lw=1.6, ls=":", label=r"Boundary ($g = 0$)")

    # 3. Callout Badge (Concise, objective, no adjectives)
    ax.annotate("Margin Compression:\n" +
                r"$g_{\max} \to 0\ \mathrm{as}\ \delta \to 0$",
                xy=(-0.85, 0.18), xytext=(-0.35, 0.22),
                fontsize=11.0, color="black", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec=pal["crimson"], lw=1.3),
                arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.4), zorder=8)

    ax.set_xlim(*ax_cfg["xlim"])
    ax.set_ylim(*ax_cfg["ylim"])
    ax.grid(ax_cfg["grid"], alpha=0.28)

    # 4. Legend (100% Opaque, Sized Up Font)
    leg_cfg = dict(config["legend"])
    loc = leg_cfg.pop("loc", "upper right")
    leg = ax.legend(loc=loc, **leg_cfg)
    leg.set_zorder(10)

    # Save Output
    if config["figure"]["tight_layout"]:
        plt.tight_layout()

    out_file = config["figure"]["output_path"]
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    plt.savefig(out_file, dpi=config["figure"]["dpi"], bbox_inches="tight")
    print("Successfully generated and saved Panel (d) to:", out_file)
    return fig, ax

if __name__ == "__main__":
    generate_panel_d()
