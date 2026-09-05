r"""Panel (f): Supercritical Hopf Bifurcation & Limit-Cycle Attractor
Modular, Fully-Configurable Python Script styled with modern Aptos typography.

Features:
- Aptos font family with STIX-Sans mathematical fontset.
- Complete `CONFIG` styling dictionary controlling all typography, colors, geometries, and texts.
- Phase-space attractor portrait: Inlet velocity u_i(t) vs Boiling boundary lambda(t).
- Unstable focus point x^* (Re(\mu) > 0) with diverging spiral trajectory.
- Stable limit-cycle orbit \Gamma with dynamical flow direction arrows.
- Inset time-series demonstrating nonlinear oscillation amplitude saturation.
- Governing Hopf Normal Form equation card.
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
        "output_path": "d:/AGravity/Tide_Tutor/Figures/figure_1/panel_f_hopf_orbit.png",
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
        "legend.fontsize": 9.6,
        "axes.linewidth": 1.3,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
        "mathtext_shrink_factor": 0.85,  # Subscript & superscript scaling (default 0.7)
    },

    # 3. Global Color Palette
    "palette": {
        "orbit": "#dc2626",            # Crimson / Red
        "spiral": "#c88a10",           # Amber / Gold
        "focus_marker": "#1e293b",     # Dark Slate
        "card_bg": "#f8fafc",
        "card_border": "#94a3b8",
        "inset_bg": "#f8fafc",
        "inset_border": "#cbd5e1",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(f)}$ Supercritical Hopf Limit-Cycle Attractor",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Inlet\ Liquid\ Velocity,\ u_i(t)\ (m/s)}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Boiling\ Boundary\ Position,\ \lambda(t)\ (m)}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 2,
        },
        "xlim": (0.06, 0.94),
        "ylim": (0.16, 0.86),
        "grid": True,
    },

    # 5. Phase-Space Trajectories
    "phase_space": {
        "focus": {
            "u_center": 0.34,
            "lam_center": 0.56,
            "label": "Unstable Focus",
            "color": "#1e293b",
            "size": 9.5,
            "width": 2.2,
            "zorder": 6,
        },
        "spiral": {
            "label": "Unstable Diverging Spiral",
            "color": "#c88a10",
            "linestyle": ":",
            "linewidth": 1.8,
            "zorder": 3,
        },
        "orbit": {
            "label": "Stable Limit Cycle",
            "color": "#dc2626",
            "linestyle": "-",
            "linewidth": 2.8,
            "zorder": 4,
        },
        "arrows": [
            {"idx": 70, "color": "#dc2626", "scale": 16},
            {"idx": 220, "color": "#dc2626", "scale": 16},
        ],
    },

    # 6. Inset Time-Series
    "inset": {
        "bounds": [0.62, 0.12, 0.35, 0.31],
        "title": r"$\mathbf{Sustained\ u_i(t)\ Oscillations}$",
        "xlabel": r"$\mathbf{Time,\ t\ \rightarrow}$",
        "ylabel": r"$\mathbf{u_i(t)\ \rightarrow}$",
        "color": "#dc2626",
        "linewidth": 2.0,
        "facecolor": "#ffffff",
        "edgecolor": "#94a3b8",
        "alpha": 0.96,
        "title_fontsize": 10.5,
        "label_fontsize": 11.5,
    },
    # 7. Legend
    "legend": {
        "loc": "upper right",
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
def generate_panel_f(config=CONFIG):
    # Apply Typography & RC Params with Subscript Scaling
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

    # Generate Geometry
    ps_cfg = config["phase_space"]
    u_c = ps_cfg["focus"]["u_center"]
    lam_c = ps_cfg["focus"]["lam_center"]

    # 1. Closed Limit Cycle Orbit
    theta = np.linspace(0, 2 * np.pi, 300)
    u_orbit = u_c + 0.24 * np.cos(theta) - 0.04 * np.sin(2 * theta)
    lam_orbit = lam_c + 0.19 * np.sin(theta) + 0.04 * np.cos(2 * theta)

    # 2. Diverging Spiral from Focus (3 turns)
    t_spiral = np.linspace(0, 4.5 * np.pi, 350)
    r_spiral = 0.015 + 0.22 * (1.0 - np.exp(-0.25 * t_spiral))
    u_spiral = u_c + r_spiral * (np.cos(t_spiral) - 0.15 * np.sin(2 * t_spiral))
    lam_spiral = lam_c + 0.72 * r_spiral * (np.sin(t_spiral) + 0.15 * np.cos(2 * t_spiral))

    # Plot Spiral & Limit Cycle
    sp_cfg = ps_cfg["spiral"]
    ax.plot(u_spiral, lam_spiral, color=sp_cfg["color"],
            lw=sp_cfg["linewidth"], ls=sp_cfg["linestyle"], label=sp_cfg["label"], zorder=sp_cfg["zorder"])

    orb_cfg = ps_cfg["orbit"]
    ax.plot(u_orbit, lam_orbit, color=orb_cfg["color"],
            lw=orb_cfg["linewidth"], ls=orb_cfg["linestyle"], label=orb_cfg["label"], zorder=orb_cfg["zorder"])

    # Direction Arrows on Orbit
    for arr in ps_cfg["arrows"]:
        idx = arr["idx"]
        ax.annotate("", xy=(u_orbit[idx], lam_orbit[idx]), xytext=(u_orbit[idx - 5], lam_orbit[idx - 5]),
                    arrowprops=dict(arrowstyle="-|>", color=arr["color"], lw=2.0, mutation_scale=arr["scale"]),
                    zorder=5)

    # Plot Unstable Focus Point Marker
    f_cfg = ps_cfg["focus"]
    ax.plot(u_c, lam_c, "x", color=f_cfg["color"],
            ms=f_cfg["size"], mew=f_cfg["width"], label=f_cfg["label"], zorder=f_cfg["zorder"])

    # 4. Inset Time Series
    ins_cfg = config["inset"]
    inset_ax = ax.inset_axes(ins_cfg["bounds"])
    t_sim = np.linspace(0, 22, 400)
    u_sim = u_c + 0.28 * np.cos(1.85 * t_sim) * (1.0 - np.exp(-0.32 * t_sim))
    inset_ax.plot(t_sim, u_sim, color=ins_cfg["color"], lw=ins_cfg["linewidth"])
    inset_ax.set_title(ins_cfg["title"], fontsize=ins_cfg.get("title_fontsize", 10.5), pad=4, fontweight="bold")
    inset_ax.set_xlabel(ins_cfg["xlabel"], fontsize=ins_cfg.get("label_fontsize", 11.0), labelpad=3, fontweight="bold")
    inset_ax.set_ylabel(ins_cfg["ylabel"], fontsize=ins_cfg.get("label_fontsize", 11.0), labelpad=3, fontweight="bold")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    inset_ax.grid(True, alpha=0.3)
    inset_ax.set_facecolor(ins_cfg["facecolor"])

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
    print("Successfully generated and saved Panel (f) with Aptos to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_f()

