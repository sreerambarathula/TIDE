r"""Panel (e): Dynamic Density-Wave Transit Delay tau & Regenerative Feedback Loop
Modular, Fully-Configurable Python Script styled with modern Aptos typography.

Features:
- Aptos font family with STIX-Sans mathematical fontset.
- Complete `CONFIG` styling dictionary controlling all typography, colors, geometries, and texts.
- Physical Density Wave Oscillation (DWO) transit delay dynamics:
    * Inlet liquid flow perturbation: \delta u_i(t)
    * Exit two-phase pressure drop feedback: \delta \Delta P_{exit}(t)
    * 180° Anti-phase relationship (\tau_{12} = T/2)
- Regenerative acoustic feedback mechanism card with exact delay physics.
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
        "output_path": "d:/AGravity/Tide_Tutor/Figures/figure_1/panel_e_transit_delay.png",
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
        "inlet_flow": "#1a3c6e",        # Deep Navy
        "exit_pressure": "#dc2626",     # Crimson / Red
        "guide_line": "#64748b",        # Slate Grey
        "delay_arrow": "#b45309",       # Amber / Gold
        "card_bg": "#f8fafc",           # Crisp light card
        "card_border": "#94a3b8",
        "badge_bg": "#fffbeb",          # Amber badge
        "badge_border": "#f59e0b",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(e)}$ Dynamic Density-Wave Transit Delay $\mathbf{\tau_{12}}$ & Feedback",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Dimensionless\ Time,\ t\ /\ T_0}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Perturbation\ Amplitude\ (\delta u_i,\ \delta \Delta P_{\mathrm{exit}})}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 2,
        },
        "xlim": (0.0, 11.5),
        "ylim": (-0.72, 0.75),
        "grid": True,
    },

    # 5. Dynamic Waveforms
    "waveforms": {
        "t_range": (0.0, 11.5, 600),
        "period": 4.0,
        "tau_delay": 2.0,  # T / 2 (180 deg phase lag)
        "inlet_flow": {
            "amp": 0.28,
            "label": r"$\mathbf{Inlet\ Velocity\ Perturbation\ \delta u_i(t)}$",
            "color": "#1a3c6e",
            "linestyle": "-",
            "linewidth": 2.6,
            "zorder": 4,
        },
        "exit_pressure": {
            "amp": 0.38,
            "label": r"$\mathbf{Exit\ Pressure\ Feedback\ \delta \Delta P_{\mathrm{exit}}(t)}$",
            "color": "#dc2626",
            "linestyle": "--",
            "linewidth": 2.4,
            "zorder": 4,
        },
    },

    # 6. Transit Delay & Phase Lag Annotations
    "transit_delay": {
        "t_peak_u": 0.0,
        "t_peak_dp": 2.0,
        "guide_lines": {
            "y_bottom": -0.66,
            "y_top": 0.46,
            "color": "#64748b",
            "linestyle": ":",
            "linewidth": 1.4,
            "zorder": 2,
        },
        "delay_arrow": {
            "y": 0.44,
            "color": "#b45309",
            "linewidth": 2.0,
            "arrowstyle": "<->",
            "zorder": 5,
        },
        "delay_badge": {
            "x": 2.0,
            "y": 0.52,
            "text": r"$\mathbf{\tau_{12} = T/2\ (180^\circ\ Phase\ Lag)}$",
            "fontsize": 9.6,
            "fontweight": "bold",
            "color": "#92400e",
            "ha": "center",
            "va": "bottom",
            "bbox": {
                "boxstyle": "round,pad=0.35",
                "fc": "#fffbeb",
                "ec": "#f59e0b",
                "lw": 1.1,
                "alpha": 0.96,
            },
            "zorder": 6,
        },
    },

    # 7. Acoustic Feedback Mechanism Card
    "mechanism_card": {
        "coordinate_system": "data",
        "x": 5.75,
        "y": -0.56,
        "ha": "center",
        "va": "center",
        "text": (
            r"$\mathbf{Self\text{-}Excited\ Feedback\ Loop:}$" + "\n"
            r"$\mathbf{\delta u_i(t)\ \rightarrow\ \delta \alpha_{\mathrm{exit}}(t)\ [\tau_{12}]\ \rightarrow\ \delta \Delta P_{\mathrm{exit}}(t)\ \rightarrow\ -\delta u_i(t)\ [180^\circ]}$" + "\n"
            r"$\mathbf{\text{Delayed exit vapor packet spikes pressure}\ \rightarrow\ \text{Further chokes inlet flow!}}$"
        ),
        "fontsize": 9.4,
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

    # 8. Legend
    "legend": {
        "loc": "upper right",
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
def generate_panel_e(config=CONFIG):
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

    # Generate Time Series Data
    w_cfg = config["waveforms"]
    t_min, t_max, t_pts = w_cfg["t_range"]
    t = np.linspace(t_min, t_max, t_pts)
    T = w_cfg["period"]
    tau = w_cfg["tau_delay"]

    # Inlet Flow Perturbation: delta u_i(t)
    u_cfg = w_cfg["inlet_flow"]
    delta_ui = u_cfg["amp"] * np.cos(2 * np.pi * t / T)

    # Exit Pressure Feedback: delta \Delta P_exit(t)
    p_cfg = w_cfg["exit_pressure"]
    delta_dp = p_cfg["amp"] * np.cos(2 * np.pi * (t - tau) / T)

    # 1. Zero Baseline
    ax.axhline(0, color="#94a3b8", lw=0.9, ls="-", alpha=0.6, zorder=1)

    # 2. Plot Waveforms
    ax.plot(t, delta_ui, color=u_cfg["color"], lw=u_cfg["linewidth"],
            ls=u_cfg["linestyle"], label=u_cfg["label"], zorder=u_cfg["zorder"])
    ax.plot(t, delta_dp, color=p_cfg["color"], lw=p_cfg["linewidth"],
            ls=p_cfg["linestyle"], label=p_cfg["label"], zorder=p_cfg["zorder"])

    # 3. Transit Delay Guide Lines
    td_cfg = config["transit_delay"]
    gl = td_cfg["guide_lines"]
    t1 = td_cfg["t_peak_u"]
    t2 = td_cfg["t_peak_dp"]

    ax.plot([t1, t1], [gl["y_bottom"], gl["y_top"]], color=gl["color"],
            ls=gl["linestyle"], lw=gl["linewidth"], zorder=gl["zorder"])
    ax.plot([t2, t2], [gl["y_bottom"], gl["y_top"]], color=gl["color"],
            ls=gl["linestyle"], lw=gl["linewidth"], zorder=gl["zorder"])

    # 4. Double-ended Delay Arrow
    da = td_cfg["delay_arrow"]
    ax.annotate("", xy=(t2, da["y"]), xytext=(t1, da["y"]),
                arrowprops=dict(arrowstyle=da["arrowstyle"], color=da["color"], lw=da["linewidth"]),
                zorder=da["zorder"])

    # 5. Delay Badge Text
    db = td_cfg["delay_badge"]
    ax.text(db["x"], db["y"], db["text"],
            fontsize=db["fontsize"],
            fontweight=db["fontweight"],
            color=db["color"],
            ha=db["ha"],
            va=db["va"],
            bbox=dict(**db["bbox"]),
            zorder=db["zorder"])

    # 6. Mechanism Equation Card
    mc_cfg = config["mechanism_card"]
    mc_trans = ax.transAxes if mc_cfg.get("coordinate_system", "data") == "axes" else ax.transData
    ax.text(mc_cfg["x"], mc_cfg["y"], mc_cfg["text"],
            transform=mc_trans,
            ha=mc_cfg["ha"], va=mc_cfg["va"],
            fontsize=mc_cfg["fontsize"], color=mc_cfg["color"],
            bbox=dict(**mc_cfg["bbox"]), zorder=7)

    # 7. Add Legend
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
    print("Successfully generated and saved Panel (e) with Aptos to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_e()