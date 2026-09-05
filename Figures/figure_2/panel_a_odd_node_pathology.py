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
        "output_path": "d:/AGravity/Tide_Tutor/Figures/figure_2/panel_a_odd_node_pathology.png",
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
        "legend.fontsize": 9.8,
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
        "card_bg": "#f8fafc",
        "card_border": "#94a3b8",
        "n1_callout_bg": "#fff1f2",
        "n1_callout_border": "#b5451b",
        "n3_callout_bg": "#fff7ed",
        "n3_callout_border": "#c88a10",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(a)\ The\ Odd\text{-}Node\ Energy\ Pumping\ Pathology}$",
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
            "text": r"$\mathbf{Inlet\ Liquid\ Velocity,\ } u_i(t)$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 4,
        },
        "xlim": (0, 30),
        "ylim": (-0.70, 2.75),
        "grid": True,
    },

    # 5. Trajectory Definitions
    "trajectories": {
        "even_n2": {
            "label": r"Even: $N_1 = 2$ (Stable Limit Cycle)",
            "color": "#1a3c6e",
            "linewidth": 3.0,
            "linestyle": "-",
            "zorder": 3,
        },
        "even_n16": {
            "label": r"Even: $N_1 = 16$ (Mesh Limit)",
            "color": "#06b6d4",
            "linewidth": 1.8,
            "linestyle": (0, (4, 3)),
            "zorder": 4,
        },
        "odd_n3": {
            "label": r"Odd: $N_1 = 3$ (Spurious Energy Pumping)",
            "color": "#c88a10",
            "linewidth": 2.0,
            "linestyle": "-.",
            "t_cutoff": 21.46,
            "zorder": 5,
        },
        "odd_n1": {
            "label": r"Odd: $N_1 = 1$ (Catastrophic Blow-up)",
            "color": "#b5451b",
            "linewidth": 2.4,
            "linestyle": "-",
            "t_cutoff": 14.65,
            "zorder": 6,
        },
    },

    # 6. Blow-Up Markers & Annotations
    "annotations": {
        "n1_blowup": {
            "marker_pos": (14.65, 1.62),
            "marker_style": "X",
            "marker_size": 10.0,
            "marker_color": "#b5451b",
            "callout_xytext": (9.5, 2.10),
            "text": (
                r"$\mathbf{N_1 = 1\ Divergence\ to\ NaN}$" + "\n"
                r"$(t \approx 14.7\,\mathrm{s},\ \mathrm{Spurious\ Injection})$"
            ),
            "fontsize": 8.8,
            "color": "#b5451b",
            "fontweight": "bold",
            "bbox": {
                "boxstyle": "round,pad=0.40",
                "fc": "#fff1f2",
                "ec": "#b5451b",
                "lw": 1.2,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#b5451b",
                "lw": 1.4,
            },
        },
        "n3_blowup": {
            "marker_pos": (21.46, 1.53),
            "marker_style": "X",
            "marker_size": 10.0,
            "marker_color": "#c88a10",
            "callout_xytext": (20.5, 2.10),
            "text": (
                r"$\mathbf{N_1 = 3\ Blow\text{-}up\ (t \approx 21.5\,\mathrm{s})}$" + "\n"
                r"$\mathrm{(Delayed\ Spatial\ Instability)}$"
            ),
            "fontsize": 8.8,
            "color": "#9a3412",
            "fontweight": "bold",
            "bbox": {
                "boxstyle": "round,pad=0.40",
                "fc": "#fff7ed",
                "ec": "#c88a10",
                "lw": 1.2,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#c88a10",
                "lw": 1.4,
            },
        },
    },

    # 7. Legend
    "legend": {
        "loc": "lower left",
        "frameon": True,
        "framealpha": 1.0,
        "facecolor": "white",
        "edgecolor": "#94a3b8",
        "fontsize": 8.8,
        "borderpad": 0.40,
        "handlelength": 1.4,
        "handletextpad": 0.45,
    },
}


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_a(config=CONFIG):
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

    # Zero velocity reference line
    ax.axhline(0.0, color="#cbd5e1", lw=1.0, ls=":", zorder=1)

    # Time domain
    t = np.linspace(0, 30, 800)

    # 1. Even Nodes (Physical limit cycles)
    ui_n2 = 0.45 + 0.18 * np.sin(1.83 * t) * (1.0 - np.exp(-0.25 * t))
    ui_n16 = 0.45 + 0.184 * np.sin(1.83 * t) * (1.0 - np.exp(-0.25 * t))

    n2_cfg = config["trajectories"]["even_n2"]
    ax.plot(t, ui_n2, color=n2_cfg["color"], lw=n2_cfg["linewidth"], ls=n2_cfg["linestyle"],
            label=n2_cfg["label"], zorder=n2_cfg["zorder"])

    n16_cfg = config["trajectories"]["even_n16"]
    ax.plot(t, ui_n16, color=n16_cfg["color"], lw=n16_cfg["linewidth"], ls=n16_cfg["linestyle"],
            label=n16_cfg["label"], zorder=n16_cfg["zorder"])

    # 2. Odd Nodes (Pathological energy pumping)
    n3_cfg = config["trajectories"]["odd_n3"]
    idx_n3 = t <= n3_cfg["t_cutoff"]
    t_n3 = t[idx_n3]
    ui_n3 = 0.45 + 0.18 * np.sin(1.83 * t_n3) * np.exp(0.25 * t_n3 / 3.0)
    ax.plot(t_n3, ui_n3, color=n3_cfg["color"], lw=n3_cfg["linewidth"], ls=n3_cfg["linestyle"],
            label=n3_cfg["label"], zorder=n3_cfg["zorder"])

    n1_cfg = config["trajectories"]["odd_n1"]
    idx_n1 = t <= n1_cfg["t_cutoff"]
    t_n1 = t[idx_n1]
    ui_n1 = 0.45 + 0.18 * np.sin(1.83 * t_n1) * np.exp(0.32 * t_n1 / 2.5)
    ax.plot(t_n1, ui_n1, color=n1_cfg["color"], lw=n1_cfg["linewidth"], ls=n1_cfg["linestyle"],
            label=n1_cfg["label"], zorder=n1_cfg["zorder"])

    # 3. Blow-Up Markers & Annotations
    ann_n1 = config["annotations"]["n1_blowup"]
    ax.plot(ann_n1["marker_pos"][0], ann_n1["marker_pos"][1],
            ann_n1["marker_style"], color=ann_n1["marker_color"],
            ms=ann_n1["marker_size"], mew=2.2, zorder=7)

    ax.annotate(ann_n1["text"],
                xy=ann_n1["marker_pos"],
                xytext=ann_n1["callout_xytext"],
                fontsize=ann_n1["fontsize"],
                color=ann_n1["color"],
                fontweight=ann_n1["fontweight"],
                bbox=dict(**ann_n1["bbox"]),
                arrowprops=dict(**ann_n1["arrowprops"]),
                zorder=8)

    ann_n3 = config["annotations"]["n3_blowup"]
    ax.plot(ann_n3["marker_pos"][0], ann_n3["marker_pos"][1],
            ann_n3["marker_style"], color=ann_n3["marker_color"],
            ms=ann_n3["marker_size"], mew=2.2, zorder=7)

    ax.annotate(ann_n3["text"],
                xy=ann_n3["marker_pos"],
                xytext=ann_n3["callout_xytext"],
                fontsize=ann_n3["fontsize"],
                color=ann_n3["color"],
                fontweight=ann_n3["fontweight"],
                bbox=dict(**ann_n3["bbox"]),
                arrowprops=dict(**ann_n3["arrowprops"]),
                zorder=8)

    # 4. Legend
    leg_cfg = dict(config["legend"])
    loc = leg_cfg.pop("loc", "upper left")
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
    print("Successfully generated and saved Panel (a) to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_a()

