import sys
"""Panel (f): 2D Newton Sum-and-Product Solver Residual Convergence
Contrasts rapid quadratic convergence of polynomial trace/determinant solver against divergence of raw-eigenvalue solver.
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
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_2", "panel_f_newton_convergence.png")),
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
        "legend.fontsize": 9.0,
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
        "callout_blue_bg": "#eff6ff",
        "callout_blue_border": "#3b82f6",
        "callout_crimson_bg": "#fff7ed",
        "callout_crimson_border": "#b5451b",
    },

    # 4. Solver Convergence Data (from double_zero.py benchmark)
    "data": {
        "iters_sum_prod": np.arange(1, 8),
        "res_sum_prod": np.array([4.2e-1, 8.5e-2, 3.1e-3, 4.8e-6, 1.2e-11, 2.5e-13, 2.1e-13]),
        "iters_raw": np.array([1, 2, 3, 4, 5]),
        "res_raw": np.array([4.2e-1, 3.8e-1, 1.2e0, 8.5e1, 1.4e4]),
        "target_tol": 1e-11,
    },

    # 5. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(f)\ 2D\ Newton\ Sum\text{-}and\text{-}Product\ Solver\ Convergence}$",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Newton\ Iteration\ Count,\ }k$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Nonlinear\ Residual\ Norm,\ }\|\mathbf{R}\|$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "xlim": (0.6, 7.4),
        "ylim": (1e-15, 2e6),
        "xticks": np.arange(1, 8),
        "grid": True,
    },

    # 6. Callout Annotations
    "callouts": {
        "convergence": {
            "xy": (5, 1.2e-11),
            "xytext": (4.7, 4.0e-5),
            "text": (
                r"$\mathbf{Smooth\ Analytic\ Formulation}$:" + "\n"
                r"$\Rightarrow$ Quadratic descent to $10^{-13}$"
            ),
            "fontsize": 9.2,
            "color": "#1e40af",
            "fontweight": "bold",
            "bbox": {
                "boxstyle": "round,pad=0.42",
                "fc": "#eff6ff",
                "ec": "#3b82f6",
                "lw": 1.3,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#3b82f6",
                "lw": 1.4,
            },
        },
        "divergence": {
            "xy": (4, 8.5e1),
            "xytext": (2.2, 2.0e3),
            "text": (
                r"$\mathbf{Branch\text{-}Point\ Singularity}$:" + "\n"
                r"$\nabla\lambda_i \to \infty \Rightarrow$ Diverges to $\mathrm{NaN}$"
            ),
            "fontsize": 9.2,
            "color": "#b5451b",
            "fontweight": "bold",
            "bbox": {
                "boxstyle": "round,pad=0.42",
                "fc": "#fff7ed",
                "ec": "#b5451b",
                "lw": 1.3,
            },
            "arrowprops": {
                "arrowstyle": "->",
                "color": "#b5451b",
                "lw": 1.4,
            },
        },
    },

    # 7. Legend
    "legend": {
        "loc": "lower left",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 10.0,
        "borderpad": 0.45,
        "handlelength": 1.5,
        "handletextpad": 0.50,
    },
}


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_f(config=CONFIG):
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

    pal = config["palette"]
    data = config["data"]

    # Plot Curves
    ax.semilogy(data["iters_sum_prod"], data["res_sum_prod"], "o-",
                color=pal["navy"], lw=2.4, ms=8.0, mec="black", mew=1.2,
                label=r"Sum-and-Product Solver")

    ax.semilogy(data["iters_raw"], data["res_raw"], "s--",
                color=pal["crimson"], lw=2.0, ms=8.0, mec="black", mew=1.2,
                label=r"Naive Raw-Eigenvalue Newton (Diverges)")

    # Target Tolerance Line
    ax.axhline(data["target_tol"], color=pal["teal"], lw=1.6, ls=":",
               label=r"Target Tolerance ($10^{-11}$)")

    # Set Limits, Ticks & Grid
    ax.set_xticks(ax_cfg["xticks"])
    ax.set_xlim(*ax_cfg["xlim"])
    ax.set_ylim(*ax_cfg["ylim"])
    ax.grid(ax_cfg["grid"], which="both")

    # Callout Annotations
    calls = config["callouts"]
    c_conv = calls["convergence"]
    ax.annotate(c_conv["text"],
                xy=c_conv["xy"],
                xytext=c_conv["xytext"],
                fontsize=c_conv["fontsize"],
                color=c_conv["color"],
                fontweight=c_conv["fontweight"],
                bbox=dict(**c_conv["bbox"]),
                arrowprops=dict(**c_conv["arrowprops"]),
                zorder=6)

    c_div = calls["divergence"]
    ax.annotate(c_div["text"],
                xy=c_div["xy"],
                xytext=c_div["xytext"],
                fontsize=c_div["fontsize"],
                color=c_div["color"],
                fontweight=c_div["fontweight"],
                bbox=dict(**c_div["bbox"]),
                arrowprops=dict(**c_div["arrowprops"]),
                zorder=6)

    # Legend
    leg_cfg = dict(config["legend"])
    loc = leg_cfg.pop("loc", "lower left")
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
    print("Successfully generated and saved Panel (f) to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_f()
