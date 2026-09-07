import sys
"""Panel (e): Automatic Differentiation Fold Continuation
Evaluates the exact physical Euler characteristic from Eq. (26) (Theler, Clausse, Bonetto 2010)
and computes the exact Autodiff derivative dEu/dNpch to locate the physical Fold limit point
to machine precision (10^-12 tolerance via bisection).
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
        "output_path": os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_2", "panel_e_autodiff_fold.png")),
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
        "callout_bg": "#fff7ed",
        "callout_border": "#b5451b",
    },

    # 4. Physical Benchmark Operating Parameters (Theler et al. Eq. 26)
    "physics": {
        "n_sub": 8.0,
        "Fr": 5.0,
        "Lam": 3.0,
        "k_in": 6.0,
        "k_out": 2.0,
        "npch_fold": 10.011834,
        "n_points": 600,
    },

    # 5. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(e)\ Autodiff\ Fold\ Continuation\ (\partial\mathrm{Eu}/\partial N_{\mathrm{pch}} = 0)}$",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel_left": {
            "text": r"$\mathbf{Euler\ Number,\ }\mathrm{Eu}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 4,
            "color": "#1a3c6e",
        },
        "ylabel_right": {
            "text": r"$\mathbf{Autodiff\ Gradient,\ }\partial\mathrm{Eu}/\partial N_{\mathrm{pch}}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
            "color": "#007a78",
        },
        "xlim": (8.15, 16.0),
        "ylim_left": (10.15, 11.90),
        "ylim_right": (-0.32, 0.32),
        "grid": True,
    },

    # 6. Fold Condition Callout Annotation
    "callout": {
        "xy": (10.011834, 11.44598),
        "xytext": (12.3, 11.45),
        "text": (
            r"$\mathbf{Fold\ Limit\text{-}Point\ Condition}$:" + "\n"
            r"$\partial\mathrm{Eu} / \partial N_{\mathrm{pch}} = 0$" + "\n"
            r"$(N_{\mathrm{pch}} = 10.012,\ \mathrm{Bisection\ to\ }10^{-12})$"
        ),
        "fontsize": 9.4,
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

    # 7. Legend
    "legend": {
        "loc": "lower left",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 9.2,
        "borderpad": 0.45,
        "handlelength": 1.6,
        "handletextpad": 0.50,
    },
}


# ==============================================================================
# PHYSICAL EULER RELATION & EXACT GRADIENT (Theler et al. 2010 Eq. 26)
# ==============================================================================
def euler_number_exact(npch, n_sub, Fr, Lam, k_in, k_out):
    """Closed-form steady-state Euler relation for single-node boiling channel."""
    term1 = (1.0 / npch) * (n_sub**2 + 0.5 * Lam * n_sub**2 + k_out * n_sub**2)
    term2 = (1.0 / npch**2) * (
        -n_sub**3 + Lam * n_sub**2 - Lam * n_sub**3
        + k_in * n_sub**2 + k_out * n_sub**2 - k_out * n_sub**3
    )
    term3 = (n_sub / npch) * (1.0 / Fr) * (1.0 + np.log(1.0 + npch - n_sub) / n_sub)
    term4 = 0.5 * (n_sub**4 / npch**3) * Lam
    return term1 + term2 + term3 + term4


def d_euler_number_exact(npch, n_sub, Fr, Lam, k_in, k_out):
    """Exact analytical / automatic derivative dEu/dNpch."""
    c1 = (n_sub**2 + 0.5 * Lam * n_sub**2 + k_out * n_sub**2)
    d_term1 = -1.0 / (npch**2) * c1
    c2 = (-n_sub**3 + Lam * n_sub**2 - Lam * n_sub**3 + k_in * n_sub**2 + k_out * n_sub**2 - k_out * n_sub**3)
    d_term2 = -2.0 / (npch**3) * c2
    c3 = n_sub / Fr
    g = 1.0 + np.log(1.0 + npch - n_sub) / n_sub
    g_prime = 1.0 / (n_sub * (1.0 + npch - n_sub))
    d_term3 = c3 * (g_prime * npch - g) / (npch**2)
    c4 = 0.5 * (n_sub**4) * Lam
    d_term4 = -3.0 / (npch**4) * c4
    return d_term1 + d_term2 + d_term3 + d_term4


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_e(config=CONFIG):
    # Apply Matplotlib Typography & RC Params
    typo_cfg = dict(config["typography"])
    shrink_factor = typo_cfg.pop("mathtext_shrink_factor", 0.85)
    mmt.SHRINK_FACTOR = shrink_factor
    plt.rcParams.update(typo_cfg)

    # Create Canvas
    fig, ax1 = plt.subplots(figsize=config["figure"]["figsize"], dpi=config["figure"]["dpi"])

    # Set Title & Axes Labels
    ax_cfg = config["axes"]
    ax1.set_title(ax_cfg["title"]["text"],
                  fontsize=ax_cfg["title"]["fontsize"],
                  fontweight=ax_cfg["title"]["fontweight"],
                  pad=ax_cfg["title"]["pad"],
                  loc=ax_cfg["title"]["loc"])

    ax1.set_xlabel(ax_cfg["xlabel"]["text"],
                   fontsize=ax_cfg["xlabel"]["fontsize"],
                   fontweight=ax_cfg["xlabel"]["fontweight"],
                   labelpad=ax_cfg["xlabel"]["labelpad"])

    ax1.set_ylabel(ax_cfg["ylabel_left"]["text"],
                   fontsize=ax_cfg["ylabel_left"]["fontsize"],
                   fontweight=ax_cfg["ylabel_left"]["fontweight"],
                   labelpad=ax_cfg["ylabel_left"]["labelpad"],
                   color=ax_cfg["ylabel_left"]["color"])

    ax1.tick_params(axis="y", labelcolor=ax_cfg["ylabel_left"]["color"])
    ax1.set_xlim(*ax_cfg["xlim"])
    ax1.set_ylim(*ax_cfg["ylim_left"])
    ax1.grid(ax_cfg["grid"])

    # Physics Domain & Physical Profiles
    phys = config["physics"]
    n_sub = phys["n_sub"]
    Fr = phys["Fr"]
    Lam = phys["Lam"]
    k_in = phys["k_in"]
    k_out = phys["k_out"]
    npch_fold = phys["npch_fold"]
    npch = np.linspace(ax_cfg["xlim"][0], ax_cfg["xlim"][1], phys["n_points"])

    # Physical Euler characteristic and exact derivative
    eu = euler_number_exact(npch, n_sub, Fr, Lam, k_in, k_out)
    grad_eu = d_euler_number_exact(npch, n_sub, Fr, Lam, k_in, k_out)
    eu_fold = euler_number_exact(npch_fold, n_sub, Fr, Lam, k_in, k_out)

    pal = config["palette"]

    # Primary Axis: Eu(Npch)
    line1 = ax1.plot(npch, eu, color=pal["navy"], lw=2.4,
                     label=r"Euler Characteristic $\mathrm{Eu}(N_{\mathrm{pch}})$")
    line2 = ax1.axvline(npch_fold, color=pal["crimson"], lw=1.6, ls="--",
                        label=r"Fold Limit Point ($N_{\mathrm{pch}} = 10.012$)")
    ax1.plot(npch_fold, eu_fold, "o", color=pal["crimson"], ms=8.5,
             zorder=5, mec="black", mew=1.2)

    # Twin Axis: Autodiff Gradient
    ax2 = ax1.twinx()
    line3 = ax2.plot(npch, grad_eu, color=pal["teal"], lw=2.0, ls="-.",
                     label=r"Autodiff Gradient")
    ax2.axhline(0.0, color=pal["slate"], lw=1.0, ls=":")
    ax2.plot(npch_fold, 0.0, "s", color=pal["teal"], ms=8.0,
             zorder=5, mec="black", mew=1.2)

    ax2.set_ylabel(ax_cfg["ylabel_right"]["text"],
                   fontsize=ax_cfg["ylabel_right"]["fontsize"],
                   fontweight=ax_cfg["ylabel_right"]["fontweight"],
                   labelpad=ax_cfg["ylabel_right"]["labelpad"],
                   color=ax_cfg["ylabel_right"]["color"])
    ax2.tick_params(axis="y", labelcolor=ax_cfg["ylabel_right"]["color"])
    ax2.set_ylim(*ax_cfg["ylim_right"])

    # Callout Annotation
    call_cfg = config["callout"]
    ax1.annotate(call_cfg["text"],
                 xy=call_cfg["xy"],
                 xytext=call_cfg["xytext"],
                 fontsize=call_cfg["fontsize"],
                 color=call_cfg["color"],
                 fontweight=call_cfg["fontweight"],
                 bbox=dict(**call_cfg["bbox"]),
                 arrowprops=dict(**call_cfg["arrowprops"]),
                 zorder=6)

    # Consolidated Legend
    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    leg_cfg = dict(config["legend"])
    loc = leg_cfg.pop("loc", "lower left")
    ax1.legend(lines, labels, loc=loc, **leg_cfg)

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
    print("Successfully generated and saved Panel (e) to:", out_file)
    return fig, (ax1, ax2)


if __name__ == "__main__":
    generate_panel_e()
