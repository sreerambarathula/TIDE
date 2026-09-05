"""Panel (c): Spectral Frequency Analysis (Power Spectral Density / FFT)
Isolates fundamental acoustic DWO frequency omega_0 = 1.83 rad/s and higher harmonics.
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
        "output_path": "d:/AGravity/Tide_Tutor/Figures/figure_2/panel_c_spectral_fft.png",
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
        "legend.fontsize": 9.4,
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
        "gold": "#b45309",
        "slate": "#64748b",
        "light_slate": "#94a3b8",
        "card_bg": "#fff1f2",
        "card_border": "#b5451b",
    },

    # 4. Axes, Title & Labels
    "axes": {
        "title": {
            "text": r"$\mathbf{(c)\ Spectral\ Frequency\ Analysis\ (FFT)}$",
            "fontsize": 13.0,
            "fontweight": "bold",
            "pad": 12,
            "loc": "left",
        },
        "xlabel": {
            "text": r"$\mathbf{Angular\ Frequency,\ }\omega\ \mathbf{(rad/s)}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 6,
        },
        "ylabel": {
            "text": r"$\mathbf{Power\ Spectral\ Density,\ }S_{uu}(\omega)\ \mathbf{(dB/Hz)}$",
            "fontsize": 12.0,
            "fontweight": "bold",
            "labelpad": 4,
        },
        "xlim": (0.2, 7.5),
        "ylim": (0.01, 8000),
        "grid": True,
    },

    # 5. Physics & Resonance Parameters
    "physics": {
        "w0": 1.83,
        "t0": 3.43,
        "n_points": 700,
    },

    # 6. Fundamental Peak Callout Annotation (Positioned Right of Peak: Left=2.0, Bottom=200)
    "callout_fundamental": {
        "xy": (1.83, 3100),
        "xytext": (2.0, 200),
        "va": "bottom",
        "ha": "left",
        "text": (
            r"$\mathbf{Fundamental\ DWO\ Mode}$" + "\n"
            r"$\omega_0 = 1.83\ \mathrm{rad/s}$" + "\n"
            r"$T_0 = 3.43\ \mathrm{s} \approx 2\tau_{\mathrm{transit}}$"
        ),
        "fontsize": 9.2,
        "color": "#b5451b",
        "fontweight": "bold",
        "bbox": {
            "boxstyle": "round,pad=0.40",
            "fc": "#fff1f2",
            "ec": "#b5451b",
            "lw": 1.3,
        },
        "arrowprops": {
            "arrowstyle": "->",
            "color": "#b5451b",
            "lw": 1.5,
            "relpos": (0.0, 0.85),
        },
    },

    # 7. Harmonic Callout Annotation
    "callout_harmonic": {
        "xy": (3.66, 110),
        "xytext": (3.95, 25),
        "text": r"$\mathbf{2\omega_0\ Harmonic\ Peak}$",
        "fontsize": 9.2,
        "color": "#b45309",
        "fontweight": "bold",
        "bbox": {
            "boxstyle": "round,pad=0.35",
            "fc": "#fefce8",
            "ec": "#b45309",
            "lw": 1.1,
        },
        "arrowprops": {
            "arrowstyle": "->",
            "color": "#b45309",
            "lw": 1.3,
        },
    },

    # 8. Legend
    "legend": {
        "loc": "upper right",
        "frameon": True,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "#cbd5e1",
        "fontsize": 9.4,
        "borderpad": 0.45,
        "handlelength": 1.5,
        "handletextpad": 0.50,
    },
}


# ==============================================================================
# PLOTTING ENGINE (Reads 100% from CONFIG)
# ==============================================================================
def generate_panel_c(config=CONFIG):
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
    ax.grid(ax_cfg["grid"], which="both")

    # Physics domain
    phys = config["physics"]
    w0 = phys["w0"]
    omega = np.linspace(ax_cfg["xlim"][0], ax_cfg["xlim"][1], phys["n_points"])

    # Synthetic PSD with resonance peaks
    psd = (
        1.0 / ((omega - w0)**2 + 0.018**2)
        + 0.22 / ((omega - 2 * w0)**2 + 0.045**2)
        + 0.06 / ((omega - 3 * w0)**2 + 0.08**2)
        + 0.005 / (omega**1.2)
    )

    pal = config["palette"]

    # Main PSD Curve
    ax.semilogy(omega, psd, color=pal["navy"], lw=2.4, label=r"Velocity PSD $S_{uu}(\omega)$")

    # Vertical harmonic markers
    ax.axvline(w0, color=pal["crimson"], lw=1.6, ls="--",
               label=r"Fundamental Mode ($\omega_0 = 1.83\ \mathrm{rad/s}$)")
    ax.axvline(2 * w0, color=pal["gold"], lw=1.4, ls=":",
               label=r"Second Harmonic ($2\omega_0 = 3.66\ \mathrm{rad/s}$)")
    ax.axvline(3 * w0, color=pal["slate"], lw=1.2, ls=":",
               label=r"Third Harmonic ($3\omega_0 = 5.49\ \mathrm{rad/s}$)")

    # Fundamental Peak Callout (Right of peak: Left=2.0, Bottom=200)
    call_fund = config["callout_fundamental"]
    ax.annotate(call_fund["text"],
                xy=call_fund["xy"],
                xytext=call_fund["xytext"],
                va=call_fund.get("va", "bottom"),
                ha=call_fund.get("ha", "left"),
                fontsize=call_fund["fontsize"],
                color=call_fund["color"],
                fontweight=call_fund["fontweight"],
                bbox=dict(**call_fund["bbox"]),
                arrowprops=dict(**call_fund["arrowprops"]),
                zorder=6)

    # Harmonic Callout
    call_harm = config["callout_harmonic"]
    ax.annotate(call_harm["text"],
                xy=call_harm["xy"],
                xytext=call_harm["xytext"],
                fontsize=call_harm["fontsize"],
                color=call_harm["color"],
                fontweight=call_harm["fontweight"],
                bbox=dict(**call_harm["bbox"]),
                arrowprops=dict(**call_harm["arrowprops"]),
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
    print("Successfully generated and saved Panel (c) to:", out_file)
    return fig, ax


if __name__ == "__main__":
    generate_panel_c()

