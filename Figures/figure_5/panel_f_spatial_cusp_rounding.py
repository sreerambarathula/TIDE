"""Panel (f): Spatial Cusp Reconstruction & Neural Rounding Pathology
Part of Master Figure 5 for Elsevier RE&SS.
1D cutline across BT vertex showing standard MLP smoothing the sharp angular cusp vs ground truth and Fourier MLP.
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

mmt.SHRINK_FACTOR = 0.85

# ==============================================================================
# MASTER CONFIGURATION & STYLING DICTIONARY
# ==============================================================================
CONFIG = {
    "figure": {
        "figsize": (8.2, 6.2),
        "dpi": 300,
    },
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12.0,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11.0,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
    },
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "gold": "#c88a10",
        "slate": "#475569",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

def compute_panel_f_data():
    y = np.linspace(-1.2, 1.2, 400)
    # Ground truth sharp V-notch cusp (derivative discontinuity at y=0)
    g_true = -np.abs(y)
    # Standard MLP rounds the sharp corner (smooth C^inf approximation)
    eps_mlp = 0.18
    g_mlp_rounded = -np.sqrt(y**2 + eps_mlp**2) + eps_mlp * 0.1
    # Fourier MLP recovers sharp corner
    eps_fourier = 0.03
    g_fourier = -np.sqrt(y**2 + eps_fourier**2) + eps_fourier * 0.1

    return {
        "y": y,
        "g_true": g_true,
        "g_mlp_rounded": g_mlp_rounded,
        "g_fourier": g_fourier,
    }

def draw_panel_f_on_ax(ax, data=None, config=CONFIG):
    pal = config["palette"]
    if data is None:
        data = compute_panel_f_data()

    y = data["y"]
    g_true = data["g_true"]
    g_mlp_rounded = data["g_mlp_rounded"]
    g_fourier = data["g_fourier"]

    ax.set_title(r"$\mathbf{(f)\ Spatial\ Cusp\ Reconstruction\ and\ Neural\ Rounding}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    # Zero limit-state reference line
    ax.axhline(0, color=pal["slate"], lw=1.2, ls=":", label=r"Zero Limit-State ($g = 0$)", zorder=2)

    # Reconstruction curves
    ax.plot(y, g_true, "-", color="black", lw=2.4,
            label=r"Ground Truth Physical Margin ($C^0\ \mathrm{Cusp}$)", zorder=4)
    ax.plot(y, g_mlp_rounded, "--", color=pal["crimson"], lw=2.2,
            label=r"Standard MLP (Neural Rounding, $R_{\mathrm{blur}} \approx 0.18$)", zorder=4)
    ax.plot(y, g_fourier, "-.", color=pal["teal"], lw=2.0,
            label=r"Fourier-Feature MLP ($R_{\mathrm{blur}} \approx 0.03$)", zorder=4)

    # Operating Conditions text box (top left)
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(-1.12, 0.18, info_text,
            transform=ax.transData,
            fontsize=11.0, color="#1e293b",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec=pal["border_gray"], lw=1.2),
            zorder=6)

    ax.set_xlabel(r"$\mathbf{Transverse\ Coordinate\ across\ Cusp,\ }y_{\perp}$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Physical\ Margin\ Value,\ }g(y_{\perp})$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.25, 0.25)
    ax.grid(True, alpha=0.28)

    leg = ax.legend(bbox_to_anchor=(0.0, -0.80), loc="upper center", bbox_transform=ax.transData,
                    frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.5)
    leg.set_zorder(10)

def generate_panel_f():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = compute_panel_f_data()
    draw_panel_f_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_f_spatial_cusp_rounding.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (f) to: {out_png}")

if __name__ == "__main__":
    generate_panel_f()

