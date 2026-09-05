"""Panel (e): 2D Spatial Distribution of Boundary Loss Weights w(Nsub, Npch)
Part of Master Figure 6 for Elsevier RE&SS.
Visualizes the 2D spatial loss weight field:
    w(x) = 1 / (|g(x)| + epsilon_w)
demonstrating how boundary-weighted loss acts as a spatial magnifying glass along g=0 manifolds.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        "figsize": (8.8, 6.2),
        "dpi": 300,
    },
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 10.0,
        "axes.linewidth": 1.1,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
    },
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "gold": "#c88a10",
        "slate": "#64748b",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

# Singularity Vertex Coordinates
NSUB_BT = 14.1428
NPCH_BT = 20.5948
EPS_W = 0.05

import matplotlib.patheffects as pe

def compute_loss_weight_field(nsub_range=(10.5, 15.0), npch_range=(16.0, 25.5), resolution=300):
    nsub = np.linspace(nsub_range[0], nsub_range[1], resolution)
    npch = np.linspace(npch_range[0], npch_range[1], resolution)
    NS, NP = np.meshgrid(nsub, npch)
    
    # Distance to Fold and Hopf branches
    d = np.maximum(0, NSUB_BT - NS)
    lower_b = NPCH_BT - 1.45 * d
    upper_b = NPCH_BT - 0.42 * d
    
    dist_lower = np.abs(NP - lower_b)
    dist_upper = np.abs(NP - upper_b)
    
    r_bt = np.sqrt((NS - NSUB_BT)**2 + (NP - NPCH_BT)**2)
    dist_boundary = np.where(NS <= NSUB_BT, np.minimum(dist_lower, dist_upper), r_bt)
    
    w_field = 1.0 / (dist_boundary + EPS_W)
    return NS, NP, w_field

def draw_panel_e(ax, config=CONFIG):
    pal = config["palette"]
    NS, NP, w_field = compute_loss_weight_field()

    ax.set_title(r"$\mathbf{(e)\ 2D\ Spatial\ Boundary\ Loss\ Weight\ Field:\ }w(N_{\mathrm{sub}}, N_{\mathrm{pch}})$",
                 fontsize=12.0, fontweight="bold", loc="left", pad=12)

    # Filled contour field covering entire domain from min(w) to 20
    levels = np.linspace(0.05, 20.0, 50)
    cf = ax.contourf(NS, NP, w_field, levels=levels, cmap="plasma", extend="max")
    
    # Contour line overlays with clean white labels
    iso_levels = [2.0, 5.0, 10.0, 15.0]
    cs = ax.contour(NS, NP, w_field, levels=iso_levels, colors="white", alpha=0.45, linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=9.0, fmt=r"$w=%.0f$", manual=False)

    # Colorbar with clean ticks
    cbar_ticks = [2.0, 5.0, 10.0, 15.0, 20.0]
    cbar = plt.colorbar(cf, ax=ax, pad=0.03, aspect=24, shrink=0.98, ticks=cbar_ticks)
    cbar.set_label(r"$\mathbf{Loss\ Weight\ Multiplier,\ }w(\mathbf{x})$", fontsize=11.0, fontweight="bold", labelpad=8)
    cbar.ax.tick_params(labelsize=10.5)

    # Ground truth boundary curves with path effects for visibility over both plasma & white box
    ns_curve = np.linspace(10.5, NSUB_BT, 300)
    d_c = NSUB_BT - ns_curve
    npch_fold = NPCH_BT - 1.45 * d_c
    npch_hopf = NPCH_BT - 0.42 * d_c

    stroke = [pe.Stroke(linewidth=3.2, foreground="#0f172a"), pe.Normal()]

    ax.plot(ns_curve, npch_fold, color="#ffffff", lw=2.2, ls="--", zorder=3,
            path_effects=stroke, label=r"$\mathbf{Lower\ Fold\ Boundary\ (}g=0\mathbf{)}$")
    ax.plot(ns_curve, npch_hopf, color="#38bdf8", lw=2.2, ls="-.", zorder=3,
            path_effects=stroke, label=r"$\mathbf{Upper\ Hopf\ Boundary\ (}g=0\mathbf{)}$")

    # Bogdanov-Takens Singularity Marker
    ax.plot(NSUB_BT, NPCH_BT, marker="*", color="#fbbf24", mec="#0f172a", mew=1.3, ms=18, zorder=6,
            label=r"$\mathbf{BT\ Singularity\ Vertex\ (}\mathbf{x}_{\mathrm{BT}}\mathbf{)}$")

    # Annotation Box placed cleanly in the upper-left open region
    annot_text = (
        r"$\mathbf{Targeted\ Loss\ Ridge}$" + "\n" +
        r"$\bullet\ w_{\max} = 1/\varepsilon_w = 20.0\ \mathrm{along\ }\partial\Omega$" + "\n" +
        r"$\bullet\ \mathrm{Automatic\ spatial\ lens\ along\ }g=0$" + "\n" +
        r"$\bullet\ \mathrm{Focuses\ surrogate\ capacity}$"
    )
    ax.annotate(
        annot_text,
        xy=(12.4, 19.8), xytext=(10.8, 22.8),
        fontsize=10.2, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["gold"], lw=1.3),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.8, shrinkA=3, shrinkB=5),
        zorder=7
    )

    # Operating Conditions Card (placed in bottom-left)
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.04, 0.04, info_text, transform=ax.transAxes,
            fontsize=10.5, color="#1e293b", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["border_gray"], lw=1.1), zorder=8)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_xlim(10.5, 15.0)
    ax.set_ylim(16.0, 25.5)
    ax.tick_params(labelsize=11.0, direction="in", length=3.5)
    ax.grid(True, linestyle=":", alpha=0.18, color="white", zorder=1)

    ax.legend(loc="lower right", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=10.0, handlelength=2.2)

def generate_panel_e():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    
    draw_panel_e(ax)
    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_e_2d_loss_weight_field.png")
    
    if os.path.exists(out_png):
        try:
            os.remove(out_png)
        except Exception:
            pass

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (e) to: {out_png}")

if __name__ == "__main__":
    generate_panel_e()
