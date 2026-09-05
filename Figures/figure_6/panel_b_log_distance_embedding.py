"""Panel (b): Analytical Log-Distance Singularity Coordinate Embedding
Part of Master Figure 6 for Elsevier RE&SS.
Visualizes the 2D analytical log-distance coordinate embedding:
    psi(x) = ln(||x - x_BT||^2 + epsilon)
centered at the Bogdanov-Takens (BT) singularity vertex.
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
        "legend.fontsize": 10.2,
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
        "slate": "#475569",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

# Singularity Vertex Coordinates
NSUB_BT = 14.1428
NPCH_BT = 20.5948
EPSILON = 1e-3

def compute_log_distance_field(nsub_range=(10.5, 15.0), npch_range=(16.0, 25.5), resolution=300):
    nsub = np.linspace(nsub_range[0], nsub_range[1], resolution)
    npch = np.linspace(npch_range[0], npch_range[1], resolution)
    NS, NP = np.meshgrid(nsub, npch)
    
    r2 = (NS - NSUB_BT)**2 + (NP - NPCH_BT)**2
    psi = np.log(r2 + EPSILON)
    return NS, NP, psi

def draw_panel_b(ax, config=CONFIG):
    pal = config["palette"]
    NS, NP, psi = compute_log_distance_field()
    
    ax.set_title(r"$\mathbf{(b)\ Analytical\ Log\text{-}Distance\ Coordinate\ Embedding:\ }\psi(\mathbf{x}) = \ln(\|\mathbf{x} - \mathbf{x}_{\mathrm{BT}}\|^2 + \varepsilon)$",
                 fontsize=12.0, fontweight="bold", loc="left", pad=12)

    # Filled contour field
    levels = np.linspace(-6.0, 3.5, 40)
    cf = ax.contourf(NS, NP, psi, levels=levels, cmap="viridis", extend="both")
    
    # Contour line overlays with clean white labels
    iso_levels = [-4.0, -2.0, 0.0, 1.5, 3.0]
    cs = ax.contour(NS, NP, psi, levels=iso_levels, colors="white", alpha=0.6, linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=9.5, fmt=r"$\psi=%.1f$", manual=False)

    # Colorbar with clean integer ticks
    cbar_ticks = [-6.0, -4.0, -2.0, 0.0, 2.0]
    cbar = plt.colorbar(cf, ax=ax, pad=0.03, aspect=24, shrink=0.98, ticks=cbar_ticks)
    cbar.set_label(r"$\mathbf{Log\text{-}Distance\ Prior,\ }\psi(\mathbf{x})$", fontsize=11.0, fontweight="bold", labelpad=8)
    cbar.ax.tick_params(labelsize=10.5)

    # Bogdanov-Takens Singularity Marker
    ax.plot(NSUB_BT, NPCH_BT, marker="*", color="#fbbf24", mec="#1e293b", mew=1.3, ms=18, zorder=6,
            label=r"$\mathbf{BT\ Singularity\ Vertex\ (}\mathbf{x}_{\mathrm{BT}}\mathbf{)}$")

    # Annotation Box pointing to BT point
    annot_text = (
        r"$\mathbf{Singular\ Distance\ Funnel}$" + "\n" +
        r"$\bullet\ \mathbf{x}_{\mathrm{BT}} = (14.14,\ 20.59)$" + "\n" +
        r"$\bullet\ \psi(\mathbf{x}) \to \ln(\varepsilon)\ \mathrm{at\ vertex}$" + "\n" +
        r"$\bullet\ \mathrm{Radial\ geometric\ prior}$"
    )
    ax.annotate(
        annot_text,
        xy=(NSUB_BT, NPCH_BT), xytext=(11.5, 23.2),
        fontsize=10.2, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["gold"], lw=1.3),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.8, shrinkA=3, shrinkB=6),
        zorder=7
    )

    # Operating Conditions Card
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.03, 0.04, info_text, transform=ax.transAxes,
            fontsize=10.5, color="#1e293b", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["border_gray"], lw=1.1), zorder=8)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_xlim(10.5, 15.0)
    ax.set_ylim(16.0, 25.5)
    ax.tick_params(labelsize=11.0, direction="in", length=3.5)
    ax.grid(True, linestyle=":", alpha=0.22, color="white", zorder=1)

    ax.legend(loc="upper right", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=10.2)

def generate_panel_b():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    
    draw_panel_b(ax)
    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_b_log_distance_embedding.png")
    
    if os.path.exists(out_png):
        try:
            os.remove(out_png)
        except Exception:
            pass

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (b) to: {out_png}")

if __name__ == "__main__":
    generate_panel_b()
