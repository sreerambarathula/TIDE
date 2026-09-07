import sys
"""Panel (c): Localized Error vs. Distance to Stability Boundary
Part of Master Figure 4 for Elsevier RE&SS.
Shows RMSE plotted against parametric distance delta = dist(x, boundary) on a log-log scale,
demonstrating the 10x localized error surge near the Bogdanov-Takens cusp compared to transversal crossing.
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
        "slate": "#475569",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

def load_data():
    cache_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "data", "generated", "fig4_decoupling_data.npz"))
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([sys.executable, os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "scripts", "generate_fig4_ground_truth_data.py"))], check=True)
    return np.load(cache_path)

def draw_panel_c_on_ax(ax, data, config=CONFIG):
    pal = config["palette"]
    deltas = data["deltas"]
    rmse_near_bt = data["rmse_near_bt"]
    rmse_far_control = data["rmse_far_control"]

    ax.set_title(r"$\mathbf{(c)\ Localized\ Error\ vs.\ Distance\ to\ Stability\ Boundary}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    # Critical boundary envelope shading
    ax.axvspan(1e-3, 0.10, color="#fee2e2", alpha=0.60,
               label=r"Critical Zone ($\delta \leq 0.10$)", zorder=1)

    # Point B (BT Cusp trajectory) vs Point A (Transversal control)
    ax.plot(deltas, rmse_near_bt, "o-", color=pal["crimson"], lw=2.2, ms=6.0,
            mec="black", mew=0.8, label="Point B (BT Cusp)", zorder=4)
    ax.plot(deltas, rmse_far_control, "s--", color=pal["teal"], lw=1.8, ms=5.5,
            mec="black", mew=0.8, label="Point A (Transversal)", zorder=4)

    # Top-Right Information Text Box (Scaling Summary & Operating Conditions)
    info_text = (
        r"$\mathbf{Error\ Scaling}$:" + "\n" +
        r"$\bullet\ \mathrm{Point\ B\ (BT\ Cusp)}:\ \mathbf{10.4\times\ surge}$" + "\n" +
        r"$\bullet\ \mathrm{Point\ A\ (Transversal)}:\ \mathrm{Uniform}$" + "\n\n" +
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.96, 0.96, info_text,
            transform=ax.transAxes,
            fontsize=11.0, color="#1e293b",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.48", fc=pal["card_bg"], ec=pal["border_gray"], lw=1.2),
            zorder=6)

    # Annotation pointer highlighting the frontier spike
    ax.annotate(r"$\mathbf{10.4\times\ Error\ Surge}$" + "\n" + r"$\mathrm{RMSE} \rightarrow 0.083$",
                xy=(1.3e-3, 0.083), xytext=(3.5e-3, 0.032),
                fontsize=10.5, color=pal["crimson"],
                arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.4),
                bbox=dict(boxstyle="round,pad=0.38", fc="#fff1f2", ec=pal["crimson"], lw=1.1),
                zorder=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathbf{Distance\ to\ Stability\ Frontier,\ }\delta = \mathrm{dist}(\mathbf{x}, \partial \Omega)$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Local\ Error,\ }\mathrm{RMSE}(\delta)$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(8e-4, 4.0)
    ax.set_ylim(4e-3, 2.2e-1)
    ax.grid(True, which="both", alpha=0.28)

    leg = ax.legend(loc="lower left", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.6)
    leg.set_zorder(10)

def generate_panel_c():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = load_data()
    draw_panel_c_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_c_error_vs_boundary_distance.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (c) to: {out_png}")

if __name__ == "__main__":
    generate_panel_c()
