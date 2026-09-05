"""Panel (e): 2D Continuous Spatial Error Field |g_pred - g_true| Near BT Cusp
Part of Master Figure 4 for Elsevier RE&SS.
2D contour map in (Nsub, Npch) space demonstrating the sharp localized error ridge concentrated along the g=0 manifold.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.patheffects as pe
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
    cache_path = os.path.normpath(r"D:\AGravity\Tide_Tutor\data\generated\fig4_decoupling_data.npz")
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([r"D:\AGravity\Tide_Tutor\.venv\Scripts\python.exe", r"D:\AGravity\Tide_Tutor\scripts\generate_fig4_ground_truth_data.py"], check=True)
    return np.load(cache_path)

def draw_panel_e_on_ax(ax, data, config=CONFIG):
    pal = config["palette"]
    NSUB_BT = float(data["NSUB_BT"])
    NPCH_BT = float(data["NPCH_BT"])
    nsub_grid = data["nsub_grid"]
    npch_grid = data["npch_grid"]
    error_field = data["error_field"]
    ns_curve = data["ns_curve"]
    lower_b_curve = data["lower_b_curve"]
    upper_b_curve = data["upper_b_curve"]

    ax.set_title(r"$\mathbf{(e)\ 2D\ Spatial\ Error\ Field}\ |\hat{g}(\mathbf{x}) - g_{\mathrm{true}}(\mathbf{x})|\ \mathbf{Near\ BT\ Cusp}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    cf = ax.contourf(nsub_grid, npch_grid, error_field, levels=np.linspace(0, 0.09, 19), cmap="inferno", extend="max")
    cbar = plt.colorbar(cf, ax=ax, pad=0.025, aspect=20)
    cbar.set_label(r"$\mathbf{Absolute\ Error},\ |\hat{g} - g_{\mathrm{true}}|$", fontsize=11.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=10.0)

    # Physical Boundary Trajectories with stroke outline for high visibility in legend & plot
    ax.plot(ns_curve, lower_b_curve, color="white", lw=2.2, ls="--",
            path_effects=[pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()],
            label="Lower Fold Boundary", zorder=4)
    ax.plot(ns_curve, upper_b_curve, color="#38bdf8", lw=2.2, ls="--",
            path_effects=[pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()],
            label="Upper Hopf Boundary", zorder=4)
    ax.plot([NSUB_BT], [NPCH_BT], marker="*", color="#fbbf24", mec="black", mew=1.2, ms=15,
            zorder=6, label="BT Cusp Vertex")

    # Information Text Box (Spatial Summary & Operating Conditions)
    info_text = (
        r"$\mathbf{Spatial\ Error\ Ridge}$:" + "\n" + 
        r"$\bullet\ \mathrm{Concentrated\ along\ boundaries}$" + "\n" +
        r"$\bullet\ \mathrm{Peak\ at\ tangent\ manifold}$" + "\n\n" +
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(10.85, 24.3, info_text,
            transform=ax.transData,
            fontsize=11.0, color="#1e293b",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.50", fc="#ffffff", ec=pal["border_gray"], lw=1.2),
            zorder=8)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(NSUB_BT - 3.5, NSUB_BT + 0.3)
    ax.set_ylim(NPCH_BT - 4.2, NPCH_BT + 4.2)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.6)
    leg.set_zorder(10)

def generate_panel_e():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = load_data()
    draw_panel_e_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_e_spatial_error_field.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (e) to: {out_png}")

if __name__ == "__main__":
    generate_panel_e()
