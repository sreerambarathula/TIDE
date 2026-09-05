"""Panel (f): Topological Contrast in Metric Decoupling (Point A vs Point B vs Point C)
Part of Master Figure 4 for Elsevier RE&SS.
Direct comparison of near-boundary vs far-field RMSE across non-degenerate vs degenerate topologies.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import numpy as np

# Font Registration
cloud_dir = r"C:\Users\User\AppData\Local\Microsoft\FontCache\4\CloudFonts"
for d in glob.glob(os.path.join(cloud_dir, "Aptos*")):
    for f in glob.glob(os.path.join(d, "*.ttf")):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass

mmt.SHRINK_FACTOR = 0.85

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
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
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
    },
}

def load_data():
    cache_path = os.path.normpath(r"D:\AGravity\Tide_Tutor\data\generated\fig4_decoupling_data.npz")
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([r"D:\AGravity\Tide_Tutor\.venv\Scripts\python.exe", r"D:\AGravity\Tide_Tutor\scripts\generate_fig4_ground_truth_data.py"], check=True)
    return np.load(cache_path)

def draw_panel_f_on_ax(ax, data, config=CONFIG):
    pal = config["palette"]
    topologies = data["topologies"]
    near_errors = data["near_errors"]
    far_errors = data["far_errors"]
    near_stds = data["near_stds"]
    far_stds = data["far_stds"]

    ax.set_title(r"$\mathbf{(f)\ Decoupling\ Contrast}$:" + r"$\ \mathbf{Transversal\ vs.\ Degenerate\ Topologies}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    x = np.arange(len(topologies))
    w = 0.32

    # Bars
    bars_near = ax.bar(x - w/2, near_errors, w, yerr=near_stds, capsize=4.0, color=pal["crimson"],
                       label=r"Near-Boundary Critical ($|g| \leq 0.10$)", edgecolor="black", lw=1.0, zorder=3)
    bars_far = ax.bar(x + w/2, far_errors, w, yerr=far_stds, capsize=4.0, color=pal["navy"],
                      label=r"Far-Field Bulk ($|g| \geq 1.00$)", edgecolor="black", lw=1.0, zorder=3)

    # Ratio Callouts with prominent, clean font sizes (>=11.0 pt)
    ax.annotate(r"$\mathbf{Uniform}$" + "\n" + r"$\approx \mathbf{1.2\times}$",
                xy=(0 - w/2, near_errors[0] + near_stds[0] + 0.003), xytext=(0 - w/2, 0.060),
                ha="center", fontsize=11.0, color=pal["teal"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#f0fdf4", ec=pal["teal"], lw=1.2),
                arrowprops=dict(arrowstyle="->", color=pal["teal"], lw=1.3), zorder=6)

    ax.annotate(r"$\mathbf{Surge}$" + "\n" + r"$\approx \mathbf{9.2\times}$",
                xy=(1 - w/2, near_errors[1] + near_stds[1] + 0.003), xytext=(1 - w/2, 0.138),
                ha="center", fontsize=11.0, color=pal["crimson"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff1f2", ec=pal["crimson"], lw=1.2),
                arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.3), zorder=6)

    ax.annotate(r"$\mathbf{Surge}$" + "\n" + r"$\approx \mathbf{10.2\times}$",
                xy=(2 - w/2, near_errors[2] + near_stds[2] + 0.003), xytext=(2 - w/2, 0.158),
                ha="center", fontsize=11.0, color=pal["crimson"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff1f2", ec=pal["crimson"], lw=1.2),
                arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.2), zorder=6)

    # Operating Conditions text box
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.03, 0.95, info_text,
            transform=ax.transAxes,
            fontsize=11.0, color="#1e293b",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec=pal["border_gray"], lw=1.2),
            zorder=8)

    ax.set_xticks(x)
    ax.set_xticklabels(topologies, fontsize=11.0, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{Root\ Mean\ Squared\ Error,\ }\mathrm{RMSE}$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylim(0, 0.220)
    ax.grid(True, axis="y", alpha=0.28, zorder=1)

    leg = ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.5)
    leg.set_zorder(10)

def generate_panel_f():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = load_data()
    draw_panel_f_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_f_topological_decoupling_contrast.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (f) to: {out_png}")

if __name__ == "__main__":
    generate_panel_f()

