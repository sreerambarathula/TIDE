"""Panel (b): Near-Boundary Parity Breakdown (|g| <= 0.10)
Part of Master Figure 4 for Elsevier RE&SS.
Exposes the severe degradation of predictions near the stability frontier where sign disagreement reaches 38%.
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
    cache_path = os.path.normpath(r"D:\AGravity\Tide_Tutor\data\generated\fig4_decoupling_data.npz")
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([r"D:\AGravity\Tide_Tutor\.venv\Scripts\python.exe", r"D:\AGravity\Tide_Tutor\scripts\generate_fig4_ground_truth_data.py"], check=True)
    return np.load(cache_path)

def draw_panel_b_on_ax(ax, data, config=CONFIG):
    pal = config["palette"]
    g_true_near = data["g_true_near"]
    g_pred_near = data["g_pred_near"]
    correct_mask = data["correct_mask"]
    r2_near = float(data["r2_near"])
    rmse_near = float(data["rmse_near"])
    sign_err_pct = float(data["sign_error_rate"]) * 100.0

    ax.set_title(r"$\mathbf{(b)\ Near\text{-}Boundary\ Parity\ Breakdown\ (}|g| \leq 0.10\mathbf{)}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    # Correct Sign vs Sign Error points
    ax.scatter(g_true_near[correct_mask], g_pred_near[correct_mask], s=28, color=pal["navy"],
               alpha=0.75, edgecolors="none", label=rf"Correct Sign (${100.0 - sign_err_pct:.0f}\%$)", zorder=3)
    ax.scatter(g_true_near[~correct_mask], g_pred_near[~correct_mask], s=30, color=pal["crimson"],
               alpha=0.88, edgecolors="black", lw=0.5, label=rf"Sign Error (${sign_err_pct:.0f}\%$)", zorder=4)

    # Reference Lines: Ideal Parity and Safety Quadrant Crosshairs
    ax.plot([-0.12, 0.12], [-0.12, 0.12], color="black", lw=1.8, ls="--", label="Ideal Parity Line", zorder=5)
    ax.axhline(0, color="#64748b", lw=1.1, ls=":", zorder=2)
    ax.axvline(0, color="#64748b", lw=1.1, ls=":", zorder=2)

    # Top-Left Information Text Box (Metrics & Operating Conditions)
    info_text = (
        r"$\mathbf{Near\text{-}Boundary\ Metrics}$:" + "\n" +
        r"$\bullet\ N = 450\ \mathrm{critical\ points}$" + "\n" +
        rf"$\bullet\ R^2 = \mathbf{{{r2_near:.2f}}}$" + "\n" +
        rf"$\bullet\ \mathrm{{RMSE}} = \mathbf{{{rmse_near:.3f}}}$" + "\n" +
        rf"$\bullet\ \mathrm{{Sign\ Error\ Rate}} = \mathbf{{{sign_err_pct:.0f}\%}}$" + "\n\n" +
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(-0.112, 0.138, info_text,
            transform=ax.transData,
            fontsize=11.0, color="#1e293b",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.48", fc="#fff1f2", ec=pal["crimson"], lw=1.2),
            zorder=6)

    ax.set_xlabel(r"$\mathbf{Ground\text{-}Truth\ Margin,\ }g_{\mathrm{true}}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Surrogate\ Prediction,\ }\hat{g}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.15, 0.15)
    ax.grid(True, alpha=0.28)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.6)
    leg.set_zorder(10)

def generate_panel_b():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = load_data()
    draw_panel_b_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_b_near_boundary_parity.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (b) to: {out_png}")

if __name__ == "__main__":
    generate_panel_b()
