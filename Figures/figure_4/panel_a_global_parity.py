import sys
"""Panel (a): Global Validation Parity Plot (N = 3,500 Test Points)
Part of Master Figure 4 for Elsevier RE&SS.
Demonstrates deceptive aggregate metrics (R^2 = 1.0000, RMSE = 0.013) across the global operational domain,
with the critical boundary region highlighted and connected to Panel (b).
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

def draw_panel_a_on_ax(ax, data, config=CONFIG):
    pal = config["palette"]
    g_true = data["g_true_global"]
    g_pred = data["g_pred_global"]
    r2 = float(data["r2_global"])
    rmse = float(data["rmse_global"])
    mae = float(data["mae_global"])

    ax.set_title(r"$\mathbf{(a)\ Global\ Validation\ Parity\ (}N = 3{,}500\mathbf{\ Test\ Points)}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    # Global scatter & ideal parity line
    ax.scatter(g_true, g_pred, s=14, color=pal["navy"], alpha=0.35, edgecolors="none",
               label="Predictions", zorder=3)
    ax.plot([-4.2, 4.2], [-4.2, 4.2], color=pal["crimson"], lw=2.0, ls="--",
            label="Ideal Parity Line", zorder=4)

    # Highlight boundary window (|g| <= 0.10)
    rect_w, rect_h = 0.24, 0.30
    zoom_rect = plt.Rectangle((-rect_w/2, -rect_h/2), rect_w, rect_h,
                              fill=True, facecolor="#fee2e2", edgecolor=pal["crimson"],
                              lw=1.6, ls="-", zorder=8, alpha=0.45)
    ax.add_patch(zoom_rect)
    ax.plot([-rect_w/2, rect_w/2, rect_w/2, -rect_w/2, -rect_w/2],
            [-rect_h/2, -rect_h/2, rect_h/2, rect_h/2, -rect_h/2],
            color=pal["crimson"], lw=1.6, zorder=9)

    ax.annotate(r"$\mathbf{Critical\ Boundary\ Zone}$" + "\n" + r"$(|g| \leq 0.10 \rightarrow \mathbf{Panel\ b})$",
                xy=(0.14, -0.05), xytext=(0.80, -1.65),
                fontsize=11.0, color=pal["crimson"],
                arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.4),
                bbox=dict(boxstyle="round,pad=0.42", fc="#fff1f2", ec=pal["crimson"], lw=1.1), zorder=10)

    # Top-Left Information Text Box (Metrics & Operating Conditions)
    info_text = (
        r"$\mathbf{Global\ Evaluation\ Metrics}$:" + "\n" +
        r"$\bullet\ N = 3{,}500\ \mathrm{test\ points}$" + "\n" +
        rf"$\bullet\ R^2 = \mathbf{{{r2:.4f}}}$" + "\n" +
        rf"$\bullet\ \mathrm{{RMSE}} = \mathbf{{{rmse:.3f}}}$" + "\n" +
        rf"$\bullet\ \mathrm{{MAE}} = \mathbf{{{mae:.3f}}}$" + "\n\n" +
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(-3.85, 3.85, info_text,
            transform=ax.transData,
            fontsize=11.0, color="#1e293b",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.52", fc=pal["card_bg"], ec=pal["border_gray"], lw=1.2),
            zorder=6)

    ax.set_xlabel(r"$\mathbf{Ground\text{-}Truth\ Margin,\ }g_{\mathrm{true}}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Surrogate\ Prediction,\ }\hat{g}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.grid(True, alpha=0.28)
    
    # Legend in lower right
    leg = ax.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.6)
    leg.set_zorder(10)

def generate_panel_a():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = load_data()
    draw_panel_a_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_a_global_parity.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (a) to: {out_png}")

if __name__ == "__main__":
    generate_panel_a()
