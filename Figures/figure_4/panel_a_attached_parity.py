import sys
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
from matplotlib.patches import ConnectionPatch
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
        "figsize": (15.2, 6.2),
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
    return np.load(cache_path)

def generate_attached_parity():
    plt.rcParams.update(CONFIG["typography"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    plt.subplots_adjust(wspace=0.22)
    
    data = load_data()
    pal = CONFIG["palette"]

    g_true = data["g_true_global"]
    g_pred = data["g_pred_global"]
    r2_global = float(data["r2_global"])
    rmse_global = float(data["rmse_global"])

    g_true_near = data["g_true_near"]
    g_pred_near = data["g_pred_near"]
    correct_mask = data["correct_mask"]
    r2_near = float(data["r2_near"])
    rmse_near = float(data["rmse_near"])
    sign_err_pct = float(data["sign_error_rate"]) * 100.0

    # -------------------------------------------------------------
    # Subpanel (a1): Global Parity
    # -------------------------------------------------------------
    ax1.set_title(r"$\mathbf{(a_1)\ Global\ Validation\ Parity\ (}N = 3{,}500\mathbf{)}$",
                  fontsize=12.5, fontweight="bold", pad=12, loc="left")

    ax1.scatter(g_true, g_pred, s=14, color=pal["navy"], alpha=0.35, edgecolors="none",
                label="Predictions", zorder=3)
    ax1.plot([-4.2, 4.2], [-4.2, 4.2], color=pal["crimson"], lw=2.0, ls="--",
             label="Ideal Parity Line", zorder=4)

    # Zoom window indicator rectangle at origin
    rect_w, rect_h = 0.24, 0.30
    zoom_rect = plt.Rectangle((-rect_w/2, -rect_h/2), rect_w, rect_h,
                              fill=True, facecolor="#fee2e2", edgecolor=pal["crimson"],
                              lw=1.6, ls="-", zorder=8, alpha=0.45)
    ax1.add_patch(zoom_rect)
    ax1.plot([-rect_w/2, rect_w/2, rect_w/2, -rect_w/2, -rect_w/2],
             [-rect_h/2, -rect_h/2, rect_h/2, rect_h/2, -rect_h/2],
             color=pal["crimson"], lw=1.6, zorder=9)

    ax1.annotate(r"$\mathbf{Boundary\ Zone}$" + "\n" + r"$(|g| \leq 0.10)$",
                 xy=(0.14, -0.05), xytext=(0.85, -1.6),
                 fontsize=11.0, color=pal["crimson"],
                 arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.4),
                 bbox=dict(boxstyle="round,pad=0.40", fc="#fff1f2", ec=pal["crimson"], lw=1.1), zorder=10)

    # Info Box: Global Metrics & Operating Conditions
    info_text_a = (
        r"$\mathbf{Global\ Metrics}$:" + "\n" +
        r"$\bullet\ N = 3{,}500\ \mathrm{test\ points}$" + "\n" +
        rf"$\bullet\ R^2 = \mathbf{{{r2_global:.4f}}}$" + "\n" +
        rf"$\bullet\ \mathrm{{RMSE}} = \mathbf{{{rmse_global:.3f}}}$" + "\n\n" +
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax1.text(-3.85, 3.85, info_text_a, fontsize=11.0, color="#1e293b",
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.50", fc=pal["card_bg"], ec=pal["border_gray"], lw=1.2), zorder=6)

    ax1.set_xlabel(r"$\mathbf{Ground\text{-}Truth\ Margin,\ }g_{\mathrm{true}}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax1.set_ylabel(r"$\mathbf{Surrogate\ Prediction,\ }\hat{g}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax1.set_xlim(-4.2, 4.2)
    ax1.set_ylim(-4.2, 4.2)
    ax1.grid(True, alpha=0.28)
    leg1 = ax1.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
                      edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5)
    leg1.set_zorder(10)

    # -------------------------------------------------------------
    # Subpanel (a2): Near-Boundary Breakdown
    # -------------------------------------------------------------
    ax2.set_title(r"$\mathbf{(a_2)\ Near\text{-}Boundary\ Breakdown\ (|g| \leq 0.10)}$",
                  fontsize=12.5, fontweight="bold", pad=12, loc="left")

    ax2.scatter(g_true_near[correct_mask], g_pred_near[correct_mask], s=28, color=pal["navy"],
                alpha=0.75, edgecolors="none", label=rf"Correct Sign (${100 - sign_err_pct:.0f}\%$)", zorder=3)
    ax2.scatter(g_true_near[~correct_mask], g_pred_near[~correct_mask], s=30, color=pal["crimson"],
                alpha=0.88, edgecolors="black", lw=0.5, label=rf"Sign Error (${sign_err_pct:.0f}\%$)", zorder=4)

    ax2.plot([-0.12, 0.12], [-0.12, 0.12], color="black", lw=1.8, ls="--", label="Ideal Parity Line", zorder=5)
    ax2.axhline(0, color="#64748b", lw=1.1, ls=":", zorder=2)
    ax2.axvline(0, color="#64748b", lw=1.1, ls=":", zorder=2)

    # Info Box: Boundary Metrics
    info_text_b = (
        r"$\mathbf{Near\text{-}Boundary\ Metrics}$:" + "\n" +
        r"$\bullet\ N = 450\ \mathrm{critical\ points}$" + "\n" +
        rf"$\bullet\ R^2 = \mathbf{{{r2_near:.2f}}}$" + "\n" +
        rf"$\bullet\ \mathrm{{RMSE}} = \mathbf{{{rmse_near:.3f}}}$" + "\n" +
        rf"$\bullet\ \mathrm{{Sign\ Error\ Rate}} = \mathbf{{{sign_err_pct:.0f}\%}}$"
    )
    ax2.text(-0.11, 0.045, info_text_b, fontsize=11.0, color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.50", fc="#fff1f2", ec=pal["crimson"], lw=1.2), zorder=6)

    ax2.set_xlabel(r"$\mathbf{Ground\text{-}Truth\ Margin,\ }g_{\mathrm{true}}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax2.set_ylabel(r"$\mathbf{Surrogate\ Prediction,\ }\hat{g}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax2.set_xlim(-0.12, 0.12)
    ax2.set_ylim(-0.15, 0.15)
    ax2.grid(True, alpha=0.28)
    leg2 = ax2.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
                      edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5)
    leg2.set_zorder(10)

    # Save outputs
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_4"))
    out_png = os.path.join(out_dir, "panel_a_attached_parity.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Attached parity figure saved to: {out_png}")

if __name__ == "__main__":
    generate_attached_parity()
