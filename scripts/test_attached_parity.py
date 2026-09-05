import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

plt.rcParams.update({
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
})

PALETTE = {
    "navy": "#1a3c6e",
    "crimson": "#b5451b",
    "teal": "#007a78",
    "slate": "#475569",
    "border_gray": "#94a3b8",
    "card_bg": "#f8fafc",
}

data = np.load(r"d:\AGravity\Tide_Tutor\data\generated\fig4_decoupling_data.npz")
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
# Style 1: Embedded Inset Zoom
# -------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8.2, 6.2), dpi=300)
ax1.set_title(r"$\mathbf{(a)\ Global\ Parity\ \&\ Near\text{-}Boundary\ Inset}$".replace(r"\&", "&"),
              fontsize=12.5, fontweight="bold", pad=12, loc="left")

ax1.scatter(g_true, g_pred, s=12, color=PALETTE["navy"], alpha=0.35, edgecolors="none",
            label="Predictions", zorder=3)
ax1.plot([-4.2, 4.2], [-4.2, 4.2], color=PALETTE["crimson"], lw=1.8, ls="--",
         label="Ideal Parity Line", zorder=4)

ax1.set_xlabel(r"$\mathbf{Ground\text{-}Truth\ Margin,\ }g_{\mathrm{true}}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
ax1.set_ylabel(r"$\mathbf{Surrogate\ Prediction,\ }\hat{g}(\mathbf{x})$", fontsize=12.0, fontweight="bold", labelpad=6)
ax1.set_xlim(-4.2, 4.2)
ax1.set_ylim(-4.2, 4.2)
ax1.grid(True, alpha=0.28)

# Inset Axis in Top Left
ax_ins = inset_axes(ax1, width="46%", height="46%", loc="upper left",
                    bbox_to_anchor=(0.04, 0.04, 0.92, 0.92), bbox_transform=ax1.transAxes)

ax_ins.scatter(g_true_near[correct_mask], g_pred_near[correct_mask], s=20, color=PALETTE["navy"],
               alpha=0.75, edgecolors="none", label="Correct Sign", zorder=3)
ax_ins.scatter(g_true_near[~correct_mask], g_pred_near[~correct_mask], s=22, color=PALETTE["crimson"],
               alpha=0.90, edgecolors="black", lw=0.4, label="Sign Error", zorder=4)
ax_ins.plot([-0.12, 0.12], [-0.12, 0.12], color="black", lw=1.3, ls="--", zorder=5)
ax_ins.axhline(0, color="#64748b", lw=0.9, ls=":", zorder=2)
ax_ins.axvline(0, color="#64748b", lw=0.9, ls=":", zorder=2)

ax_ins.set_xlim(-0.11, 0.11)
ax_ins.set_ylim(-0.13, 0.13)
ax_ins.set_title(r"$\mathbf{Boundary\ Zoom\ (}N = 450\mathbf{)}$", fontsize=10.0, fontweight="bold", pad=4)
ax_ins.set_xlabel(r"$g_{\mathrm{true}}$", fontsize=9.5, fontweight="bold", labelpad=1)
ax_ins.set_ylabel(r"$\hat{g}$", fontsize=9.5, fontweight="bold", labelpad=1)
ax_ins.tick_params(labelsize=8.5)
ax_ins.grid(True, alpha=0.25)
ax_ins.legend(loc="lower right", fontsize=8.5, framealpha=0.9, facecolor="white")

mark_inset(ax1, ax_ins, loc1=2, loc2=4, fc="none", ec=PALETTE["crimson"], lw=1.2, ls=":")

metrics_text = (
    r"$\mathbf{Global\ Metrics}$:" + "\n" +
    r"$\bullet\ N = 3{,}500$" + "\n" +
    rf"$\bullet\ R^2 = \mathbf{{{r2_global:.4f}}}$" + "\n" +
    rf"$\bullet\ \mathrm{{RMSE}} = \mathbf{{{rmse_global:.3f}}}$" + "\n\n" +
    r"$\mathbf{Near\text{-}Boundary}$:" + "\n" +
    rf"$\bullet\ R^2 = \mathbf{{{r2_near:.2f}}}$" + "\n" +
    rf"$\bullet\ \mathrm{{Sign\ Error}} = \mathbf{{{sign_err_pct:.0f}\%}}$"
)
ax1.text(1.2, -2.0, metrics_text, fontsize=11.0, color="#1e293b",
         bbox=dict(boxstyle="round,pad=0.5", fc=PALETTE["card_bg"], ec=PALETTE["border_gray"], lw=1.1), zorder=6)

leg = ax1.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
                 edgecolor=PALETTE["border_gray"], fontsize=11.0, borderpad=0.45)
leg.set_zorder(10)

out1 = r"d:\AGravity\Tide_Tutor\Figures\figure_4\panel_a_inset_zoom.png"
fig1.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig1)

# -------------------------------------------------------------
# Style 2: Side-by-Side Double Subpanel
# -------------------------------------------------------------
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14.0, 5.8), dpi=300)

ax2a.set_title(r"$\mathbf{(a_1)\ Global\ Validation\ Parity\ (}N = 3{,}500\mathbf{)}$",
               fontsize=12.2, fontweight="bold", pad=12, loc="left")
ax2a.scatter(g_true, g_pred, s=12, color=PALETTE["navy"], alpha=0.35, edgecolors="none",
             label="Predictions", zorder=3)
ax2a.plot([-4.2, 4.2], [-4.2, 4.2], color=PALETTE["crimson"], lw=1.8, ls="--",
          label="Ideal Parity Line", zorder=4)

zoom_rect = plt.Rectangle((-0.11, -0.13), 0.22, 0.26, fill=False, edgecolor=PALETTE["crimson"], lw=1.5, ls="-", zorder=8)
ax2a.add_patch(zoom_rect)
ax2a.annotate(r"$\mathbf{Boundary\ Zone}$" + "\n" + r"$(|g| \leq 0.10)$",
              xy=(0.15, 0.15), xytext=(0.8, -1.5),
              fontsize=10.5, fontweight="bold", color=PALETTE["crimson"],
              arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.3),
              bbox=dict(boxstyle="round,pad=0.35", fc="#fff1f2", ec=PALETTE["crimson"], lw=1.0), zorder=9)

metrics_text_a = (
    r"$\mathbf{Global\ Metrics}$:" + "\n" +
    r"$\bullet\ N = 3{,}500\ \mathrm{points}$" + "\n" +
    rf"$\bullet\ R^2 = \mathbf{{{r2_global:.4f}}}$" + "\n" +
    rf"$\bullet\ \mathrm{{RMSE}} = \mathbf{{{rmse_global:.3f}}}$" + "\n\n" +
    r"$\mathbf{Operating\ Conditions}$:" + "\n" +
    r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
    r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
)
ax2a.text(-3.85, 3.85, metrics_text_a, fontsize=11.0, color="#1e293b",
          verticalalignment="top",
          bbox=dict(boxstyle="round,pad=0.45", fc=PALETTE["card_bg"], ec=PALETTE["border_gray"], lw=1.1), zorder=6)

ax2a.set_xlabel(r"$\mathbf{Ground\text{-}Truth\ Margin,\ }g_{\mathrm{true}}(\mathbf{x})$", fontsize=11.8, fontweight="bold", labelpad=5)
ax2a.set_ylabel(r"$\mathbf{Surrogate\ Prediction,\ }\hat{g}(\mathbf{x})$", fontsize=11.8, fontweight="bold", labelpad=5)
ax2a.set_xlim(-4.2, 4.2)
ax2a.set_ylim(-4.2, 4.2)
ax2a.grid(True, alpha=0.28)
ax2a.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
            edgecolor=PALETTE["border_gray"], fontsize=11.0, borderpad=0.45)

# Right: Near-Boundary Breakdown
ax2b.set_title(r"$\mathbf{(a_2)\ Near\text{-}Boundary\ Breakdown\ (|g| \leq 0.10)}$",
               fontsize=12.2, fontweight="bold", pad=12, loc="left")
ax2b.scatter(g_true_near[correct_mask], g_pred_near[correct_mask], s=26, color=PALETTE["navy"],
             alpha=0.75, edgecolors="none", label=rf"Correct Sign (${100 - sign_err_pct:.0f}\%$)", zorder=3)
ax2b.scatter(g_true_near[~correct_mask], g_pred_near[~correct_mask], s=28, color=PALETTE["crimson"],
             alpha=0.88, edgecolors="black", lw=0.5, label=rf"Sign Error (${sign_err_pct:.0f}\%$)", zorder=4)

ax2b.plot([-0.12, 0.12], [-0.12, 0.12], color="black", lw=1.6, ls="--", label="Ideal Parity Line", zorder=5)
ax2b.axhline(0, color="#64748b", lw=1.0, ls=":", zorder=2)
ax2b.axvline(0, color="#64748b", lw=1.0, ls=":", zorder=2)

metrics_text_b = (
    r"$\mathbf{Near\text{-}Boundary\ Metrics}$:" + "\n" +
    r"$\bullet\ N = 450\ \mathrm{points}$" + "\n" +
    rf"$\bullet\ R^2 = \mathbf{{{r2_near:.2f}}}$" + "\n" +
    rf"$\bullet\ \mathrm{{RMSE}} = \mathbf{{{rmse_near:.3f}}}$" + "\n" +
    rf"$\bullet\ \mathrm{{Sign\ Error\ Rate}} = \mathbf{{{sign_err_pct:.0f}\%}}$"
)
ax2b.text(-0.11, 0.045, metrics_text_b, fontsize=11.0, color="#1e293b",
          bbox=dict(boxstyle="round,pad=0.45", fc="#fff1f2", ec=PALETTE["crimson"], lw=1.1), zorder=6)

ax2b.set_xlabel(r"$\mathbf{Ground\text{-}Truth\ Margin,\ }g_{\mathrm{true}}(\mathbf{x})$", fontsize=11.8, fontweight="bold", labelpad=5)
ax2b.set_ylabel(r"$\mathbf{Surrogate\ Prediction,\ }\hat{g}(\mathbf{x})$", fontsize=11.8, fontweight="bold", labelpad=5)
ax2b.set_xlim(-0.12, 0.12)
ax2b.set_ylim(-0.15, 0.15)
ax2b.grid(True, alpha=0.28)
ax2b.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
            edgecolor=PALETTE["border_gray"], fontsize=11.0, borderpad=0.45)

out2 = r"d:\AGravity\Tide_Tutor\Figures\figure_4\panel_a_linked_side_by_side.png"
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)

print("[SUCCESS] Both test figures generated!")
