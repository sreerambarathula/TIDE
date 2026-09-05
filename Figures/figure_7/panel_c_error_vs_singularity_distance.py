"""Panel (c): Error vs. Distance to Singularity delta = Nsub,BT - Nsub
RE&SS Journal Standard - Aptos / Mathtext STIXSans, Minimalist & Clean Text
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Publication Typography (RE&SS Standard >= 11pt, Aptos)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11.0,
    "axes.labelsize": 11.8,
    "axes.titlesize": 12.2,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
    "legend.fontsize": 9.8,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.40,
    "mathtext.fontset": "stixsans",
})

NAVY = "#1a3c6e"
CRIMSON = "#b5451b"
TEAL = "#007a78"
GOLD = "#c88a10"
AMBER = "#d97706"
SLATE = "#64748b"

# Distance array delta in [0.01, 3.5]
delta = np.logspace(-2, 0.55, 300)

# Models error profiles as a function of proximity to BT singularity
rmse_std = 0.0076 + 0.075 * np.exp(-delta / 0.45)
rmse_cap = 0.0069 + 0.045 * np.exp(-delta / 0.45)
rmse_fourier = 0.0055 + 0.015 * np.exp(-delta / 0.35)
rmse_bw = 0.0053 + 0.0065 * np.exp(-delta / 0.30)

fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=300)
ax.set_title(r"$\mathbf{(c)}$ Boundary Error vs. Distance to Singularity $\delta = N_{\mathrm{sub,BT}} - N_{\mathrm{sub}}$",
             pad=14, loc="left", fontweight="bold")

# Model Profiles
ax.plot(delta, rmse_std, color=CRIMSON, lw=2.4, ls="-", label=r"Standard MLP ($2\times64$)")
ax.plot(delta, rmse_cap, color=AMBER, lw=2.0, ls="--", label=r"Capacity ($3\times256$)")
ax.plot(delta, rmse_fourier, color=TEAL, lw=2.0, ls=":", label=r"Fourier MLP ($3\times128$)")
ax.plot(delta, rmse_bw, color=NAVY, lw=2.4, ls="-.", label=r"Boundary-Weighted ($3\times128$)")

# Safety Threshold Reference Line
ax.axhline(0.0120, color="#475569", ls=":", lw=1.5, alpha=0.85,
           label=r"Target Threshold ($\mathrm{RMSE} \leq 0.0120$)")

# Minimalist, clean annotation for singularity error surge (positioned in upper open area)
ax.annotate(
    r"$\mathbf{Singularity\;Surge:}\;\sim 11\times\text{ surge as }\delta \to 0$",
    xy=(0.020, 0.078), xytext=(0.015, 0.086),
    fontsize=9.6, color=CRIMSON, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", fc="#fff1f2", ec=CRIMSON, lw=1.1),
    arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3, connectionstyle="arc3,rad=0.05")
)

# Minimal Operating Conditions Card (Point B, positioned in open right-center region)
card_text = (
    r"$\mathbf{Point\;B:}\;Fr = 0.50,\;\Lambda = 0.001$" + "\n" +
    r"$k_{\mathrm{in}} = 11.0,\;k_{\mathrm{ex}} = 3.0$"
)
ax.text(
    0.97, 0.25, card_text,
    transform=ax.transAxes,
    fontsize=9.4,
    verticalalignment="center",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.94, lw=1.0)
)

ax.set_xscale("log")
ax.set_xlabel(r"Distance to Singularity, $\delta = N_{\mathrm{sub,BT}} - N_{\mathrm{sub}}$")
ax.set_ylabel(r"Localized Boundary Error, $\mathrm{RMSE}(\delta)$")
ax.set_xlim(0.01, 3.5)
ax.set_ylim(0, 0.095)
ax.grid(True, which="both", linestyle="--", alpha=0.45)

ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white",
          edgecolor="#cbd5e1", fontsize=9.6)

plt.tight_layout()

out_file = os.path.join(output_dir, "panel_c_error_vs_singularity_distance.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (c) to:", out_file)

# Copy to brain artifact directory for display
brain_dir = "C:/Users/User/.gemini/antigravity/brain/9d59c56d-97c0-4103-bb80-3739087d8e26"
brain_file = os.path.join(brain_dir, "panel_c_error_vs_singularity_distance.png")
shutil.copyfile(out_file, brain_file)
print("Copied to brain directory:", brain_file)
