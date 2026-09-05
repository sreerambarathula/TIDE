"""Panel (c): Analytical Log-Distance Singularity Coordinate Embedding
Shows 2D scalar field of psi(x) = ln(||x - x_BT||^2 + eps) centered at Bogdanov-Takens point.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Publication Typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.2,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "mathtext.fontset": "cm",
})

NAVY = "#1a3c6e"
CRIMSON = "#b5451b"
GOLD = "#c88a10"

NSUB_BT = 14.1428
NPCH_BT = 20.5948

nsub_grid = np.linspace(NSUB_BT - 3.5, NSUB_BT + 0.8, 200)
npch_grid = np.linspace(NPCH_BT - 4.5, NPCH_BT + 4.5, 200)
NS, NP = np.meshgrid(nsub_grid, npch_grid)

eps = 1e-3
r2 = (NS - NSUB_BT)**2 + (NP - NPCH_BT)**2
psi_log = np.log(r2 + eps)

fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=300)
ax.set_title(r"$\mathbf{(c)}$ Log-Distance Coordinate Prior $\psi(\mathbf{x}) = \ln(\|\mathbf{x} - \mathbf{x}_{\mathrm{BT}}\|^2 + \varepsilon)$", pad=14, loc="left", fontweight="bold")

cf = ax.contourf(NS, NP, psi_log, levels=np.linspace(-6, 3, 20), cmap="viridis")
cbar = plt.colorbar(cf, ax=ax, pad=0.03, aspect=22)
cbar.set_label(r"Coordinate Prior Magnitude, $\psi(\mathbf{x})$", fontsize=10.5)

cs = ax.contour(NS, NP, psi_log, levels=[-4, -2, 0, 2], colors="white", alpha=0.5, linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=8.0, fmt=r"$\psi = %.0f$")

ax.plot([NSUB_BT], [NPCH_BT], marker="*", color="#f59e0b", mec="black", mew=1.2, ms=16, zorder=6, label=r"BT Singularity Center ($\star$)")

ax.annotate(r"$\mathbf{Singular \; Distance \; Funnel:}$" + "\n" + 
            r"Injects explicit radial distance prior" + "\n" + 
            r"to the Bogdanov-Takens vertex",
            xy=(NSUB_BT, NPCH_BT), xytext=(11.0, 22.5),
            fontsize=8.8, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=GOLD, lw=1.2),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.4))

ax.set_xlabel(r"Subcooling Number, $N_{\mathrm{sub}}$")
ax.set_ylabel(r"Phase Change Number, $N_{\mathrm{pch}}$")
ax.set_xlim(NSUB_BT - 3.5, NSUB_BT + 0.8)
ax.set_ylim(NPCH_BT - 4.5, NPCH_BT + 4.5)
ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="#1e293b", labelcolor="white", edgecolor="#475569")

out_file = os.path.join(output_dir, "panel_c_log_distance_embedding.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (c) to:", out_file)
