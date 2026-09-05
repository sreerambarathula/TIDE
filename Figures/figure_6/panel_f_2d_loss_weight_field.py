"""Panel (f): 2D Spatial Distribution of Boundary Loss Weights w(Nsub, Npch)
Shows 2D contour map demonstrating how inverse-distance weighting acts as a spatial magnifying glass along g=0 manifolds.
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

NSUB_BT = 14.1428
NPCH_BT = 20.5948

nsub_grid = np.linspace(NSUB_BT - 3.5, NSUB_BT + 0.5, 200)
npch_grid = np.linspace(NPCH_BT - 4.5, NPCH_BT + 4.5, 200)
NS, NP = np.meshgrid(nsub_grid, npch_grid)

d = np.maximum(0, NSUB_BT - NS)
lower_b = NPCH_BT - 1.48 * d
upper_b = NPCH_BT - 0.42 * d

# Margin g(x)
dist_lower = np.abs(NP - lower_b)
dist_upper = np.abs(NP - upper_b)
dist_boundary = np.minimum(dist_lower, dist_upper)

eps_w = 0.05
w_field = 1.0 / (dist_boundary + eps_w)

fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=300)
ax.set_title(r"$\mathbf{(f)}$ 2D Spatial Loss Weight Field $w(N_{\mathrm{sub}}, N_{\mathrm{pch}})$", pad=14, loc="left", fontweight="bold")

cf = ax.contourf(NS, NP, w_field, levels=np.linspace(1, 20, 20), cmap="plasma", extend="max")
cbar = plt.colorbar(cf, ax=ax, pad=0.03, aspect=22)
cbar.set_label(r"Loss Gradient Weight, $w(\mathbf{x}) = \frac{1}{|g(\mathbf{x})| + \varepsilon_w}$", fontsize=10.5)

# Ground truth boundary lines
ns_curve = np.linspace(NSUB_BT - 3.5, NSUB_BT, 300)
d_c = NSUB_BT - ns_curve
ax.plot(ns_curve, NPCH_BT - 1.48 * d_c, color="white", lw=2.0, ls="--", label=r"Lower Fold Boundary ($g = 0$)")
ax.plot(ns_curve, NPCH_BT - 0.42 * d_c, color="cyan", lw=2.0, ls="--", label=r"Upper Hopf Boundary ($g = 0$)")
ax.plot([NSUB_BT], [NPCH_BT], marker="*", color="yellow", ms=14, zorder=6, label=r"$\mathbf{BT}$ Vertex ($\star$)")

ax.annotate(r"$\mathbf{Targeted \; Loss \; Ridge:}$" + "\n" + 
            r"Acts as an automatic spatial lens," + "\n" + 
            r"focusing backpropagation gradients" + "\n" + 
            r"strictly onto the knife-edge boundary",
            xy=(12.5, 17.5), xytext=(10.8, 22.8),
            fontsize=8.8, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1e293b", ec="#38bdf8", lw=1.2),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.3))

ax.set_xlabel(r"Subcooling Number, $N_{\mathrm{sub}}$")
ax.set_ylabel(r"Phase Change Number, $N_{\mathrm{pch}}$")
ax.set_xlim(NSUB_BT - 3.5, NSUB_BT + 0.5)
ax.set_ylim(NPCH_BT - 4.5, NPCH_BT + 4.5)
ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="#1e293b", labelcolor="white", edgecolor="#475569")

out_file = os.path.join(output_dir, "panel_f_2d_loss_weight_field.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (f) to:", out_file)
