"""Panel (d): Misfit-Driven Adaptive Active Learning Allocation
Shows adaptive sampling concentrating points along high-uncertainty boundary corridors.
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
TEAL = "#007a78"
GOLD = "#c88a10"

NSUB_BT = 14.1428
NPCH_BT = 20.5948

np.random.seed(42)
# Initial uniform LHS samples
N_init = 120
nsub_init = np.random.uniform(NSUB_BT - 3.5, NSUB_BT + 0.5, N_init)
npch_init = np.random.uniform(NPCH_BT - 4.5, NPCH_BT + 4.5, N_init)

# Active learning enriched samples (concentrated along fold and hopf lines)
N_adapt = 180
nsub_adapt = np.random.uniform(NSUB_BT - 3.2, NSUB_BT, N_adapt)
d_adapt = NSUB_BT - nsub_adapt
branch_choice = np.random.choice([0, 1], size=N_adapt)
npch_adapt = np.where(branch_choice == 0,
                      NPCH_BT - 1.48 * d_adapt + np.random.normal(0, 0.22, N_adapt),
                      NPCH_BT - 0.42 * d_adapt + np.random.normal(0, 0.22, N_adapt))

fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=300)
ax.set_title(r"$\mathbf{(d)}$ Misfit-Driven Active Learning Allocation", pad=14, loc="left", fontweight="bold")

# Ground truth boundary lines
ns_curve = np.linspace(NSUB_BT - 3.5, NSUB_BT, 300)
d_c = NSUB_BT - ns_curve
ax.plot(ns_curve, NPCH_BT - 1.48 * d_c, color=NAVY, lw=2.0, ls="--", label=r"Lower Fold Boundary ($g = 0$)")
ax.plot(ns_curve, NPCH_BT - 0.42 * d_c, color=TEAL, lw=2.0, ls="--", label=r"Upper Hopf Boundary ($g = 0$)")

# Points
ax.scatter(nsub_init, npch_init, s=20, color="#94a3b8", alpha=0.6, edgecolors="none", label=r"Initial Latin Hypercube Samples ($N = 120$)")
ax.scatter(nsub_adapt, npch_adapt, s=28, color=CRIMSON, alpha=0.85, edgecolors="black", lw=0.6, label=r"Adaptive Active Enriched Points ($N = 180$)")
ax.plot([NSUB_BT], [NPCH_BT], marker="*", color=GOLD, ms=14, zorder=6, label=r"BT Vertex ($\star$)")

ax.annotate(r"$\mathbf{Boundary \; Density \; Surge:}$" + "\n" + 
            r"Active learning packs $\mathbf{60\%}$ of sample" + "\n" + 
            r"budget into the narrow stability corridor",
            xy=(12.8, 19.8), xytext=(10.8, 16.8),
            fontsize=8.8, color=CRIMSON, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff1f2", ec=CRIMSON, lw=1.2),
            arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3))

ax.set_xlabel(r"Subcooling Number, $N_{\mathrm{sub}}$")
ax.set_ylabel(r"Phase Change Number, $N_{\mathrm{pch}}$")
ax.set_xlim(NSUB_BT - 3.5, NSUB_BT + 0.5)
ax.set_ylim(NPCH_BT - 4.5, NPCH_BT + 4.5)
ax.grid(True)
ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=8.6)

out_file = os.path.join(output_dir, "panel_d_active_learning_sampling.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (d) to:", out_file)
