"""Panel (c): Scale-Up from 1D Channel to Natural-Circulation Boiling Loops
Shows bifurcation topology invariance across friction scales Lambda = 0.001 (1D channel) to Lambda = 4.5 (full loop).
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

fig, ax = plt.subplots(figsize=(7.8, 6.8), dpi=300)
ax.set_title(r"$\mathbf{(c)}$ Scale-Up Invariance: 1D Channel vs. Natural-Circulation Loop", pad=14, loc="left", fontweight="bold")

# Case 1: 1D Channel (Lambda = 0.001, Forced)
ns1 = np.linspace(10.5, 14.14, 200)
d1 = 14.14 - ns1
fold1 = 20.59 - 1.48 * d1
hopf1 = 20.59 - 0.42 * d1

# Case 2: Natural Circulation Loop (Lambda = 4.5, Risers & Downcomers)
ns2 = np.linspace(8.0, 12.80, 200)
d2 = 12.80 - ns2
fold2 = 18.20 - 1.62 * d2
hopf2 = 18.20 - 0.50 * d2

ax.plot(ns1, fold1, color=NAVY, lw=2.2, ls="-", label=r"1D Channel Fold ($\Lambda = 0.001$)")
ax.plot(ns1, hopf1, color=TEAL, lw=2.2, ls="-", label=r"1D Channel Hopf ($\Lambda = 0.001$)")
ax.plot([14.14], [20.59], marker="*", color=GOLD, ms=14, zorder=6, label=r"1D BT Singularity ($\star$)")

ax.plot(ns2, fold2, color=CRIMSON, lw=2.2, ls="--", label=r"Loop Scale-Up Fold ($\Lambda = 4.5$)")
ax.plot(ns2, hopf2, color="#ea580c", lw=2.2, ls="--", label=r"Loop Scale-Up Hopf ($\Lambda = 4.5$)")
ax.plot([12.80], [18.20], marker="^", color=CRIMSON, ms=10, zorder=6, label=r"Loop BT Singularity ($\blacktriangle$)")

ax.fill_between(ns1, fold1, hopf1, color="#e0f2fe", alpha=0.35)
ax.fill_between(ns2, fold2, hopf2, color="#fee2e2", alpha=0.25)

ax.annotate(r"$\mathbf{Topological \; Invariance:}$" + "\n" + 
            r"Bogdanov-Takens tangency and acute" + "\n" + 
            r"knife-edge cusp geometry persist" + "\n" + 
            r"in full loop systems ($\Lambda \approx 3\text{--}6$)",
            xy=(12.80, 18.20), xytext=(8.5, 20.2),
            fontsize=8.8, color=NAVY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec=NAVY, lw=1.2),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.3))

ax.set_xlabel(r"Subcooling Number, $N_{\mathrm{sub}}$")
ax.set_ylabel(r"Phase Change Number, $N_{\mathrm{pch}}$")
ax.set_xlim(7.5, 14.8)
ax.set_ylim(11.0, 22.0)
ax.grid(True)
ax.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=8.2)

out_file = os.path.join(output_dir, "panel_c_loop_scaleup_invariance.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (c) to:", out_file)
