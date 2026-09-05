"""Figure 1: schematic S-shaped pressure-drop-vs-flow-rate curve illustrating
the Ledinegg (fold) mechanism, plus the flat external/pump characteristic
and the three intersection points. Purely illustrative -- not simulated
from the project's physical model, which is why it lives in manuscript/,
not src/tide/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.linewidth": 1.1,
})

G = np.linspace(0.05, 3.2, 2000)

# Physically-motivated S-shape: near-zero flow -> near-zero pressure drop
# (no motion, no friction); rises as velocity increases (normal pipe
# behaviour); a LOCAL MAXIMUM where the vapor-content effect starts to
# dominate; a NEGATIVE-SLOPE region as rising G sharply cuts vapor
# production (less residence time to boil); a LOCAL MINIMUM once quality
# is low enough; rises again at high G as ordinary liquid-phase friction
# takes back over. A cubic captures exactly this rise-fall-rise shape.
def channel_curve(G):
    x = G - 1.5
    return x**3 - 1.2 * x + 1.5

dP = channel_curve(G)

# Flat external/pump characteristic
pump_level = 1.4
pump = np.full_like(G, pump_level)

# find the three intersections numerically
diff = dP - pump
sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
intersections = []
for idx in sign_changes:
    # linear interpolation for the crossing G
    g0, g1 = G[idx], G[idx + 1]
    d0, d1 = diff[idx], diff[idx + 1]
    gc = g0 - d0 * (g1 - g0) / (d1 - d0)
    intersections.append(gc)

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(G, dP, color="#1a3c6e", lw=2.4, label="Channel (internal) characteristic")
ax.plot(G, pump, color="#b5451b", lw=2.2, ls="--", label="External / pump characteristic")

labels = ["A\n(stable,\nvapor-rich)", "B\n(unstable)", "C\n(stable,\nliquid-rich)"]
offsets = [(0.0, -0.42), (0.0, -0.42), (0.0, -0.42)]
for gc, lab, (dx, dy) in zip(intersections, labels, offsets):
    yc = pump_level
    ax.plot(gc, yc, "o", color="#111111", ms=8, zorder=5)
    ax.annotate(lab, (gc, yc), xytext=(gc + dx, yc + dy), fontsize=10.5,
                ha="center", va="top")

ax.set_xlabel("Flow rate,  $G$", labelpad=10)
ax.set_ylabel("Pressure drop,  $\\Delta P$", labelpad=14)
ax.set_xlim(0.0, 3.25)
ax.set_ylim(-0.75, 2.75)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="lower right", frameon=False, fontsize=10.5)

# annotate the negative-slope region
ax.annotate("", xy=(1.9, 1.15), xytext=(1.05, 1.92),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3,
                             connectionstyle="arc3,rad=-0.3"))
ax.text(1.3, 2.35, "negative-slope region\n(more flow $\\Rightarrow$ less $\\Delta P$)",
        fontsize=9.5, color="#444444", ha="center")

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig1_scurve_schematic.png")
fig.savefig(out, dpi=300)
print("saved", out, "intersections at G =", intersections)
