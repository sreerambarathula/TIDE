"""Panel (e): Inverse-Distance Boundary Loss Weighting Function w(g)
Shows analytical weighting curve w(g) = 1 / (|g| + eps_w) across regularization scales eps_w.
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

g_vals = np.linspace(-1.5, 1.5, 400)

epsilons = [0.02, 0.05, 0.10, 0.25]
colors = [CRIMSON, GOLD, TEAL, NAVY]

fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=300)
ax.set_title(r"$\mathbf{(e)}$ Inverse-Distance Boundary Weight Function $w(g) = \frac{1}{|g| + \varepsilon_w}$", pad=14, loc="left", fontweight="bold")

for eps, col in zip(epsilons, colors):
    w = 1.0 / (np.abs(g_vals) + eps)
    ax.plot(g_vals, w, lw=2.2, color=col, label=f"Weight Regularizer $\\varepsilon_w = {eps:.2f}$")

# Highlight optimal epsilon_w = 0.02
ax.annotate(r"$\mathbf{Hyperbolic \; Gradient \; Focus:}$" + "\n" + 
            r"At the boundary ($g \to 0$), weight surges to" + "\n" + 
            r"$w_{\max} = \frac{1}{\varepsilon_w} = \mathbf{50.0}$, amplifying safety gradients" + "\n" + 
            r"by $\mathbf{50\times}$ over far-field points ($g = 1.0 \Rightarrow w \approx 1.0$)",
            xy=(0, 50.0), xytext=(0.25, 32.0),
            fontsize=8.8, color=CRIMSON, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff1f2", ec=CRIMSON, lw=1.2),
            arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3))

ax.set_xlabel(r"Limit-State Margin Value, $g(\mathbf{x})$")
ax.set_ylabel(r"Optimization Loss Weight, $w(g)$")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0, 55)
ax.grid(True)
ax.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0")

out_file = os.path.join(output_dir, "panel_e_boundary_weight_function.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (e) to:", out_file)
