"""Panel (d): The Safety-Critical Pareto Frontier
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
    "legend.fontsize": 9.4,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.40,
    "mathtext.fontset": "stixsans",
})

NAVY = "#1a3c6e"
CRIMSON = "#b5451b"
TEAL = "#007a78"
GOLD = "#c88a10"
ORANGE = "#ea580c"
AMBER = "#d97706"
SLATE = "#64748b"

# Models data: (Far RMSE, Near RMSE, Color, Marker)
models = {
    "Baseline (2×64)": (0.00766, 0.02316, CRIMSON, "o"),
    "Capacity (3×128)": (0.00691, 0.01501, ORANGE, "s"),
    "Capacity (4×128)": (0.01508, 0.01851, AMBER, "D"),
    "Capacity (3×256)": (0.01226, 0.02115, "#b45309", "^"),
    "Fourier MLP (3×128)": (0.00620, 0.01320, TEAL, "v"),
    "Log-Distance Feature": (0.00650, 0.01410, GOLD, "p"),
    "Boundary-Weighted (3×128)": (0.00534, 0.01200, NAVY, "*"),
}

fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=300)
ax.set_title(r"$\mathbf{(d)}$ The Safety-Critical Pareto Frontier", pad=14, loc="left", fontweight="bold")

# Shaded Uncertified Zone (Near RMSE > 0.0120)
ax.fill_between([0.0035, 0.018], [0.0120, 0.0120], [0.026, 0.026],
                color="#fee2e2", alpha=0.45, label="Uncertified Zone (RMSE > 0.012)")

# Empirical Pareto Frontier curve connecting non-dominated solutions
pareto_far = [0.00534, 0.00620, 0.00691, 0.00766]
pareto_near = [0.01200, 0.01320, 0.01501, 0.02316]
ax.plot(pareto_far, pareto_near, color=NAVY, lw=1.8, ls="--", alpha=0.85, label="Pareto Frontier")

# Plot each model point
for name, (far, near, col, marker) in models.items():
    msize = 16 if marker == "*" else 9.5
    ax.plot(far, near, marker=marker, color=col, ms=msize, mew=1.2, mec="#0f172a", ls="none", label=name, zorder=6)

# Threshold reference lines
ax.axhline(0.0120, color="#64748b", ls=":", lw=1.3, alpha=0.8)

# Minimalist Pareto Optimum Callout
ax.annotate(
    r"$\mathbf{Pareto\;Optimum:}$" + "\n" +
    r"$\mathrm{RMSE}_{\mathrm{near}} = \mathbf{0.0120}$" + "\n" +
    r"$\mathrm{RMSE}_{\mathrm{far}} = \mathbf{0.0053}$",
    xy=(0.00534, 0.01200), xytext=(0.0078, 0.0078),
    fontsize=9.6, color=NAVY, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.40", fc="#eff6ff", ec=NAVY, lw=1.2),
    arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4, connectionstyle="arc3,rad=0.10")
)

# Minimal Operating Conditions Card (Point B)
card_text = (
    r"$\mathbf{Point\;B:}\;Fr = 0.50,\;\Lambda = 0.001$" + "\n" +
    r"$k_{\mathrm{in}} = 11.0,\;k_{\mathrm{ex}} = 3.0$"
)
ax.text(
    0.03, 0.95, card_text,
    transform=ax.transAxes,
    fontsize=9.4,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.94, lw=1.0)
)

ax.set_xlabel(r"Far-Field Generalization Error, $\mathrm{RMSE}_{\mathrm{far}}$")
ax.set_ylabel(r"Near-Boundary Safety Error, $\mathrm{RMSE}_{\mathrm{near}}$")
ax.set_xlim(0.0035, 0.0170)
ax.set_ylim(0.0050, 0.0275)
ax.grid(True, linestyle="--", alpha=0.45)

ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white",
          edgecolor="#cbd5e1", fontsize=8.8, borderpad=0.50, labelspacing=0.40)

plt.tight_layout()

out_file = os.path.join(output_dir, "panel_d_safety_pareto_frontier.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (d) to:", out_file)

# Copy to brain artifact directory for display
brain_dir = "C:/Users/User/.gemini/antigravity/brain/9d59c56d-97c0-4103-bb80-3739087d8e26"
brain_file = os.path.join(brain_dir, "panel_d_safety_pareto_frontier.png")
shutil.copyfile(out_file, brain_file)
print("Copied to brain directory:", brain_file)
