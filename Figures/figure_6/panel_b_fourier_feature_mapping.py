"""Panel (b): Random Fourier Feature Harmonic Mapping
Visualizes multi-scale sinusoidal projection basis vectors [sin(2*pi*B*x), cos(2*pi*B*x)] across coordinate space.
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

np.random.seed(42)
x_norm = np.linspace(-1.0, 1.0, 300)

# Sample 4 random Fourier basis modes from B ~ N(0, sigma^2) with sigma = 3.5
b_modes = np.array([0.8, 2.2, 4.5, 7.8])
colors = [NAVY, TEAL, GOLD, CRIMSON]

fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=300)
ax.set_title(r"$\mathbf{(b)}$ Multi-Scale Random Fourier Basis Functions", pad=14, loc="left", fontweight="bold")

for i, (b, col) in enumerate(zip(b_modes, colors)):
    basis_sin = np.sin(2 * np.pi * b * x_norm)
    ax.plot(x_norm, basis_sin, color=col, lw=2.0, alpha=0.9, 
            label=f"Basis Mode {i+1} ($B_{{{i+1}}} = {b:.1f}$)")

ax.annotate(r"$\mathbf{Harmonic \; Basis \; Embedding:}$" + "\n" + 
            r"$\gamma(\mathbf{x}) = [\sin(2\pi\mathbf{Bx}), \cos(2\pi\mathbf{Bx})]$" + "\n" + 
            r"spans multiple spatial octaves to capture" + "\n" + 
            r"sub-millimeter boundary cusps",
            xy=(0.25, 0.95), xytext=(-0.85, 0.45),
            fontsize=8.8, color=TEAL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f0fdfa", ec=TEAL, lw=1.2),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))

ax.set_xlabel(r"Standardized Parameter Coordinate, $x_{\mathrm{norm}} \in [-1, 1]$")
ax.set_ylabel(r"Harmonic Feature Amplitude, $\sin(2\pi B x)$")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.25, 1.35)
ax.grid(True)
ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0")

out_file = os.path.join(output_dir, "panel_b_fourier_feature_mapping.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (b) to:", out_file)
