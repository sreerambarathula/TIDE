"""Panel (a): Physical Channel Architecture & Instability Mechanisms
Compact 3-Tube Comparative Schematic (Base Model, Ledinegg Excursion, Dynamic DWO).
Target width: ~7.5 - 8.5 cm single column. Large, bold, non-overlapping typography.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Publication Typography - Bold & Legible at 7-8 cm width
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
})

NAVY = "#1a3c6e"
CRIMSON = "#b5451b"
TEAL = "#007a78"
SLATE = "#1e293b"
LIGHT_BLUE = "#bfdbfe"
LIGHT_ORANGE = "#fed7aa"
DRYOUT_HOT = "#dc2626"
DRYOUT_BG = "#fee2e2"
BUBBLE_EC = "#ea580c"
BUBBLE_FC = "#ffedd5"

fig, ax = plt.subplots(figsize=(6.0, 4.3), dpi=300)
ax.set_xlim(-0.4, 6.4)
ax.set_ylim(-0.35, 4.5)
ax.axis("off")

# Title
ax.set_title(r"$\mathbf{(a)}$ Boiling Physics & Instability Mechanisms", fontsize=12, fontweight="bold", pad=12, loc="left")

# Geometry coordinates
tube_w = 1.05
tube_bottom = 0.50
tube_top = 3.60
tube_h = tube_top - tube_bottom

# Common Plenums
ax.add_patch(FancyBboxPatch((-0.1, tube_top), 6.15, 0.40, boxstyle="round,pad=0.02", fc="#e2e8f0", ec=NAVY, lw=1.3))
ax.text(2.975, tube_top + 0.20, r"Upper Plenum ($P_{\mathrm{out}}$)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY)

ax.add_patch(FancyBboxPatch((-0.1, tube_bottom - 0.40), 6.15, 0.40, boxstyle="round,pad=0.02", fc="#e2e8f0", ec=NAVY, lw=1.3))
ax.text(2.975, tube_bottom - 0.20, r"Lower Plenum ($P_{\mathrm{in}}$)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY)


# ==============================================================================
# TUBE 1: BASE MOVING-BOUNDARY MODEL
# ==============================================================================
x1 = 0.35
ax.text(x1 + tube_w/2, tube_top + 0.55, "Base Channel", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NAVY)

# Channel walls
ax.plot([x1, x1], [tube_bottom, tube_top], color=SLATE, lw=2.4)
ax.plot([x1 + tube_w, x1 + tube_w], [tube_bottom, tube_top], color=SLATE, lw=2.4)

# 1-phase liquid
z_lam = tube_bottom + 1.35
ax.add_patch(Rectangle((x1, tube_bottom), tube_w, z_lam - tube_bottom, fc=LIGHT_BLUE, ec="none", alpha=0.9))
ax.text(x1 + tube_w/2, (tube_bottom + z_lam)/2 - 0.05, r"$\mathbf{1\phi}$" + " " + r"$\mathbf{Liquid}$" + "\n" + r"$\boldsymbol{\rho = \rho_f}$", 
        ha="center", va="center", fontsize=9.5, color="#1e40af")

# Moving boundary line
ax.plot([x1 - 0.15, x1 + tube_w + 0.15], [z_lam, z_lam], color=CRIMSON, lw=2.0, ls="--")
ax.text(x1 + tube_w/2, z_lam, r"$\mathbf{z = \lambda(t)}$", ha="center", va="center", fontsize=9.2, color=CRIMSON,
        bbox=dict(boxstyle="round,pad=0.12", fc="#ffffff", ec=CRIMSON, lw=0.8))

# 2-phase mixture
ax.add_patch(Rectangle((x1, z_lam), tube_w, tube_top - z_lam, fc=LIGHT_ORANGE, ec="none", alpha=0.9))
ax.text(x1 + tube_w/2, tube_top - 0.65, r"$\mathbf{2\phi}$" + " " + r"$\mathbf{Mix}$" + "\n" + r"$\boldsymbol{\rho_m \ll \rho_f}$", 
        ha="center", va="center", fontsize=9.5, color="#9a3412")

# Deterministic bubble positions
bubble_coords = [
    (x1 + 0.25, z_lam + 0.22, 0.065),
    (x1 + 0.78, z_lam + 0.28, 0.075),
    (x1 + 0.45, z_lam + 0.42, 0.060),
    (x1 + 0.82, z_lam + 0.60, 0.070),
    (x1 + 0.22, z_lam + 0.68, 0.065)
]
for bx, by, br in bubble_coords:
    ax.add_patch(Circle((bx, by), br, fc=BUBBLE_FC, ec=BUBBLE_EC, lw=0.6))

# Heat flux arrows
for y_pos in [1.1, 2.0, 2.9]:
    ax.annotate("", xy=(x1 - 0.02, y_pos), xytext=(x1 - 0.26, y_pos), arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3))
ax.text(x1 - 0.35, 2.0, r"$\mathbf{q''}$", ha="center", va="center", fontsize=11.5, color=CRIMSON)

# Velocity arrows
ax.annotate("", xy=(x1 + tube_w/2 + 0.15, tube_bottom + 0.35), xytext=(x1 + tube_w/2 + 0.15, tube_bottom + 0.05), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
ax.text(x1 + tube_w/2 - 0.10, tube_bottom + 0.20, r"$\mathbf{u_i}$", ha="right", va="center", fontsize=10.0, color=NAVY)

ax.annotate("", xy=(x1 + tube_w/2 + 0.15, tube_top - 0.05), xytext=(x1 + tube_w/2 + 0.15, tube_top - 0.35), arrowprops=dict(arrowstyle="->", color="#9a3412", lw=2.0))
ax.text(x1 + tube_w/2 - 0.10, tube_top - 0.20, r"$\mathbf{u_e}$", ha="right", va="center", fontsize=10.0, color="#9a3412")


# ==============================================================================
# TUBE 2: LEDINEGG (STATIC / EXCURSIVE) INSTABILITY
# ==============================================================================
x2 = 2.40
ax.text(x2 + tube_w/2, tube_top + 0.55, "Ledinegg Mode", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#c2410c")

# Channel walls
ax.plot([x2, x2], [tube_bottom, tube_top], color=SLATE, lw=2.4)
ax.plot([x2 + tube_w, x2 + tube_w], [tube_bottom, tube_top], color=SLATE, lw=2.4)

# Collapsed boiling boundary
z_lam_led = tube_bottom + 0.52
ax.add_patch(Rectangle((x2, tube_bottom), tube_w, z_lam_led - tube_bottom, fc=LIGHT_BLUE, ec="none", alpha=0.9))
ax.add_patch(Rectangle((x2, z_lam_led), tube_w, tube_top - z_lam_led, fc=DRYOUT_BG, ec="none", alpha=0.95))
ax.plot([x2 - 0.12, x2 + tube_w + 0.12], [z_lam_led, z_lam_led], color=CRIMSON, lw=1.8, ls="--")
ax.text(x2 + tube_w/2, z_lam_led, r"$\boldsymbol{\lambda \to 0}$", ha="center", va="center", fontsize=8.8, color=CRIMSON,
        bbox=dict(boxstyle="round,pad=0.1", fc="#ffffff", ec=CRIMSON, lw=0.7))

# Vapor lock box
ax.add_patch(FancyBboxPatch((x2 + 0.12, tube_bottom + 0.80), tube_w - 0.24, 1.80, boxstyle="round,pad=0.03", fc="#fed7aa", ec=BUBBLE_EC, lw=1.0))
ax.text(x2 + tube_w/2, tube_bottom + 1.85, r"$\mathbf{Vapor}$" + "\n" + r"$\mathbf{Lock}$", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#9a3412")

# Hot dryout wall (highlighted on right wall)
ax.plot([x2 + tube_w, x2 + tube_w], [tube_bottom + 0.75, tube_top - 0.25], color=DRYOUT_HOT, lw=4.5)
ax.text(x2 + tube_w/2, tube_bottom + 1.12, r"$\mathbf{CHF / Dryout}$" + "\n" + r"$\mathbf{(T_w \uparrow\uparrow)}$", 
        ha="center", va="center", fontsize=8.2, color=DRYOUT_HOT,
        bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec=DRYOUT_HOT, lw=0.7))

# Starvation flow
ax.annotate("", xy=(x2 + tube_w/2 + 0.28, tube_bottom + 0.35), xytext=(x2 + tube_w/2 + 0.28, tube_bottom + 0.05), arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.5))
ax.text(x2 + tube_w/2 - 0.05, tube_bottom + 0.20, r"$\mathbf{u_i \downarrow\downarrow}$", ha="right", va="center", fontsize=9.5, color=CRIMSON)


# ==============================================================================
# TUBE 3: DENSITY-WAVE OSCILLATION (DYNAMIC DWO)
# ==============================================================================
x3 = 4.45
ax.text(x3 + tube_w/2, tube_top + 0.55, "Dynamic DWO", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=TEAL)

# Channel walls
ax.plot([x3, x3], [tube_bottom, tube_top], color=SLATE, lw=2.4)
ax.plot([x3 + tube_w, x3 + tube_w], [tube_bottom, tube_top], color=SLATE, lw=2.4)

# Traveling wave packets
h_pack = tube_h / 5.0
for i in range(5):
    y_i = tube_bottom + i * h_pack
    if i % 2 == 0:
        ax.add_patch(Rectangle((x3, y_i), tube_w, h_pack, fc=LIGHT_BLUE, ec="none", alpha=0.9))
        ax.text(x3 + tube_w/2, y_i + h_pack/2, r"$\boldsymbol{\rho \uparrow}$", ha="center", va="center", fontsize=11.0, color="#1e40af")
    else:
        ax.add_patch(Rectangle((x3, y_i), tube_w, h_pack, fc=LIGHT_ORANGE, ec="none", alpha=0.9))
        ax.text(x3 + tube_w/2, y_i + h_pack/2, r"$\boldsymbol{\rho \downarrow}$", ha="center", va="center", fontsize=11.0, color="#9a3412")

# Transit delay bracket on right
ax.annotate("", xy=(x3 + tube_w + 0.12, tube_top - 0.05), xytext=(x3 + tube_w + 0.12, tube_bottom + 0.05), arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.6))
ax.text(x3 + tube_w + 0.22, (tube_bottom + tube_top)/2, r"$\boldsymbol{\tau}_{\mathrm{transit}}$", ha="left", va="center", fontsize=11.5, color=TEAL)

# Dynamic 180 deg feedback loop (exit to inlet on left of Tube 3)
feed_arr = FancyArrowPatch((x3 - 0.06, tube_top - 0.15), (x3 - 0.06, tube_bottom + 0.15),
                           connectionstyle="arc3,rad=0.36", color=CRIMSON, lw=1.6,
                           arrowstyle="-|>", mutation_scale=11)
ax.add_patch(feed_arr)
ax.text(x3 - 0.32, (tube_bottom + tube_top)/2, r"$\mathbf{180^\circ}$" + "\n" + r"$\mathbf{Feedback}$", ha="center", va="center", fontsize=8.6, color=CRIMSON)

plt.tight_layout()
out_file = os.path.join(output_dir, "panel_a_zonation.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Refined 3-Tube Panel (a) to:", out_file)
