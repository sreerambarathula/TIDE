"""Panel (d): Cross-Domain Multi-Physics Universality Map
Extends metric decoupling and boundary-weighted remediation across 4 major engineering safety domains.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
SLATE = "#334155"

fig, ax = plt.subplots(figsize=(7.8, 6.8), dpi=300)
ax.set_title(r"$\mathbf{(d)}$ Cross-Domain Multi-Physics Universality Map", pad=14, loc="left", fontweight="bold")
ax.axis("off")

domains = [
    {
        "title": "1. Two-Phase Thermal-Hydraulics",
        "system": "Nuclear BWRs, Industrial Boilers, GPU Cooling",
        "phenom": "Ledinegg Excursion & Density Wave Oscillations",
        "singularity": "Fold-Hopf / Bogdanov-Takens Tangency",
        "color": NAVY,
        "bg": "#eff6ff",
        "rect": (0.02, 0.53, 0.46, 0.43)
    },
    {
        "title": "2. Aeroelastic Dynamics",
        "system": "Transonic Wings, Turbine Blades",
        "phenom": "Aeroelastic Flutter & Limit Cycles",
        "singularity": "Shock-Induced Hopf Bifurcations",
        "color": TEAL,
        "bg": "#f0fdfa",
        "rect": (0.52, 0.53, 0.46, 0.43)
    },
    {
        "title": "3. Chemical Process Safety",
        "system": "Exothermic Stirred Reactors (CSTR)",
        "phenom": "Thermal Runaway & Explosive Yield",
        "singularity": "Saddle-Node & Cusp Ignition Extinction",
        "color": CRIMSON,
        "bg": "#fff1f2",
        "rect": (0.02, 0.05, 0.46, 0.43)
    },
    {
        "title": "4. Power Grid Dynamic Stability",
        "system": "High-Voltage Renewable Grids",
        "phenom": "Voltage Collapse & Dynamic Instability",
        "singularity": "Saddle-Node / Hopf Manifolds",
        "color": GOLD,
        "bg": "#fefce8",
        "rect": (0.52, 0.05, 0.46, 0.43)
    }
]

for d in domains:
    x, y, w, h = d["rect"]
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=d["bg"], edgecolor=d["color"], linewidth=1.5)
    ax.add_patch(rect)
    
    ax.text(x + 0.02, y + h - 0.09, d["title"], fontsize=8.8, color=d["color"], fontweight="bold")
    ax.text(x + 0.02, y + h - 0.19, r"$\mathbf{Application:}$ " + d["system"], fontsize=7.6, color=SLATE)
    ax.text(x + 0.02, y + h - 0.29, r"$\mathbf{Instability:}$ " + d["phenom"], fontsize=7.6, color=SLATE)
    ax.text(x + 0.02, y + h - 0.38, r"$\mathbf{Bifurcation:}$ " + d["singularity"], fontsize=7.6, color=d["color"], fontweight="bold")

out_file = os.path.join(output_dir, "panel_d_multiphysics_universality.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (d) to:", out_file)
