"""Master Figure 8: Reliability Certification Framework, Industrial Licensing, and Multi-Physics Universality
Assembles all 4 panels in a 2-column by 2-row publication-grade layout (300 DPI).
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)
os.makedirs("d:/AGravity/Tide_Tutor/manuscript/figures", exist_ok=True)

# Publication Typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.6,
    "axes.linewidth": 1.1,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "mathtext.fontset": "cm",
})

NAVY = "#1a3c6e"
CRIMSON = "#b5451b"
TEAL = "#007a78"
GOLD = "#c88a10"
SLATE = "#334155"

def draw_panel_a(ax):
    ax.set_title(r"$\mathbf{(a)}$ The Four Golden Rules for Safety-Critical Surrogates", pad=12, loc="left", fontweight="bold")
    ax.axis("off")
    rules = [
        {
            "num": "RULE 1",
            "title": r"Prohibition of Global $R^2$ as Standalone Metric",
            "desc": r"Global $R^2 > 0.999$ masks near-boundary breakdown ($R^2 < 0$)." + "\n" +
                    r"Safety claims MUST never rely solely on global aggregate metrics.",
            "color": CRIMSON,
            "bg": "#fff1f2",
            "rect": (0.01, 0.75, 0.98, 0.22)
        },
        {
            "num": "RULE 2",
            "title": r"Continuation Pre-Screening for Singularities",
            "desc": r"Pre-screen operating envelopes via pseudo-arclength continuation" + "\n" +
                    r"to identify Bogdanov-Takens tangencies and acute boundary cusps.",
            "color": GOLD,
            "bg": "#fefce8",
            "rect": (0.01, 0.51, 0.98, 0.22)
        },
        {
            "num": "RULE 3",
            "title": r"Mandatory Boundary-Weighted Loss Optimization",
            "desc": r"Standard MSE optimization produces severe spectral cusp rounding." + "\n" +
                    r"Training MUST employ inverse-distance weighting $\mathcal{L}_{\mathrm{BW}} = \frac{1}{|g| + \varepsilon_w} (\hat{g}-g)^2$.",
            "color": TEAL,
            "bg": "#f0fdfa",
            "rect": (0.01, 0.27, 0.98, 0.22)
        },
        {
            "num": "RULE 4",
            "title": r"Absolute Near-Boundary Local Certification",
            "desc": r"Surrogates must undergo localized validation strictly within" + "\n" +
                    r"$|g(\mathbf{x})| \leq \delta_{\mathrm{cert}}$, guaranteeing false-stable error rates $\leq 0.1\%$.",
            "color": NAVY,
            "bg": "#eff6ff",
            "rect": (0.01, 0.03, 0.98, 0.22)
        }
    ]
    for rule in rules:
        x, y, w, h = rule["rect"]
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                      facecolor=rule["bg"], edgecolor=rule["color"], linewidth=1.3)
        ax.add_patch(rect)
        badge = patches.FancyBboxPatch((x + 0.02, y + h - 0.08), 0.16, 0.06, boxstyle="round,pad=0.01,rounding_size=0.02",
                                      facecolor=rule["color"], edgecolor="none")
        ax.add_patch(badge)
        ax.text(x + 0.10, y + h - 0.05, rule["num"], fontsize=8.0, color="white", fontweight="bold", ha="center", va="center")
        ax.text(x + 0.20, y + h - 0.05, rule["title"], fontsize=8.6, color=rule["color"], fontweight="bold", va="center")
        ax.text(x + 0.03, y + 0.055, rule["desc"], fontsize=7.8, color=SLATE, linespacing=1.3)

def draw_panel_b(ax):
    ax.set_title(r"$\mathbf{(b)}$ Industrial AI Digital Twin Licensing Decision Tree", pad=12, loc="left", fontweight="bold")
    ax.axis("off")
    nodes = [
        {
            "text": "1. Surrogate Training Completed\n(Evaluate Initial Model)",
            "rect": (0.28, 0.86, 0.44, 0.11),
            "color": NAVY,
            "bg": "#eff6ff"
        },
        {
            "text": "2. Codimension-2 Tangency?\n(Check BT / Sharp Cusp Geometry)",
            "rect": (0.28, 0.68, 0.44, 0.11),
            "color": GOLD,
            "bg": "#fefce8"
        },
        {
            "text": "Apply Boundary-Weighted Loss\n+ Multi-Scale Fourier Encoding",
            "rect": (0.02, 0.48, 0.42, 0.11),
            "color": TEAL,
            "bg": "#f0fdfa"
        },
        {
            "text": "Standard Uniform Validation\n(Transversal Boundaries Only)",
            "rect": (0.56, 0.48, 0.42, 0.11),
            "color": SLATE,
            "bg": "#f8fafc"
        },
        {
            "text": "3. Localized Near-Boundary Audit\n($\\mathrm{RMSE}_{\\mathrm{near}} \\leq \\varepsilon_{\\mathrm{cert}}$, False-Stable $\\leq 0.1\\%$)",
            "rect": (0.28, 0.28, 0.44, 0.11),
            "color": NAVY,
            "bg": "#eff6ff"
        },
        {
            "text": "LICENSED & CERTIFIED\n(Safe Industrial Deployment)",
            "rect": (0.56, 0.08, 0.42, 0.11),
            "color": "#16a34a",
            "bg": "#f0fdf4"
        },
        {
            "text": "LICENSE REJECTED\n(Unsafe Cusp Breakdown)",
            "rect": (0.02, 0.08, 0.42, 0.11),
            "color": CRIMSON,
            "bg": "#fff1f2"
        }
    ]
    for node in nodes:
        x, y, w, h = node["rect"]
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                      facecolor=node["bg"], edgecolor=node["color"], linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, node["text"], fontsize=7.8, color=node["color"], fontweight="bold",
                ha="center", va="center", linespacing=1.2)

    def draw_arrow(x1, y1, x2, y2, text="", color=SLATE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3))
        if text:
            ax.text((x1+x2)/2 + 0.02, (y1+y2)/2, text, fontsize=7.6, fontweight="bold", color=color)

    draw_arrow(0.50, 0.86, 0.50, 0.79)
    draw_arrow(0.38, 0.68, 0.23, 0.59, text="YES (Cusp)", color=CRIMSON)
    draw_arrow(0.62, 0.68, 0.77, 0.59, text="NO (Transv)", color=TEAL)
    draw_arrow(0.23, 0.48, 0.40, 0.39)
    draw_arrow(0.77, 0.48, 0.60, 0.39)
    draw_arrow(0.38, 0.28, 0.23, 0.19, text="FAIL", color=CRIMSON)
    draw_arrow(0.62, 0.28, 0.77, 0.19, text="PASS", color="#16a34a")

def draw_panel_c(ax):
    ax.set_title(r"$\mathbf{(c)}$ Scale-Up Invariance: 1D Channel vs. Natural Loop", pad=12, loc="left", fontweight="bold")
    ns1 = np.linspace(10.5, 14.14, 200)
    d1 = 14.14 - ns1
    fold1 = 20.59 - 1.48 * d1
    hopf1 = 20.59 - 0.42 * d1

    ns2 = np.linspace(8.0, 12.80, 200)
    d2 = 12.80 - ns2
    fold2 = 18.20 - 1.62 * d2
    hopf2 = 18.20 - 0.50 * d2

    ax.plot(ns1, fold1, color=NAVY, lw=2.0, ls="-", label=r"1D Fold ($\Lambda = 0.001$)")
    ax.plot(ns1, hopf1, color=TEAL, lw=2.0, ls="-", label=r"1D Hopf ($\Lambda = 0.001$)")
    ax.plot([14.14], [20.59], marker="*", color=GOLD, ms=12, zorder=6, label=r"1D BT Singularity ($\star$)")

    ax.plot(ns2, fold2, color=CRIMSON, lw=2.0, ls="--", label=r"Loop Fold ($\Lambda = 4.5$)")
    ax.plot(ns2, hopf2, color="#ea580c", lw=2.0, ls="--", label=r"Loop Hopf ($\Lambda = 4.5$)")
    ax.plot([12.80], [18.20], marker="^", color=CRIMSON, ms=9, zorder=6, label=r"Loop BT Singularity ($\blacktriangle$)")

    ax.fill_between(ns1, fold1, hopf1, color="#e0f2fe", alpha=0.35)
    ax.fill_between(ns2, fold2, hopf2, color="#fee2e2", alpha=0.25)

    ax.annotate(r"$\mathbf{Topological \; Invariance:}$" + "\n" + 
                r"BT tangency & knife-edge cusp" + "\n" + 
                r"persist in full loops ($\Lambda \approx 3\text{--}6$)",
                xy=(12.80, 18.20), xytext=(8.2, 19.8),
                fontsize=8.0, color=NAVY, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec=NAVY, lw=1.1),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2))

    ax.set_xlabel(r"Subcooling Number, $N_{\mathrm{sub}}$")
    ax.set_ylabel(r"Phase Change Number, $N_{\mathrm{pch}}$")
    ax.set_xlim(7.5, 14.8)
    ax.set_ylim(11.0, 22.0)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=7.6)

def draw_panel_d(ax):
    ax.set_title(r"$\mathbf{(d)}$ Cross-Domain Multi-Physics Universality Map", pad=12, loc="left", fontweight="bold")
    ax.axis("off")
    domains = [
        {
            "title": "1. Two-Phase Thermal-Hydraulics",
            "system": "Nuclear BWRs, Boilers, GPU Cooling",
            "phenom": "Ledinegg Excursion & DWO Oscillations",
            "singularity": "Fold-Hopf / BT Tangency",
            "color": NAVY,
            "bg": "#eff6ff",
            "rect": (0.01, 0.53, 0.47, 0.43)
        },
        {
            "title": "2. Aeroelastic Dynamics",
            "system": "Transonic Wings, Turbine Blades",
            "phenom": "Aeroelastic Flutter & Limit Cycles",
            "singularity": "Shock-Induced Hopf Bifurcations",
            "color": TEAL,
            "bg": "#f0fdfa",
            "rect": (0.52, 0.53, 0.47, 0.43)
        },
        {
            "title": "3. Chemical Process Safety",
            "system": "Exothermic Stirred Reactors (CSTR)",
            "phenom": "Thermal Runaway & Explosive Yield",
            "singularity": "Saddle-Node & Cusp Ignition Extinction",
            "color": CRIMSON,
            "bg": "#fff1f2",
            "rect": (0.01, 0.05, 0.47, 0.43)
        },
        {
            "title": "4. Power Grid Stability",
            "system": "High-Voltage Renewable Grids",
            "phenom": "Voltage Collapse & Dynamic Instability",
            "singularity": "Saddle-Node / Hopf Manifolds",
            "color": GOLD,
            "bg": "#fefce8",
            "rect": (0.52, 0.05, 0.47, 0.43)
        }
    ]
    for d in domains:
        x, y, w, h = d["rect"]
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                      facecolor=d["bg"], edgecolor=d["color"], linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.09, d["title"], fontsize=8.4, color=d["color"], fontweight="bold")
        ax.text(x + 0.02, y + h - 0.19, r"$\mathbf{Application:}$ " + d["system"], fontsize=7.4, color=SLATE)
        ax.text(x + 0.02, y + h - 0.29, r"$\mathbf{Instability:}$ " + d["phenom"], fontsize=7.4, color=SLATE)
        ax.text(x + 0.02, y + h - 0.38, r"$\mathbf{Bifurcation:}$ " + d["singularity"], fontsize=7.4, color=d["color"], fontweight="bold")

# Assemble Master Multi-Panel Figure (2 columns x 2 rows)
fig = plt.figure(figsize=(15.5, 13.5), dpi=300)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.22)

draw_panel_a(fig.add_subplot(gs[0, 0]))
draw_panel_b(fig.add_subplot(gs[0, 1]))
draw_panel_c(fig.add_subplot(gs[1, 0]))
draw_panel_d(fig.add_subplot(gs[1, 1]))

out_png_local = os.path.join(output_dir, "fig8_reliability_certification_master.png")
out_pdf_local = os.path.join(output_dir, "fig8_reliability_certification_master.pdf")
out_png_ms = "d:/AGravity/Tide_Tutor/manuscript/figures/fig8_reliability_certification.png"

fig.savefig(out_png_local, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf_local, bbox_inches="tight")
fig.savefig(out_png_ms, dpi=300, bbox_inches="tight")

print("Generated Master Figure 8 in:")
print("  -", out_png_local)
print("  -", out_pdf_local)
print("  -", out_png_ms)
