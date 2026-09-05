"""Panel (b): Industrial Digital Twin Licensing Decision Tree
Regulatory flowchart for evaluating and certifying machine learning surrogates in safety systems.
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
ax.set_title(r"$\mathbf{(b)}$ Industrial AI Digital Twin Licensing Decision Tree", pad=14, loc="left", fontweight="bold")
ax.axis("off")

nodes = [
    {
        "id": "start",
        "text": "1. Surrogate Training Completed\n(Evaluate Initial Model)",
        "rect": (0.28, 0.86, 0.44, 0.11),
        "color": NAVY,
        "bg": "#eff6ff"
    },
    {
        "id": "q1",
        "text": "2. Codimension-2 Tangency Present?\n(Check BT / Sharp Cusp Geometry)",
        "rect": (0.28, 0.68, 0.44, 0.11),
        "color": GOLD,
        "bg": "#fefce8"
    },
    {
        "id": "remedy",
        "text": "Apply Boundary-Weighted Loss\n+ Multi-Scale Fourier Encoding",
        "rect": (0.02, 0.48, 0.42, 0.11),
        "color": TEAL,
        "bg": "#f0fdfa"
    },
    {
        "id": "standard",
        "text": "Standard Uniform Validation\n(Transversal Boundaries Only)",
        "rect": (0.56, 0.48, 0.42, 0.11),
        "color": SLATE,
        "bg": "#f8fafc"
    },
    {
        "id": "cert",
        "text": "3. Localized Near-Boundary Audit\n($\\mathrm{RMSE}_{\\mathrm{near}} \\leq \\varepsilon_{\\mathrm{cert}}$, False-Stable $\\leq 0.1\\%$)",
        "rect": (0.28, 0.28, 0.44, 0.11),
        "color": NAVY,
        "bg": "#eff6ff"
    },
    {
        "id": "pass",
        "text": "LICENSED & CERTIFIED\n(Safe Industrial Deployment)",
        "rect": (0.56, 0.08, 0.42, 0.11),
        "color": "#16a34a",
        "bg": "#f0fdf4"
    },
    {
        "id": "fail",
        "text": "LICENSE REJECTED\n(Unsafe Cusp Breakdown)",
        "rect": (0.02, 0.08, 0.42, 0.11),
        "color": CRIMSON,
        "bg": "#fff1f2"
    }
]

for node in nodes:
    x, y, w, h = node["rect"]
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                  facecolor=node["bg"], edgecolor=node["color"], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, node["text"], fontsize=8.4, color=node["color"], fontweight="bold",
            ha="center", va="center", linespacing=1.2)

# Flow Arrows
def draw_arrow(x1, y1, x2, y2, text="", color=SLATE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if text:
        ax.text((x1+x2)/2 + 0.02, (y1+y2)/2, text, fontsize=8.0, fontweight="bold", color=color)

draw_arrow(0.50, 0.86, 0.50, 0.79)
draw_arrow(0.38, 0.68, 0.23, 0.59, text="YES (Cusp)", color=CRIMSON)
draw_arrow(0.62, 0.68, 0.77, 0.59, text="NO (Transv)", color=TEAL)

draw_arrow(0.23, 0.48, 0.40, 0.39)
draw_arrow(0.77, 0.48, 0.60, 0.39)

draw_arrow(0.38, 0.28, 0.23, 0.19, text="FAIL", color=CRIMSON)
draw_arrow(0.62, 0.28, 0.77, 0.19, text="PASS", color="#16a34a")

out_file = os.path.join(output_dir, "panel_b_licensing_decision_tree.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (b) to:", out_file)
