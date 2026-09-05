"""Panel (a): The Four Golden Rules for Safety-Critical Surrogate Deployment
RE&SS Journal Standard - Aptos / Mathtext STIXSans, Sized-Up Font Hierarchy
"""
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Publication Typography (RE&SS Standard >= 11pt, Aptos)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11.0,
    "axes.labelsize": 11.8,
    "axes.titlesize": 12.2,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.6,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.40,
    "mathtext.fontset": "stixsans",
})

NAVY = "#1a3c6e"
CRIMSON = "#b5451b"
TEAL = "#007a78"
AMBER = "#b45309"
SLATE = "#1e293b"

fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=300)
ax.set_title(r"$\mathbf{(a)}$ The Four Golden Rules for Safety-Critical Surrogates",
             pad=14, loc="left", fontweight="bold")
ax.axis("off")

rules = [
    {
        "num": "RULE 1",
        "title": r"Prohibition of Global $R^2$ as Standalone Metric",
        "desc": r"Global metrics ($R^2 > 0.999$) conceal localized near-boundary breakdown ($R^2 < 0$)." + "\n" +
                r"Regulatory safety claims MUST never rely solely on global aggregate metrics.",
        "color": CRIMSON,
        "bg": "#fff1f2",
        "rect": (0.02, 0.745, 0.96, 0.225)
    },
    {
        "num": "RULE 2",
        "title": r"Continuation Pre-Screening for Singularities",
        "desc": r"Identify Bogdanov–Takens tangencies, acute cusps, and codimension-2 bifurcations" + "\n" +
                r"a priori via numerical continuation before training neural network surrogates.",
        "color": AMBER,
        "bg": "#fefce8",
        "rect": (0.02, 0.505, 0.96, 0.225)
    },
    {
        "num": "RULE 3",
        "title": r"Mandatory Boundary-Weighted Loss Optimization",
        "desc": r"Standard MSE optimization produces spectral smoothing and catastrophic cusp truncation." + "\n" +
                r"Surrogates MUST employ inverse-distance weighting: $\mathcal{L}_{\mathrm{BW}} = \frac{1}{|g| + \varepsilon_w} (\hat{g} - g)^2$.",
        "color": TEAL,
        "bg": "#f0fdfa",
        "rect": (0.02, 0.265, 0.96, 0.225)
    },
    {
        "num": "RULE 4",
        "title": r"Absolute Near-Boundary Local Certification",
        "desc": r"Surrogates must undergo localized non-parametric validation within safety band $|g(\mathbf{x})| \leq \delta_{\mathrm{cert}}$," + "\n" +
                r"guaranteeing localized false-stable operating risk $\leq 0.1\%$ across random seeds.",
        "color": NAVY,
        "bg": "#eff6ff",
        "rect": (0.02, 0.025, 0.96, 0.225)
    }
]

for rule in rules:
    x, y, w, h = rule["rect"]
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=rule["bg"], edgecolor=rule["color"], linewidth=1.4
    )
    ax.add_patch(rect)
    
    # Badge Pill
    badge = patches.FancyBboxPatch(
        (x + 0.022, y + h - 0.075), 0.155, 0.054,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        facecolor=rule["color"], edgecolor="none"
    )
    ax.add_patch(badge)
    ax.text(x + 0.099, y + h - 0.048, rule["num"], fontsize=9.2, color="white",
            fontweight="bold", ha="center", va="center")
    
    # Title
    ax.text(x + 0.198, y + h - 0.048, rule["title"], fontsize=10.4, color=rule["color"],
            fontweight="bold", va="center")
    
    # Description
    ax.text(x + 0.035, y + 0.065, rule["desc"], fontsize=9.0, color=SLATE,
            linespacing=1.35, va="center")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()

out_file = os.path.join(output_dir, "panel_a_four_golden_rules.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print("Saved Panel (a) to:", out_file)

# Copy to brain artifact directory for display
brain_dir = "C:/Users/User/.gemini/antigravity/brain/9d59c56d-97c0-4103-bb80-3739087d8e26"
brain_file = os.path.join(brain_dir, "panel_a_four_golden_rules.png")
shutil.copyfile(out_file, brain_file)
print("Copied to brain directory:", brain_file)

