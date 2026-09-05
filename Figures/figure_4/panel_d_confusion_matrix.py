"""Panel (d): Critical-Zone Safety Confusion Matrix (|g| <= 0.10)
Part of Master Figure 4 for Elsevier RE&SS.
Clean 2x2 confusion matrix with square aspect ratio, vertical y-axis labels, and no bottom text clutter.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import numpy as np

# ==============================================================================
# 0. FONT REGISTRATION (Aptos / CloudFonts)
# ==============================================================================
cloud_dir = r"C:\Users\User\AppData\Local\Microsoft\FontCache\4\CloudFonts"
for d in glob.glob(os.path.join(cloud_dir, "Aptos*")):
    for f in glob.glob(os.path.join(d, "*.ttf")):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass

mmt.SHRINK_FACTOR = 0.85

# ==============================================================================
# MASTER CONFIGURATION & STYLING DICTIONARY
# ==============================================================================
CONFIG = {
    "figure": {
        "figsize": (8.2, 6.2),
        "dpi": 300,
    },
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12.0,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "axes.linewidth": 1.2,
        "mathtext.fontset": "stixsans",
    },
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "slate": "#475569",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

def load_data():
    cache_path = os.path.normpath(r"D:\AGravity\Tide_Tutor\data\generated\fig4_decoupling_data.npz")
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([r"D:\AGravity\Tide_Tutor\.venv\Scripts\python.exe", r"D:\AGravity\Tide_Tutor\scripts\generate_fig4_ground_truth_data.py"], check=True)
    return np.load(cache_path)

def draw_panel_d_on_ax(ax, data, config=CONFIG):
    pal = config["palette"]
    cm = data["cm"]

    ax.set_title(r"$\mathbf{(d)\ Critical\text{-}Zone\ Safety\ Confusion\ Matrix}\ (|g| \leq 0.10)$",
                 fontsize=12.5, fontweight="bold", pad=14, loc="left")

    # Heatmap with square aspect ratio
    ax.imshow(cm, cmap="Blues", interpolation="nearest", alpha=0.82)
    ax.set_aspect("equal")

    # Grid lines separating cells
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="#94a3b8", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Clean, concise 2-line cell labels
    labels = [
        [
            r"$\mathbf{True\ Safe}$" + f"\n$N = {cm[0,0]}$ ($30.7\\%$)",
            r"$\mathbf{False\ Unstable}$" + f"\n$N = {cm[0,1]}$ ($19.3\\%$)"
        ],
        [
            r"$\mathbf{False\ Safe\ (Hazard)}$" + f"\n$N = {cm[1,0]}$ ($18.7\\%$)",
            r"$\mathbf{True\ Unstable}$" + f"\n$N = {cm[1,1]}$ ($31.3\\%$)"
        ]
    ]

    for i in range(2):
        for j in range(2):
            if i == j:
                color = "white" if cm[i, j] > 130 else "#1e293b"
                weight = "bold"
            elif i == 1 and j == 0:
                color = "#991b1b"
                weight = "bold"
            else:
                color = "#1e293b"
                weight = "normal"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color,
                    fontsize=12.0, fontweight=weight, linespacing=1.3)

    # X-axis ticks (horizontal)
    ax.set_xticklabels([r"$\mathbf{Predicted\ Safe\ (\hat{g} > 0)}$",
                        r"$\mathbf{Predicted\ Unstable\ (\hat{g} < 0)}$"], fontsize=11.0)
    
    # Y-axis ticks (vertical orientation)
    ax.set_yticklabels([r"$\mathbf{True\ Safe\ (g > 0)}$",
                        r"$\mathbf{True\ Unstable\ (g < 0)}$"], fontsize=11.0,
                       rotation=90, va="center", ha="right")

    ax.set_xlabel(r"$\mathbf{Surrogate\ Operational\ Decision}$", fontsize=12.0, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"$\mathbf{Ground\text{-}Truth\ Physical\ State}$", fontsize=12.0, fontweight="bold", labelpad=10)

def generate_panel_d():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = load_data()
    draw_panel_d_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_d_confusion_matrix.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (d) to: {out_png}")

if __name__ == "__main__":
    generate_panel_d()
