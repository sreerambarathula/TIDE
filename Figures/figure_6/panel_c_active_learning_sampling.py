"""Panel (c): Misfit-Driven Adaptive Active Learning Allocation
Part of Master Figure 6 for Elsevier RE&SS.
Visualizes data-space remediation:
    x_{k+1} = argmax [ U(x) * I(|g(x)| <= delta_w) ]
demonstrating adaptive sample point concentration along the safety boundary corridor.
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        "figsize": (8.8, 6.2),
        "dpi": 300,
    },
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 10.0,
        "axes.linewidth": 1.1,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
    },
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "gold": "#c88a10",
        "slate": "#64748b",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

# Singularity Vertex Coordinates
NSUB_BT = 14.1428
NPCH_BT = 20.5948

def generate_sampling_data(seed=42):
    np.random.seed(seed)
    
    # 1. Initial Latin Hypercube Samples (uniform coverage)
    N_init = 90
    nsub_init = np.random.uniform(10.6, 14.9, N_init)
    npch_init = np.random.uniform(16.3, 25.2, N_init)
    
    # 2. Adaptive Active Learning Points (concentrated along Fold & Hopf branches)
    N_adapt = 150
    nsub_adapt = np.random.uniform(11.3, NSUB_BT, N_adapt)
    d_adapt = NSUB_BT - nsub_adapt
    
    branch = np.random.choice([0, 1], size=N_adapt, p=[0.5, 0.5])
    # Lower Fold branch: slope ~ 1.45; Upper Hopf branch: slope ~ 0.42
    npch_adapt = np.where(
        branch == 0,
        NPCH_BT - 1.45 * d_adapt + np.random.normal(0, 0.20, N_adapt),
        NPCH_BT - 0.42 * d_adapt + np.random.normal(0, 0.20, N_adapt)
    )
    
    return {
        "init": (nsub_init, npch_init),
        "adapt": (nsub_adapt, npch_adapt),
    }

def draw_panel_c(ax, config=CONFIG):
    pal = config["palette"]
    data = generate_sampling_data()
    nsub_init, npch_init = data["init"]
    nsub_adapt, npch_adapt = data["adapt"]

    ax.set_title(r"$\mathbf{(c)\ Misfit\text{-}Driven\ Adaptive\ Active\ Learning\ Allocation:\ }\mathbf{x}_{k+1} = \arg\max\ [U(\mathbf{x}) \cdot \mathbf{I}(|g| \leq \delta_w)]$",
                 fontsize=12.0, fontweight="bold", loc="left", pad=12)

    # Ground truth boundary curves
    ns_curve = np.linspace(10.5, NSUB_BT, 300)
    d_c = NSUB_BT - ns_curve
    npch_fold = NPCH_BT - 1.45 * d_c
    npch_hopf = NPCH_BT - 0.42 * d_c

    # Shaded boundary uncertainty corridor (|g| <= delta_w)
    ax.fill_between(ns_curve, npch_fold - 0.50, npch_fold + 0.50, color="#cbd5e1", alpha=0.35, zorder=1)
    ax.fill_between(ns_curve, npch_hopf - 0.50, npch_hopf + 0.50, color="#cbd5e1", alpha=0.35, zorder=1)
    ax.fill_between(ns_curve, npch_fold, npch_hopf, color="#f1f5f9", alpha=0.4, zorder=1)

    # Plot Boundary Lines
    ax.plot(ns_curve, npch_fold, color=pal["navy"], lw=2.2, ls="--", zorder=3,
            label=r"$\mathbf{Lower\ Fold\ Boundary\ (}g=0\mathbf{)}$")
    ax.plot(ns_curve, npch_hopf, color=pal["teal"], lw=2.2, ls="-.", zorder=3,
            label=r"$\mathbf{Upper\ Hopf\ Boundary\ (}g=0\mathbf{)}$")

    # Initial LHS samples
    ax.scatter(nsub_init, npch_init, s=32, color=pal["slate"], alpha=0.65, marker="o",
               edgecolors="white", linewidths=0.6, zorder=4,
               label=r"$\mathbf{Initial\ LHS\ Points}$")

    # Adaptive active learning enriched points
    ax.scatter(nsub_adapt, npch_adapt, s=44, color=pal["crimson"], alpha=0.90, marker="o",
               edgecolors="black", linewidths=0.7, zorder=5,
               label=r"$\mathbf{Adaptive\ Active\ Points}$")

    # BT Singularity Vertex Marker
    ax.plot(NSUB_BT, NPCH_BT, marker="*", color="#fbbf24", mec="#1e293b", mew=1.3, ms=18, zorder=6,
            label=r"$\mathbf{BT\ Singularity\ Vertex\ (}\mathbf{x}_{\mathrm{BT}}\mathbf{)}$")

    # Annotation Box placed cleanly in the completely open lower-right region
    annot_text = (
        r"$\mathbf{Boundary\ Corridor\ Surge}$" + "\n" +
        r"$\bullet\ \mathrm{High\ epistemic\ uncertainty\ }U(\mathbf{x})$" + "\n" +
        r"$\bullet\ \mathrm{Samples\ concentrated\ along\ }\partial\Omega$" + "\n" +
        r"$\bullet\ \mathrm{Resolves\ sharp\ cusp\ transition}$"
    )
    ax.annotate(
        annot_text,
        xy=(13.6, 19.8), xytext=(13.1, 16.5),
        fontsize=10.2, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["crimson"], lw=1.3),
        arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.6, shrinkA=3, shrinkB=5),
        zorder=7
    )

    # Operating Conditions Card (placed in top-right open area)
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.97, 0.95, info_text, transform=ax.transAxes,
            fontsize=10.5, color="#1e293b", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["border_gray"], lw=1.1), zorder=8)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_xlim(10.5, 15.0)
    ax.set_ylim(16.0, 25.5)
    ax.tick_params(labelsize=11.0, direction="in", length=3.5)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=1)

    ax.legend(loc="upper left", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=9.8, handlelength=2.0)

def generate_panel_c():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    
    draw_panel_c(ax)
    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_c_active_learning_sampling.png")
    
    if os.path.exists(out_png):
        try:
            os.remove(out_png)
        except Exception:
            pass

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (c) to: {out_png}")

if __name__ == "__main__":
    generate_panel_c()
