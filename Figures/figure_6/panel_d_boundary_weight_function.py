"""Panel (d): Inverse-Distance Boundary Loss Weighting Function w(g)
Part of Master Figure 6 for Elsevier RE&SS.
Visualizes the 1D hyperbolic weighting function:
    w(g) = 1 / (|g| + epsilon_w)
showing backprop gradient amplification concentrated at the safety boundary g = 0.
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

def draw_panel_d(ax, config=CONFIG):
    pal = config["palette"]
    g_vals = np.linspace(-1.5, 1.5, 601)  # Exact 0.0 at center
    
    epsilons = [0.02, 0.05, 0.10, 0.25]
    colors = [pal["crimson"], pal["gold"], pal["teal"], pal["slate"]]
    linestyles = ["-", "--", "-.", ":"]
    linewidths = [2.5, 2.0, 1.9, 1.8]
    
    ax.set_title(r"$\mathbf{(d)\ Inverse\text{-}Distance\ Boundary\ Loss\ Weighting:\ }w(g) = \frac{1}{|g| + \varepsilon_w}$",
                 fontsize=12.0, fontweight="bold", loc="left", pad=12)

    # Safety boundary indicator at g = 0
    ax.axvline(0, color="#475569", ls="--", lw=1.3, zorder=2, label=r"$\mathbf{Safety\ Boundary\ (}g=0\mathbf{)}$")

    # Shaded hyperbolic concentration under optimal curve
    w_opt = 1.0 / (np.abs(g_vals) + 0.02)
    ax.fill_between(g_vals, 0, w_opt, color=pal["crimson"], alpha=0.10, zorder=1)

    for eps, col, ls, lw in zip(epsilons, colors, linestyles, linewidths):
        w = 1.0 / (np.abs(g_vals) + eps)
        ax.plot(g_vals, w, color=col, ls=ls, lw=lw, zorder=4,
                label=rf"$\mathbf{{Regularizer\ }}\varepsilon_w = {eps:.2f}$")

    # Annotation Box
    annot_text = (
        r"$\mathbf{Hyperbolic\ Gradient\ Surge}$" + "\n" +
        r"$\bullet\ w_{\max} = 1/\varepsilon_w = 50.0\ \mathrm{at\ boundary}$" + "\n" +
        r"$\bullet\ \mathrm{Focuses\ backpropagation\ on\ }g=0$" + "\n" +
        r"$\bullet\ \mathrm{Decays\ to\ }w \approx 1.0\ \mathrm{in\ far\ field}$"
    )
    ax.annotate(
        annot_text,
        xy=(0.0, 50.0), xytext=(-1.38, 22.0),
        fontsize=10.2, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["crimson"], lw=1.3),
        arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.6, shrinkA=3, shrinkB=5),
        zorder=7
    )

    # Operating Conditions Card (placed in top-left)
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.04, 0.95, info_text, transform=ax.transAxes,
            fontsize=10.5, color="#1e293b", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=pal["border_gray"], lw=1.1), zorder=8)

    ax.set_xlabel(r"$\mathbf{Limit\text{-}State\ Margin,\ }g(\mathbf{x})$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Loss\ Weight\ Multiplier,\ }w(g)$", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 56)
    ax.tick_params(labelsize=11.0, direction="in", length=3.5)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=1)

    ax.legend(loc="upper right", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=10.0, handlelength=2.2)

def generate_panel_d():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    
    draw_panel_d(ax)
    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_d_boundary_weight_function.png")
    
    if os.path.exists(out_png):
        try:
            os.remove(out_png)
        except Exception:
            pass

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (d) to: {out_png}")

if __name__ == "__main__":
    generate_panel_d()
