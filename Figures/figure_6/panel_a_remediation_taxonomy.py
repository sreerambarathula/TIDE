"""Panel (a): Multi-Scale Random Fourier Basis Functions
Part of Master Figure 6 for Elsevier RE&SS.
Visualizes multi-scale sinusoidal projection basis vectors [sin(2*pi*B*x), cos(2*pi*B*x)] across coordinate space.
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
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11.0,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "mathtext.fontset": "stixsans",
    },
    "palette": {
        "navy": "#1a3c6e",
        "crimson": "#b5451b",
        "teal": "#007a78",
        "gold": "#c88a10",
        "slate": "#475569",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

def compute_panel_a_data():
    x_norm = np.linspace(-1.0, 1.0, 400)
    # Sample 4 random Fourier basis frequencies B ~ N(0, sigma^2) with sigma = 3.5
    b_modes = np.array([0.8, 2.2, 4.5, 7.8])
    return {
        "x_norm": x_norm,
        "b_modes": b_modes,
    }

def draw_panel_a_on_ax(ax, data=None, config=CONFIG):
    pal = config["palette"]
    if data is None:
        data = compute_panel_a_data()

    x_norm = data["x_norm"]
    b_modes = data["b_modes"]
    colors = [pal["navy"], pal["teal"], pal["gold"], pal["crimson"]]

    ax.set_title(r"$\mathbf{(a)\ Multi\text{-}Scale\ Random\ Fourier\ Basis\ Functions}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    for i, (b, col) in enumerate(zip(b_modes, colors)):
        basis_sin = np.sin(2 * np.pi * b * x_norm)
        ax.plot(x_norm, basis_sin, color=col, lw=2.2, alpha=0.92,
                label=f"Basis Mode {i+1} ($B_{{{i+1}}} = {b:.1f}$)", zorder=4)

    # Harmonic Basis Embedding Callout box (top left)
    callout_text = (
        r"$\mathbf{Harmonic\ Feature\ Embedding}$:" + "\n" +
        r"$\bullet\ \gamma(\mathbf{x}) = [\sin(2\pi\mathbf{Bx}),\ \cos(2\pi\mathbf{Bx})]$" + "\n" +
        r"$\bullet\ \mathrm{Spans\ multiple\ spatial\ octaves}$" + "\n" +
        r"$\bullet\ \mathrm{Resolves\ sub\text{-}boundary\ cusp\ modes}$"
    )
    ax.text(-0.95, 0.45, callout_text,
            transform=ax.transData,
            fontsize=11.0, color=pal["teal"],
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f0fdfa", ec=pal["teal"], lw=1.2),
            zorder=6)

    # Operating Conditions text box (bottom left)
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(-0.95, -0.65, info_text,
            transform=ax.transData,
            fontsize=11.0, color="#1e293b",
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec=pal["border_gray"], lw=1.2),
            zorder=6)

    ax.set_xlabel(r"$\mathbf{Standardized\ Parameter\ Coordinate,\ }x_{\mathrm{norm}} \in [-1, 1]$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Harmonic\ Feature\ Amplitude,\ }\sin(2\pi B x)$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.30, 1.40)
    ax.grid(True, alpha=0.28)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.5)
    leg.set_zorder(10)

def generate_panel_a():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = compute_panel_a_data()
    draw_panel_a_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_a_fourier_feature_mapping.png")
    
    if os.path.exists(out_png):
        try:
            os.remove(out_png)
        except Exception:
            pass

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (a) to: {out_png}")

if __name__ == "__main__":
    generate_panel_a()


