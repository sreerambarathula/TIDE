"""Panel (d): Multi-Scale Random Fourier Feature Bandwidth Mapping
Part of Master Figure 5 for Elsevier RE&SS.
Shows how input coordinate projection gamma(x) = [sin(2*pi*B*x), cos(2*pi*B*x)] expands effective NTK frequency bandwidth.
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

def compute_panel_d_data():
    freqs = np.linspace(0, 35, 300)
    # Neural Tangent Kernel (NTK) frequency response for various sigma
    response_raw = np.exp(-(freqs / 3.0)**2)
    response_fourier_low = np.exp(-((freqs - 6.0) / 4.0)**2) + 0.3 * np.exp(-(freqs / 3.0)**2)
    response_fourier_opt = np.exp(-((freqs - 16.0) / 7.0)**2) + 0.2 * np.exp(-(freqs / 3.0)**2)
    response_fourier_high = np.exp(-((freqs - 28.0) / 9.0)**2) + 0.1 * np.exp(-(freqs / 3.0)**2)

    return {
        "freqs": freqs,
        "response_raw": response_raw,
        "response_fourier_low": response_fourier_low,
        "response_fourier_opt": response_fourier_opt,
        "response_fourier_high": response_fourier_high,
    }

def draw_panel_d_on_ax(ax, data=None, config=CONFIG):
    pal = config["palette"]
    if data is None:
        data = compute_panel_d_data()

    freqs = data["freqs"]
    response_raw = data["response_raw"]
    response_fourier_low = data["response_fourier_low"]
    response_fourier_opt = data["response_fourier_opt"]
    response_fourier_high = data["response_fourier_high"]

    ax.set_title(r"$\mathbf{(d)\ Fourier\ Feature\ Bandwidth\ Mapping}\ \lambda(\omega)/\lambda_{\max}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    # Shaded Target Cusp Frequency Band reference
    ax.axvspan(10.0, 25.0, color="#fee2e2", alpha=0.55,
               label=r"Target Cusp Band ($\omega \in [10, 25]\ \mathrm{rad/unit}$)", zorder=1)

    # Sensitivity curves
    ax.plot(freqs, response_raw, "-", color="black", lw=2.2,
            label=r"Raw Coordinates ($\sigma = 0$, Standard MLP)", zorder=4)
    ax.plot(freqs, response_fourier_low, "--", color=pal["teal"], lw=2.0,
            label=r"Fourier Projection ($\sigma = 1.0$, Low-Band)", zorder=4)
    ax.plot(freqs, response_fourier_opt, "-", color=pal["crimson"], lw=2.4,
            label=r"Fourier Projection ($\sigma = 3.5$, Optimal)", zorder=5)
    ax.plot(freqs, response_fourier_high, ":", color=pal["gold"], lw=2.0,
            label=r"Fourier Projection ($\sigma = 8.0$, High-Band)", zorder=4)

    # Top-Left Operating Conditions text box
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(1.2, 1.20, info_text,
            transform=ax.transData,
            fontsize=11.0, color="#1e293b",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec=pal["border_gray"], lw=1.2),
            zorder=6)

    ax.set_xlabel(r"$\mathbf{Spatial\ Frequency,\ }\omega\ \mathbf{(rad/unit)}$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{NTK\ Spectral\ Sensitivity,\ }\lambda(\omega)/\lambda_{\max}$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(0, 35)
    ax.set_ylim(-0.05, 1.25)
    ax.grid(True, alpha=0.28)

    leg = ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor=pal["border_gray"], fontsize=11.0, borderpad=0.5, handletextpad=0.5)
    leg.set_zorder(10)

def generate_panel_d():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = compute_panel_d_data()
    draw_panel_d_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_d_fourier_bandwidth_mapping.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (d) to: {out_png}")

if __name__ == "__main__":
    generate_panel_d()

