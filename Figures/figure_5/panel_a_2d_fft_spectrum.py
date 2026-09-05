"""Panel (a): 2D Spatial Fourier Spectrum |G(kx, ky)| of Limit-State Stability Margin Field
Part of Master Figure 5 for Elsevier RE&SS.
Shows 2D FFT magnitude spectrum of the true margin field g(Nsub, Npch), demonstrating high-frequency energy radiating along the cusp.
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
        "slate": "#475569",
        "border_gray": "#94a3b8",
        "card_bg": "#f8fafc",
    },
}

def compute_panel_a_data():
    N = 256
    nsub = np.linspace(10.5, 14.5, N)
    npch = np.linspace(16.0, 25.0, N)
    NS, NP = np.meshgrid(nsub, npch)

    NSUB_BT = 14.1428
    NPCH_BT = 20.5948

    d = np.maximum(0, NSUB_BT - NS)
    lower = NPCH_BT - 1.48 * d
    upper = NPCH_BT - 0.42 * d

    # Continuous physical stability margin g(x)
    g_field = np.minimum(NP - lower, upper - NP)
    win2d = np.outer(np.hanning(N), np.hanning(N))
    g_win = g_field * win2d

    # 2D FFT
    fft2 = np.fft.fftshift(np.fft.fft2(g_win))
    psd2d = np.abs(fft2)**2
    psd2d_log = np.log10(psd2d + 1e-3)
    psd2d_log = (psd2d_log - psd2d_log.min()) / (psd2d_log.max() - psd2d_log.min())

    kx = np.fft.fftshift(np.fft.fftfreq(N, d=(nsub[1]-nsub[0]))) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(N, d=(npch[1]-npch[0]))) * 2 * np.pi

    return {
        "kx": kx,
        "ky": ky,
        "psd2d_log": psd2d_log,
    }

def draw_panel_a_on_ax(ax, data=None, config=CONFIG):
    pal = config["palette"]
    if data is None:
        data = compute_panel_a_data()

    kx = data["kx"]
    ky = data["ky"]
    psd2d_log = data["psd2d_log"]

    ax.set_title(r"$\mathbf{(a)\ 2D\ Spatial\ Fourier\ Spectrum}\ |G(k_x, k_y)|\ \mathbf{of\ Margin\ Field}$",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")

    im = ax.imshow(psd2d_log, extent=[kx.min(), kx.max(), ky.min(), ky.max()],
                   cmap="magma", origin="lower", aspect="auto")

    cbar = plt.colorbar(im, ax=ax, pad=0.025, aspect=20)
    cbar.set_label(r"$\mathbf{Normalized\ Log\ Power\ Spectral\ Density},\ \log_{10} |G(k_x, k_y)|^2$",
                   fontsize=11.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=10.0)

    # Broadband radiation callout badge
    ax.annotate(r"$\mathbf{Broadband\ Cusp\ Radiation}$:" + "\n" +
                r"$\bullet\ \mathrm{Sharp\ knife\text{-}edge\ excites}$" + "\n" +
                r"$\bullet\ \mathrm{High\text{-}frequency\ modes}\ (k > 15)$",
                xy=(8.0, 7.5), xytext=(4.0, 16.5),
                fontsize=11.0, color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.45", fc="#1e293b", ec="#f43f5e", lw=1.3),
                arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1.4),
                zorder=6)

    # Top-Left Operating Conditions text box
    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(-23.0, 23.0, info_text,
            transform=ax.transData,
            fontsize=11.0, color="#ffffff",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#0f172a", ec=pal["border_gray"], lw=1.2, alpha=0.90),
            zorder=6)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Spatial\ Wavenumber,\ }k_x\ \mathbf{(rad/unit)}$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Phase\text{-}Change\ Spatial\ Wavenumber,\ }k_y\ \mathbf{(rad/unit)}$",
                  fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.grid(False)

def generate_panel_a():
    plt.rcParams.update(CONFIG["typography"])
    fig, ax = plt.subplots(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])
    data = compute_panel_a_data()
    draw_panel_a_on_ax(ax, data)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, "panel_a_2d_fft_spectrum.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel (a) to: {out_png}")

if __name__ == "__main__":
    generate_panel_a()

