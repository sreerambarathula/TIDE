import sys
"""Master Figure 5: Spectral Bias and High-Frequency Boundary Attenuation
Publication-Grade 6-Panel Layout (2 Columns x 3 Rows) at 300 DPI for Elsevier RE&SS.

Panels:
  (a) 2D Spatial Fourier Spectrum |G(kx, ky)| -> broadband high-frequency cusp radiation
  (b) Radial Power Spectral Density E(k) -> standard MLP attenuation > 10^2 vs Fourier preservation
  (c) Spatial Gradient Norm ||grad g|| across transverse cut -> refutes vanishing gradients
  (d) Multi-scale Random Fourier Feature NTK frequency bandwidth expansion into cusp band
  (e) Dual-Scale Loss Convergence Dynamics -> exponential bulk decay vs boundary loss plateau
  (f) Spatial Cusp Reconstruction & Neural Rounding -> razor-sharp ground truth vs smoothed MLP
"""
import os
import glob
import shutil
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.gridspec as gridspec
import numpy as np

# Import modular panel drawing functions
from panel_a_2d_fft_spectrum import draw_panel_a_on_ax, CONFIG as CONFIG_A
from panel_b_radial_power_spectrum import draw_panel_b_on_ax, CONFIG as CONFIG_B
from panel_c_gradient_field_fidelity import draw_panel_c_on_ax, CONFIG as CONFIG_C
from panel_d_fourier_bandwidth_mapping import draw_panel_d_on_ax, CONFIG as CONFIG_D
from panel_e_dual_scale_loss_dynamics import draw_panel_e_on_ax, CONFIG as CONFIG_E
from panel_f_spatial_cusp_rounding import draw_panel_f_on_ax, CONFIG as CONFIG_F

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

MASTER_TYPOGRAPHY = {
    "font.family": "Aptos",
    "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12.0,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
    "legend.fontsize": 11.0,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.28,
    "mathtext.fontset": "stixsans",
}

def generate_master_fig5():
    plt.rcParams.update(MASTER_TYPOGRAPHY)
    fig = plt.figure(figsize=(16.8, 20.5), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    # Draw individual panels using modular components
    draw_panel_a_on_ax(ax_a, config=CONFIG_A)
    draw_panel_b_on_ax(ax_b, config=CONFIG_B)
    draw_panel_c_on_ax(ax_c, config=CONFIG_C)
    draw_panel_d_on_ax(ax_d, config=CONFIG_D)
    draw_panel_e_on_ax(ax_e, config=CONFIG_E)
    draw_panel_f_on_ax(ax_f, config=CONFIG_F)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    ms_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "manuscript", "figures"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ms_dir, exist_ok=True)

    out_png_local = os.path.join(output_dir, "fig5_spectral_bias_master.png")
    out_pdf_local = os.path.join(output_dir, "fig5_spectral_bias_master.pdf")
    out_svg_local = os.path.join(output_dir, "fig5_spectral_bias_master.svg")

    out_png_ms = os.path.join(ms_dir, "fig5_spectral_bias.png")
    out_pdf_ms = os.path.join(ms_dir, "fig5_spectral_bias.pdf")
    out_svg_ms = os.path.join(ms_dir, "fig5_spectral_bias.svg")

    # Remove previous files if present to prevent Windows lock conflicts
    for path in [out_png_local, out_pdf_local, out_svg_local]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    fig.savefig(out_png_local, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf_local, bbox_inches="tight")
    fig.savefig(out_svg_local, bbox_inches="tight")

    # Sync to manuscript
    shutil.copy2(out_png_local, out_png_ms)
    shutil.copy2(out_pdf_local, out_pdf_ms)
    shutil.copy2(out_svg_local, out_svg_ms)
    plt.close(fig)

    print("[SUCCESS] Master Figure 5 generated & synced successfully:")
    print(f"  Local  PNG: {out_png_local}")
    print(f"  Local  PDF: {out_pdf_local}")
    print(f"  Local  SVG: {out_svg_local}")
    print(f"  Synced MS PNG: {out_png_ms}")
    print(f"  Synced MS PDF: {out_pdf_ms}")
    print(f"  Synced MS SVG: {out_svg_ms}")

if __name__ == "__main__":
    generate_master_fig5()

