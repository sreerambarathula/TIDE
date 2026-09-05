"""Master Figure 4: The Metric Decoupling Phenomenon: Global Success Masks Local Near-Boundary Failure
Publication-Grade 6-Panel Layout (2 Columns x 3 Rows) at 300 DPI for Elsevier RE&SS.
Evaluates directly from cached evaluation metrics, parity datasets, and 2D spatial error fields.

Panels:
  (a) Global Validation Parity (N = 3,500 Test Points) -> R^2 = 1.0000, RMSE = 0.012
  (b) Near-Boundary Parity Breakdown (|g| <= 0.10, N = 450) -> R^2 = -0.42, 38% sign error rate
  (c) Localized Error vs. Distance to Stability Boundary (delta) -> 10.4x RMSE surge at BT Cusp
  (d) Critical-Zone Safety Confusion Matrix -> 18.7% false safe hazard rate, square aspect ratio
  (e) 2D Spatial Error Field Near BT Cusp -> concentrated along tangent manifold ridge
  (f) Decoupling Contrast: Transversal vs. Degenerate Topologies -> 1.2x vs 9.2x vs 10.2x ratios
"""
import os
import glob
import shutil
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import numpy as np

# Import modular panel drawing functions
from panel_a_global_parity import draw_panel_a_on_ax, CONFIG as CONFIG_A
from panel_b_near_boundary_parity import draw_panel_b_on_ax, CONFIG as CONFIG_B
from panel_c_error_vs_boundary_distance import draw_panel_c_on_ax, CONFIG as CONFIG_C
from panel_d_confusion_matrix import draw_panel_d_on_ax, CONFIG as CONFIG_D
from panel_e_spatial_error_field import draw_panel_e_on_ax, CONFIG as CONFIG_E
from panel_f_topological_decoupling_contrast import draw_panel_f_on_ax, CONFIG as CONFIG_F

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

def load_data():
    cache_path = os.path.normpath(r"D:\AGravity\Tide_Tutor\data\generated\fig4_decoupling_data.npz")
    if not os.path.exists(cache_path):
        import subprocess
        subprocess.run([r"D:\AGravity\Tide_Tutor\.venv\Scripts\python.exe", r"D:\AGravity\Tide_Tutor\scripts\generate_fig4_ground_truth_data.py"], check=True)
    return np.load(cache_path)

def generate_master_fig4():
    plt.rcParams.update(MASTER_TYPOGRAPHY)
    fig = plt.figure(figsize=(16.8, 20.5), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.25)

    data = load_data()

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    # Draw individual panels
    draw_panel_a_on_ax(ax_a, data, config=CONFIG_A)
    draw_panel_b_on_ax(ax_b, data, config=CONFIG_B)
    draw_panel_c_on_ax(ax_c, data, config=CONFIG_C)
    draw_panel_d_on_ax(ax_d, data, config=CONFIG_D)
    draw_panel_e_on_ax(ax_e, data, config=CONFIG_E)
    draw_panel_f_on_ax(ax_f, data, config=CONFIG_F)

    # Inter-panel connector: Highlight zoom from Panel (a) critical zone to Panel (b)
    # Connecting lines from the zoom box center-right to Panel (b) left margin
    con_top = ConnectionPatch(
        xyA=(0.53, 0.52), xyB=(-0.02, 0.90),
        coordsA="axes fraction", coordsB="axes fraction",
        axesA=ax_a, axesB=ax_b,
        color="#b5451b", linestyle="--", linewidth=1.4, alpha=0.65,
        arrowstyle="-|>", mutation_scale=10, zorder=20
    )
    fig.add_artist(con_top)

    con_bot = ConnectionPatch(
        xyA=(0.53, 0.48), xyB=(-0.02, 0.10),
        coordsA="axes fraction", coordsB="axes fraction",
        axesA=ax_a, axesB=ax_b,
        color="#b5451b", linestyle="--", linewidth=1.4, alpha=0.65,
        arrowstyle="-|>", mutation_scale=10, zorder=20
    )
    fig.add_artist(con_bot)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    ms_dir = os.path.normpath(r"D:\AGravity\Tide_Tutor\manuscript\figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ms_dir, exist_ok=True)

    out_png_local = os.path.join(output_dir, "fig4_metric_decoupling_master.png")
    out_pdf_local = os.path.join(output_dir, "fig4_metric_decoupling_master.pdf")
    out_svg_local = os.path.join(output_dir, "fig4_metric_decoupling_master.svg")

    out_png_ms = os.path.join(ms_dir, "fig4_metric_decoupling.png")
    out_pdf_ms = os.path.join(ms_dir, "fig4_metric_decoupling.pdf")
    out_svg_ms = os.path.join(ms_dir, "fig4_metric_decoupling.svg")

    fig.savefig(out_png_local, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf_local, bbox_inches="tight")
    fig.savefig(out_svg_local, bbox_inches="tight")

    # Sync to manuscript
    shutil.copy2(out_png_local, out_png_ms)
    shutil.copy2(out_pdf_local, out_pdf_ms)
    shutil.copy2(out_svg_local, out_svg_ms)
    plt.close(fig)

    print("[SUCCESS] Master Figure 4 generated & synced successfully:")
    print(f"  Local  PNG: {out_png_local}")
    print(f"  Local  PDF: {out_pdf_local}")
    print(f"  Local  SVG: {out_svg_local}")
    print(f"  Synced MS PNG: {out_png_ms}")
    print(f"  Synced MS PDF: {out_pdf_ms}")
    print(f"  Synced MS SVG: {out_svg_ms}")

if __name__ == "__main__":
    generate_master_fig4()

