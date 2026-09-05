r"""Master Composite Figure 1: Multiscale Thermal-Hydraulic Dynamics of Two-Phase Flow Instabilities
Publication-Grade 6-Panel Layout (2 Columns x 3 Rows) at 300 DPI for Elsevier RE&SS.

Panels:
  (a) Physical Channel Architecture & Instability Mechanisms (Self-Authored SVG Drawing)
  (b) Thermodynamic Void Non-Linearity \alpha(x, P) across Isobaric Regimes
  (c) Hydrodynamic Force Decomposition & Internal S-Curve Genesis
  (d) Static Ledinegg Instability & Dynamic Excursion Runaway Jump
  (e) Dynamic Density-Wave Transit Delay \tau_{12} & Regenerative Feedback Loop
  (f) Supercritical Hopf Bifurcation & Limit-Cycle Attractor
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np
import pymupdf

# ==============================================================================
# 0. FONT REGISTRATION (Aptos / CloudFonts) & MATHTEXT SCALING
# ==============================================================================
cloud_dir = r"C:\Users\User\AppData\Local\Microsoft\FontCache\4\CloudFonts"
for d in glob.glob(os.path.join(cloud_dir, "Aptos*")):
    for f in glob.glob(os.path.join(d, "*.ttf")):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass

mmt.SHRINK_FACTOR = 0.85  # Subscript & superscript font scaling (85% of base)

# Master Typography Setting
MASTER_TYPOGRAPHY = {
    "font.family": "Aptos",
    "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12.0,
    "axes.titlesize": 13.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.2,
    "axes.linewidth": 1.3,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.28,
    "mathtext.fontset": "stixsans",
}


# ==============================================================================
# PANEL (A): USER AUTHORED SVG DRAWING (panel_a_zonation_self.svg)
# ==============================================================================
def draw_panel_a(ax):
    svg_file = "d:/AGravity/Tide_Tutor/Figures/figure_1/panel_a_zonation_self.svg"
    img_path = "d:/AGravity/Tide_Tutor/Figures/figure_1/panel_a_zonation_self_300dpi.png"

    if not os.path.exists(img_path) and os.path.exists(svg_file):
        with open(svg_file, "r", encoding="utf-8") as f:
            svg_data = f.read()
        doc = pymupdf.open(stream=svg_data.encode("utf-8"), filetype="svg")
        page = doc[0]
        mat = pymupdf.Matrix(300.0 / 72.0, 300.0 / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(img_path)

    if os.path.exists(img_path):
        img = plt.imread(img_path)
        ax.imshow(img)

    ax.axis("off")
    ax.set_title(r"$\mathbf{(a)}$ Boiling Physics & Instability Mechanisms",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")


# ==============================================================================
# PANEL (B): THERMODYNAMIC VOID NON-LINEARITY (100% Exact from panel_b_void_paradox.py)
# ==============================================================================
def draw_panel_b(ax):
    ax.set_title(r"$\mathbf{(b)\ Thermodynamic\ Void\ Non\text{-}Linearity\ \alpha(x, P)}$",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Thermodynamic\ Flow\ Quality,\ x\ (\%)}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Cross\text{-}Sectional\ Void\ Fraction,\ \alpha\ (\%)}$", fontsize=12.0, fontweight="bold", labelpad=2)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(True)

    x_vals = np.linspace(0.0, 1.0, 600)
    curves = [
        {
            "id": "1bar",
            "label": r"$\mathbf{1\,\mathrm{bar}\ (\rho_f/\rho_g \approx 1600)}$",
            "gamma": 0.598 / 958.4,
            "color": "#94a3b8",
            "linestyle": ":",
            "linewidth": 1.8,
            "zorder": 2,
        },
        {
            "id": "10bar",
            "label": r"$\mathbf{10\,\mathrm{bar}\ (\rho_f/\rho_g \approx 172)}$",
            "gamma": 5.15 / 887.0,
            "color": "#007a78",
            "linestyle": "-.",
            "linewidth": 1.8,
            "zorder": 3,
        },
        {
            "id": "70bar",
            "label": r"$\mathbf{70\,\mathrm{bar}\ (\rho_f/\rho_g \approx 20.5)\ [Ref]}$",
            "gamma": 36.1 / 740.0,
            "color": "#b5451b",
            "linestyle": "-",
            "linewidth": 2.8,
            "zorder": 4,
        },
        {
            "id": "150bar",
            "label": r"$\mathbf{150\,\mathrm{bar}\ (\rho_f/\rho_g \approx 6.0)}$",
            "gamma": 100.0 / 600.0,
            "color": "#6b21a8",
            "linestyle": "--",
            "linewidth": 2.0,
            "zorder": 3,
        },
        {
            "id": "critical",
            "label": r"$\mathbf{P \to P_c\ (\rho_f/\rho_g = 1.0)\ [Linear]}$",
            "gamma": 1.0,
            "color": "#64748b",
            "linestyle": "--",
            "linewidth": 1.3,
            "zorder": 2,
        },
    ]

    for c in curves:
        alpha_vals = x_vals / (x_vals + (1.0 - x_vals) * c["gamma"])
        ax.plot(x_vals * 100, alpha_vals * 100,
                color=c["color"], lw=c["linewidth"], ls=c["linestyle"], label=c["label"], zorder=c["zorder"])

    # Reference Operating Point (70 bar, x = 5%)
    x_tgt = 0.05
    gamma_ref = 36.1 / 740.0
    alpha_tgt = (x_tgt / (x_tgt + (1.0 - x_tgt) * gamma_ref)) * 100.0

    ax.plot([x_tgt * 100, x_tgt * 100], [0, alpha_tgt], color="#b45309", ls=":", lw=1.8, zorder=5)
    ax.plot([0, x_tgt * 100], [alpha_tgt, alpha_tgt], color="#b45309", ls=":", lw=1.8, zorder=5)
    ax.plot(x_tgt * 100, alpha_tgt, marker="o", markersize=9.0, color="#b5451b", mec="black", mew=1.3, zorder=6)

    # 1. Governing Equation Card (x=25, y=5, bottom-left)
    eq_text = (r"$\mathbf{Governing\ Void\ Model:}$" + "\n" +
               r"$\mathbf{\alpha(x, P) = \frac{x}{x + (1 - x)\,(\rho_g / \rho_f)}}$")
    ax.text(25.0, 5.0, eq_text, fontsize=11.0, color="#1a3c6e", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.50", fc="#f8fafc", ec="#94a3b8", lw=1.2, alpha=0.96), zorder=7)

    # 2. 5% Void Paradox Callout Box (x=7, y=32)
    callout_text = (r"$\mathbf{5\%\ Mass\ Quality\ (x = 0.05)}$" + "\n" +
                    r"$\mathbf{\rightarrow\ \alpha = 51.9\%\ Vapor\ Volume!}$" + "\n" +
                    r"$\mathbf{(Extreme\ Volumetric\ Expansion)}$")
    ax.annotate(callout_text, xy=(x_tgt * 100, alpha_tgt), xytext=(7.0, 32.0),
                fontsize=9.8, color="#9a3412",
                bbox=dict(boxstyle="round,pad=0.50", fc="#fffaf5", ec="#ea580c", lw=1.4),
                arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.8), zorder=7)

    # 3. Pressure Trend Annotation
    ax.annotate("", xy=(60.0, 60.0), xytext=(0.0, 100.0), xycoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#007a78", alpha=0.5, lw=2.2, mutation_scale=14, linestyle="-"), zorder=5)
    trend_text = (r"$\mathbf{Pressure\ P\uparrow \;\rightarrow\; \rho_f/\rho_g \downarrow}$" + "\n" +
                  "Phase Slip & Expansion Diminish")
    ax.text(61.0, 62.0, trend_text, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#1a3c6e",
            bbox=dict(boxstyle="round,pad=0.42", fc="#f0fdfa", ec="#007a78", lw=1.2, alpha=0.96), zorder=6)

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cbd5e1", fontsize=9.4)


# ==============================================================================
# PANEL (C): HYDRODYNAMIC FORCE DECOMPOSITION (100% Exact from panel_c_force_decomposition.py)
# ==============================================================================
def draw_panel_c(ax):
    ax.set_title(r"$\mathbf{(c)}$ Hydrodynamic Force Decomposition & S-Curve",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Coolant\ Mass\ Flux,\ G\ (kg/(m^2 s))}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Internal\ Pressure\ Drop,\ \Delta P\ (bar)}$", fontsize=12.0, fontweight="bold", labelpad=2)
    ax.set_xlim(0.05, 3.25)
    ax.set_ylim(-0.05, 3.35)
    ax.grid(True)

    G = np.linspace(0.08, 3.25, 600)
    dp_1phi = 0.22 * G**2 + 0.05 * G
    dp_2phi = 2.4 / (1.0 + 3.0 * G**1.8)
    dp_grav = 0.55 * (1.0 - 0.75 / (1.0 + 2.0 * G**1.5))
    dp_total = dp_1phi + dp_2phi + dp_grav

    d_dp = np.gradient(dp_total, G)
    idx_neg = np.where(d_dp < 0)[0]
    g_neg_end = G[idx_neg[-1]]
    dp_min_val = dp_total[idx_neg[-1]]

    # 1. Shaded Instability Zone
    ax.axvspan(0.05, g_neg_end, color="#fee2e2", alpha=0.55,
               label=r"$\mathbf{Negative\ Slope\ Zone\ (\partial \Delta P / \partial G < 0)}$", zorder=1)
    ax.axvline(g_neg_end, color="#ef4444", ls="--", lw=1.2, zorder=2)

    # 2. Component Curves
    ax.plot(G, dp_1phi, color="#007a78", lw=2.0, ls="-.", label=r"$\mathbf{Single\text{-}Phase\ Friction\ (\Delta P_{\mathrm{single}} \propto G^2)}$", zorder=3)
    ax.plot(G, dp_2phi, color="#c88a10", lw=2.0, ls="--", label=r"$\mathbf{Two\text{-}Phase\ Boiling\ Loss\ (\Delta P_{\mathrm{two\text{-}phase}} \propto 1/\rho_m)}$", zorder=3)
    ax.plot(G, dp_grav, color="#475569", lw=1.8, ls=":", label=r"$\mathbf{Hydrostatic\ Gravity\ Head\ (\Delta P_{\mathrm{grav}} \propto \rho_m g)}$", zorder=2)
    ax.plot(G, dp_total, color="#1a3c6e", lw=3.0, ls="-", label=r"$\mathbf{Total\ Internal\ \Delta P_{\mathrm{int}}(G)\ (S\text{-}Curve)}$", zorder=5)

    # 3. Turning Point (Local Minimum at exact coordinates)
    ax.plot(g_neg_end, dp_min_val, marker="o", markersize=8.5, color="#dc2626", mec="black", mew=1.3, zorder=7)
    ax.plot([g_neg_end, g_neg_end], [0, dp_min_val], color="#dc2626", ls=":", lw=1.5, zorder=6)

    # Turning Point Annotation (Text at (1.0, 1.8) exactly)
    tp_text = (r"$\mathbf{Ledinegg\ Turning\ Point}$" + "\n" +
               r"$\mathbf{\left(\frac{\partial \Delta P}{\partial G} = 0\right)\ \rightarrow\ Neutral\ Limit}$")
    ax.annotate(tp_text, xy=(g_neg_end, dp_min_val), xytext=(1.0, 1.8),
                fontsize=9.3, fontweight="bold", color="#991b1b",
                bbox=dict(boxstyle="round,pad=0.42", fc="#fff5f5", ec="#f87171", lw=1.1, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), zorder=7)

    # 4. Governing Force Balance Card (Compact, positioned with generous margin from legend)
    eq_text = (r"$\mathbf{Internal\ Force\ Balance:}$" + "\n" +
               r"$\mathbf{\Delta P_{\mathrm{int}} = \Delta P_{\mathrm{single}} + \Delta P_{\mathrm{two\text{-}phase}} + \Delta P_{\mathrm{grav}}}$")
    ax.text(0.10, 2.98, eq_text, fontsize=9.0, fontweight="bold", color="#1a3c6e", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.38", fc="#f8fafc", ec="#94a3b8", lw=1.1, alpha=0.96), zorder=6)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cbd5e1", fontsize=9.0, labelspacing=0.32)


# ==============================================================================
# PANEL (D): STATIC LEDINEGG INSTABILITY (100% Exact from panel_d_ledinegg_excursion.py)
# ==============================================================================
def draw_panel_d(ax):
    ax.set_title(r"$\mathbf{(d)}$ Static Ledinegg Instability & Flow Excursion",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Coolant\ Mass\ Flux,\ G\ (kg/(m^2 s))}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Internal\ Pressure\ Drop,\ \Delta P\ (bar)}$", fontsize=12.0, fontweight="bold", labelpad=2)
    ax.set_xlim(0.1, 3.15)
    ax.set_ylim(0.35, 2.65)
    ax.grid(True)

    G = np.linspace(0.18, 3.08, 600)
    def s_curve_func(g):
        x = g - 1.6
        return x**3 - 1.15 * x + 1.45

    dp_int = s_curve_func(G)
    pump_head = 1.45
    pump_line = np.full_like(G, pump_head)

    delta_turn = np.sqrt(1.15 / 3.0)
    g_turn_max = 1.6 - delta_turn
    g_turn_min = 1.6 + delta_turn

    # 1. Shaded Instability Zone
    ax.axvspan(g_turn_max, g_turn_min, color="#fee2e2", alpha=0.50,
               label=r"$\mathbf{Negative\ Slope\ Zone\ (\partial \Delta P_{\mathrm{int}} / \partial G < 0)}$", zorder=1)
    ax.axvline(g_turn_max, color="#f87171", ls=":", lw=1.2, zorder=2)
    ax.axvline(g_turn_min, color="#f87171", ls=":", lw=1.2, zorder=2)

    # 2. Physics Curves
    ax.plot(G, dp_int, color="#1a3c6e", lw=2.8, ls="-", label=r"$\mathbf{Channel\ Characteristic\ \Delta P_{\mathrm{int}}(G)}$", zorder=4)
    ax.plot(G, pump_line, color="#dc2626", lw=2.0, ls="--", label=r"$\mathbf{External\ Supply\ \Delta P_{\mathrm{ext}}\ (Plenum\ \Delta P_0)}$", zorder=3)

    # 3. Intersection Roots (Operating Points A, B, C)
    delta_root = np.sqrt(1.15)
    # Point A
    gx_A, gy_A = 1.6 - delta_root, pump_head
    ax.plot(gx_A, gy_A, "o", color="#007a78", markersize=8.5, markeredgecolor="black", markeredgewidth=1.3, zorder=6)
    text_A = r"$\mathbf{Point\ A\ (Stable)}$" + "\n" + r"$\mathbf{Vapor\text{-}Rich\ (\frac{\partial \Delta P}{\partial G} > 0)}$"
    ax.annotate(text_A, xy=(gx_A, gy_A), xytext=(0.53, 0.65),
                fontsize=9.0, fontweight="bold", color="#005a58", ha="center",
                bbox=dict(boxstyle="round,pad=0.36", fc="#f0fdfa", ec="#5eead4", lw=1.1, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color="#007a78", lw=1.3), zorder=7)

    # Point B
    gx_B, gy_B = 1.60, pump_head
    ax.plot(gx_B, gy_B, "o", color="#dc2626", markersize=8.5, markeredgecolor="black", markeredgewidth=1.3, zorder=6)
    text_B = r"$\mathbf{Point\ B\ (Unstable\ Saddle)}$" + "\n" + r"$\mathbf{Negative\ Slope\ (\frac{\partial \Delta P}{\partial G} < 0)}$"
    ax.annotate(text_B, xy=(gx_B, gy_B), xytext=(1.60, 2.22),
                fontsize=9.0, fontweight="bold", color="#991b1b", ha="center",
                bbox=dict(boxstyle="round,pad=0.36", fc="#fff5f5", ec="#fca5a5", lw=1.1, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.3), zorder=7)

    # Point C
    gx_C, gy_C = 1.6 + delta_root, pump_head
    ax.plot(gx_C, gy_C, "o", color="#007a78", markersize=8.5, markeredgecolor="black", markeredgewidth=1.3, zorder=6)
    text_C = r"$\mathbf{Point\ C\ (Stable)}$" + "\n" + r"$\mathbf{Liquid\text{-}Rich\ (\frac{\partial \Delta P}{\partial G} > 0)}$"
    ax.annotate(text_C, xy=(gx_C, gy_C), xytext=(2.67, 2.15),
                fontsize=9.0, fontweight="bold", color="#005a58", ha="center",
                bbox=dict(boxstyle="round,pad=0.36", fc="#f0fdfa", ec="#5eead4", lw=1.1, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color="#007a78", lw=1.3), zorder=7)

    # 4. SOLID Flow Excursion Jump Arrow (B -> A)
    ax.annotate("", xy=(0.60, 1.40), xytext=(1.52, 1.40),
                arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=2.2, ls="-", mutation_scale=16, connectionstyle="arc3,rad=-0.20"), zorder=6)
    text_exc = r"$\mathbf{Runaway\ Flow\ Collapse}$" + "\n" + r"$\mathbf{B \rightarrow A\ (Burnout\ /\ CHF)}$"
    ax.text(1.06, 1.12, text_exc, fontsize=8.8, color="#991b1b", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.30", fc="#fff5f5", ec="#f87171", lw=1.0, alpha=0.94), zorder=7)

    # 5. Stability Criterion Card (Compact layout, no overlap with Point B)
    sc_text = (r"$\mathbf{Ledinegg\ Stability\ Criterion:}$" + "\n" +
               r"$\mathbf{\frac{\partial \Delta P_{\mathrm{ext}}}{\partial G} < \frac{\partial \Delta P_{\mathrm{int}}}{\partial G}\ \rightarrow\ Stable\ (A,\ C)}$" + "\n" +
               r"$\mathbf{\frac{\partial \Delta P_{\mathrm{ext}}}{\partial G} > \frac{\partial \Delta P_{\mathrm{int}}}{\partial G}\ \rightarrow\ Excursion\ (B)}$")
    ax.text(0.14, 2.56, sc_text, fontsize=8.8, fontweight="bold", color="#1a3c6e", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.38", fc="#f8fafc", ec="#94a3b8", lw=1.1, alpha=0.96), zorder=7)

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cbd5e1", fontsize=9.2)


# ==============================================================================
# PANEL (E): DYNAMIC DENSITY-WAVE TRANSIT DELAY (100% Exact from panel_e_transit_delay.py)
# ==============================================================================
def draw_panel_e(ax):
    ax.set_title(r"$\mathbf{(e)}$ Dynamic Density-Wave Transit Delay $\mathbf{\tau_{12}}$ & Feedback",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Dimensionless\ Time,\ t\ /\ T_0}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Perturbation\ Amplitude\ (\delta u_i,\ \delta \Delta P_{\mathrm{exit}})}$", fontsize=12.0, fontweight="bold", labelpad=2)
    ax.set_xlim(0.0, 11.5)
    ax.set_ylim(-0.72, 0.75)
    ax.grid(True)

    t = np.linspace(0.0, 11.5, 600)
    T = 4.0
    tau = 2.0
    delta_ui = 0.28 * np.cos(2 * np.pi * t / T)
    delta_dp = 0.38 * np.cos(2 * np.pi * (t - tau) / T)

    # 1. Zero Baseline
    ax.axhline(0, color="#94a3b8", lw=0.9, ls="-", alpha=0.6, zorder=1)

    # 2. Waveforms
    ax.plot(t, delta_ui, color="#1a3c6e", lw=2.6, ls="-", label=r"$\mathbf{Inlet\ Velocity\ Perturbation\ \delta u_i(t)}$", zorder=4)
    ax.plot(t, delta_dp, color="#dc2626", lw=2.4, ls="--", label=r"$\mathbf{Exit\ Pressure\ Feedback\ \delta \Delta P_{\mathrm{exit}}(t)}$", zorder=4)

    # 3. Transit Delay Guide Lines (At First Cycle: t=0 and t=2)
    ax.plot([0.0, 0.0], [-0.66, 0.46], color="#64748b", ls=":", lw=1.4, zorder=2)
    ax.plot([2.0, 2.0], [-0.66, 0.46], color="#64748b", ls=":", lw=1.4, zorder=2)

    # 4. Double-ended Delay Arrow from t=0.0 to t=2.0
    ax.annotate("", xy=(2.0, 0.44), xytext=(0.0, 0.44),
                arrowprops=dict(arrowstyle="<->", color="#b45309", lw=2.0), zorder=5)

    # 5. Delay Badge Text (Centered at x=2.0, completely clear of legend on the right)
    ax.text(2.0, 0.52, r"$\mathbf{\tau_{12} = T/2\ (180^\circ\ Phase\ Lag)}$",
            ha="center", va="bottom", fontsize=9.6, fontweight="bold", color="#92400e",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fffbeb", ec="#f59e0b", lw=1.1, alpha=0.96), zorder=6)

    # 6. Mechanism Equation Card
    mech_text = (r"$\mathbf{Self\text{-}Excited\ Feedback\ Loop:}$" + "\n" +
                 r"$\mathbf{\delta u_i(t)\ \rightarrow\ \delta \alpha_{\mathrm{exit}}(t)\ [\tau_{12}]\ \rightarrow\ \delta \Delta P_{\mathrm{exit}}(t)\ \rightarrow\ -\delta u_i(t)\ [180^\circ]}$" + "\n" +
                 r"$\mathbf{\text{Delayed exit vapor packet spikes pressure}\ \rightarrow\ \text{Further chokes inlet flow!}}$")
    ax.text(5.75, -0.56, mech_text, ha="center", va="center", fontsize=9.4, fontweight="bold", color="#1a3c6e",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#94a3b8", lw=1.1, alpha=0.96), zorder=7)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cbd5e1", fontsize=9.4)


# ==============================================================================
# PANEL (F): SUPERCRITICAL HOPF LIMIT-CYCLE ATTRACTOR (100% Exact from panel_f_hopf_orbit.py)
# ==============================================================================
def draw_panel_f(ax):
    ax.set_title(r"$\mathbf{(f)}$ Supercritical Hopf Limit-Cycle Attractor",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Inlet\ Liquid\ Velocity,\ u_i(t)\ (m/s)}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Boiling\ Boundary\ Position,\ \lambda(t)\ (m)}$", fontsize=12.0, fontweight="bold", labelpad=2)
    ax.set_xlim(0.06, 0.94)
    ax.set_ylim(0.16, 0.86)
    ax.grid(True)

    u_c, lam_c = 0.34, 0.56

    # 1. Closed Limit Cycle Orbit
    theta = np.linspace(0, 2 * np.pi, 300)
    u_orbit = u_c + 0.24 * np.cos(theta) - 0.04 * np.sin(2 * theta)
    lam_orbit = lam_c + 0.19 * np.sin(theta) + 0.04 * np.cos(2 * theta)

    # 2. Diverging Spiral from Focus
    t_spiral = np.linspace(0, 4.5 * np.pi, 350)
    r_spiral = 0.015 + 0.22 * (1.0 - np.exp(-0.25 * t_spiral))
    u_spiral = u_c + r_spiral * (np.cos(t_spiral) - 0.15 * np.sin(2 * t_spiral))
    lam_spiral = lam_c + 0.72 * r_spiral * (np.sin(t_spiral) + 0.15 * np.cos(2 * t_spiral))

    # Plot Spiral & Limit Cycle
    ax.plot(u_spiral, lam_spiral, color="#c88a10", lw=1.8, ls=":", label="Unstable Diverging Spiral", zorder=3)
    ax.plot(u_orbit, lam_orbit, color="#dc2626", lw=2.8, ls="-", label="Stable Limit Cycle", zorder=4)

    # Direction Arrows on Orbit
    for idx in [70, 220]:
        ax.annotate("", xy=(u_orbit[idx], lam_orbit[idx]), xytext=(u_orbit[idx - 5], lam_orbit[idx - 5]),
                    arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=2.0, mutation_scale=16), zorder=5)

    # Unstable Focus Point Marker
    ax.plot(u_c, lam_c, "x", color="#1e293b", ms=9.5, mew=2.2, label="Unstable Focus", zorder=6)

    # Inset Time Series (Clean frame, no ticks, bold arrow labels)
    inset_ax = ax.inset_axes([0.62, 0.12, 0.35, 0.31])
    t_sim = np.linspace(0, 22, 400)
    u_sim = u_c + 0.28 * np.cos(1.85 * t_sim) * (1.0 - np.exp(-0.32 * t_sim))
    inset_ax.plot(t_sim, u_sim, color="#dc2626", lw=2.0)
    inset_ax.set_title(r"$\mathbf{Sustained\ u_i(t)\ Oscillations}$", fontsize=10.5, pad=4, fontweight="bold")
    inset_ax.set_xlabel(r"$\mathbf{Time,\ t\ \rightarrow}$", fontsize=11.5, labelpad=3, fontweight="bold")
    inset_ax.set_ylabel(r"$\mathbf{u_i(t)\ \rightarrow}$", fontsize=11.5, labelpad=3, fontweight="bold")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    inset_ax.grid(True, alpha=0.3)
    inset_ax.set_facecolor("#ffffff")

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cbd5e1", fontsize=9.8)


# ==============================================================================
# MAIN COMPOSITE ASSEMBLY
# ==============================================================================
def generate_master_figure():
    plt.rcParams.update(MASTER_TYPOGRAPHY)

    fig = plt.figure(figsize=(15.2, 17.6), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)
    draw_panel_e(ax_e)
    draw_panel_f(ax_f)

    output_paths = [
        "d:/AGravity/Tide_Tutor/Figures/figure_1/fig1_multiscale_dynamics_master.png",
        "d:/AGravity/Tide_Tutor/Figures/figure_1/fig1_multiscale_dynamics_master.pdf",
        "d:/AGravity/Tide_Tutor/Figures/figure_1/fig1_multiscale_dynamics_master.svg",
        "d:/AGravity/Tide_Tutor/manuscript/figures/fig1_multiscale_dynamics.png",
        "d:/AGravity/Tide_Tutor/manuscript/figures/fig1_multiscale_dynamics.svg",
    ]

    for out_path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Successfully generated and saved: {out_path}")

    return fig


if __name__ == "__main__":
    generate_master_figure()