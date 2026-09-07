import sys
"""Master Figure 6: Remediation Frameworks & Architectural Mechanisms
Elsevier Reliability Engineering & System Safety (RE&SS)
Standardized 5-Panel Architecture (Panels a–e):
- Panel (a): Multi-Scale Random Fourier Feature Remediation (Tier a1: State Reconstruction, Tier a2: Local Error Impact)
- Panel (b): Analytical Log-Distance Singularity Coordinate Embedding psi(x)
- Panel (c): Misfit-Driven Adaptive Active Learning Allocation
- Panel (d): Inverse-Distance Boundary Loss Weighting Function w(g)
- Panel (e): 2D Spatial Boundary Loss Weight Field w(N_sub, N_pch)
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.patheffects as pe
import numpy as np
from scipy.ndimage import gaussian_filter1d

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
        "figsize": (16.2, 14.2),
        "dpi": 300,
    },
    "typography": {
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
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
EPSILON_PSI = 1e-3
EPSILON_W = 0.05

# ==============================================================================
# 1. DRAW PANEL (a) — STACKED SUBPLOTS (a1, a2)
# ==============================================================================
def draw_panel_a(fig, sub_spec, config=CONFIG):
    pal = config["palette"]
    x = np.linspace(-1.0, 1.0, 600)
    
    y_gt = np.where(x < 0, 0.45 * np.exp(2.2 * x) - 0.45, -1.25 * np.sqrt(np.maximum(0, x)) * np.exp(-0.75 * x))
    y_mlp = gaussian_filter1d(y_gt, sigma=55) - 0.05 * np.sin(np.pi * x)
    y_high = y_gt + 0.08 * np.sin(26 * np.pi * x) * (0.6 + 0.4 * np.cos(np.pi * x))
    y_multi = y_gt + 0.010 * np.sin(4 * np.pi * x) * np.exp(-2.5 * np.abs(x))

    err_mlp = np.abs(y_mlp - y_gt)
    err_high = np.abs(y_high - y_gt)
    err_multi = np.abs(y_multi - y_gt)

    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=sub_spec, height_ratios=[1.3, 1.0], hspace=0.22)
    ax0 = fig.add_subplot(inner_gs[0, 0])
    ax1 = fig.add_subplot(inner_gs[1, 0], sharex=ax0)

    # (a1) Reconstruction
    ax0.set_title(r"$\mathbf{(a1)\ Boundary\ State\ Reconstruction:\ Spectral\ Bias\ vs.\ Multi\text{-}Scale\ Fourier}$",
                  fontsize=11.8, fontweight="bold", loc="left", pad=8)
    ax0.axvline(0, color="#64748b", ls="--", lw=1.3, zorder=2)
    ax0.text(-0.03, 0.30, r"$\mathbf{Safety\ Boundary\ (Cusp\ at\ }x=0\mathbf{)}$",
             fontsize=10.0, color="#1e293b", fontweight="bold", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.22", fc="#f8fafc", ec="#cbd5e1", lw=0.9), zorder=6)

    ax0.plot(x, y_gt, color="black", lw=2.8, label=r"$\mathbf{Ground\ Truth\ Profile,\ }f(x)$", zorder=7)
    ax0.plot(x, y_mlp, color="#64748b", ls="--", lw=2.0, label=r"$\mathbf{Standard\ MLP\ (No\ Fourier)}$", zorder=3)
    ax0.plot(x, y_high, color=pal["gold"], ls=":", lw=1.8, label=r"$\mathbf{Single\text{-}Scale\ Fourier\ (}B=14\mathbf{)}$", zorder=4)
    ax0.plot(x, y_multi, color=pal["crimson"], ls="-", lw=2.2, label=r"$\mathbf{Multi\text{-}Scale\ Fourier}$", zorder=5)

    ax0.set_ylabel(r"$\mathbf{Surrogate\ State,\ }\hat{y}(x)$", fontsize=11.2, fontweight="bold")
    ax0.set_xlim(-1.02, 1.02)
    ax0.set_ylim(-1.30, 0.45)
    ax0.tick_params(labelsize=10.2, direction="in", length=3.0)
    ax0.grid(True, linestyle=":", alpha=0.35, zorder=1)

    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax0.text(0.98, 0.90, info_text, transform=ax0.transAxes,
             fontsize=10.0, color="#1e293b", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["border_gray"], lw=1.0), zorder=8)
    ax0.legend(loc="lower left", frameon=True, framealpha=0.96, facecolor="white",
               edgecolor=pal["border_gray"], fontsize=9.6, handlelength=2.0)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # (a2) Local Error
    ax1.set_title(r"$\mathbf{(a2)\ Local\ Prediction\ Error\ (Impact\ on\ Boundary\ Fidelity):}\ |\hat{y}(x) - f(x)|$",
                  fontsize=11.8, fontweight="bold", loc="left", pad=8)
    ax1.axvline(0, color="#64748b", ls="--", lw=1.3, zorder=2)
    ax1.fill_between(x, 0, err_mlp, color="#64748b", alpha=0.15, zorder=2)
    ax1.plot(x, err_mlp, color="#64748b", ls="--", lw=2.0, label=r"$\mathbf{Standard\ MLP}$", zorder=3)
    ax1.fill_between(x, 0, err_high, color=pal["gold"], alpha=0.12, zorder=2)
    ax1.plot(x, err_high, color=pal["gold"], ls=":", lw=1.8, label=r"$\mathbf{Single\text{-}Scale\ Fourier\ (}B=14\mathbf{)}$", zorder=4)
    ax1.fill_between(x, 0, err_multi, color=pal["crimson"], alpha=0.25, zorder=3)
    ax1.plot(x, err_multi, color=pal["crimson"], ls="-", lw=2.2, label=r"$\mathbf{Multi\text{-}Scale\ Fourier}$", zorder=5)

    ax1.set_xlabel(r"$\mathbf{Standardized\ Coordinate\ Across\ Boundary,\ }x_{\mathrm{norm}} \in [-1, 1]$",
                   fontsize=11.2, fontweight="bold", labelpad=5)
    ax1.set_ylabel(r"$\mathbf{Local\ Error,\ }\epsilon(x)$", fontsize=11.2, fontweight="bold")
    ax1.set_ylim(-0.01, 0.38)
    ax1.tick_params(labelsize=10.2, direction="in", length=3.0)
    ax1.grid(True, linestyle=":", alpha=0.35, zorder=1)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.96, facecolor="white",
               edgecolor=pal["border_gray"], fontsize=9.6, handlelength=2.0)

# ==============================================================================
# 2. DRAW PANEL (b) — LOG-DISTANCE EMBEDDING
# ==============================================================================
def draw_panel_b(ax, config=CONFIG):
    pal = config["palette"]
    nsub = np.linspace(10.5, 15.0, 250)
    npch = np.linspace(16.0, 25.5, 250)
    NS, NP = np.meshgrid(nsub, npch)
    r2 = (NS - NSUB_BT)**2 + (NP - NPCH_BT)**2
    psi = np.log(r2 + EPSILON_PSI)

    ax.set_title(r"$\mathbf{(b)\ Analytical\ Log\text{-}Distance\ Coordinate\ Embedding:\ }\psi(\mathbf{x}) = \ln(\|\mathbf{x} - \mathbf{x}_{\mathrm{BT}}\|^2 + \varepsilon)$",
                 fontsize=11.8, fontweight="bold", loc="left", pad=10)

    levels = np.linspace(-6.0, 3.5, 40)
    cf = ax.contourf(NS, NP, psi, levels=levels, cmap="viridis", extend="both")
    iso_levels = [-4.0, -2.0, 0.0, 1.5, 3.0]
    cs = ax.contour(NS, NP, psi, levels=iso_levels, colors="white", alpha=0.6, linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=8.8, fmt=r"$\psi=%.1f$", manual=False)

    cbar = plt.colorbar(cf, ax=ax, pad=0.025, aspect=20, shrink=0.96, ticks=[-6.0, -4.0, -2.0, 0.0, 2.0])
    cbar.set_label(r"$\mathbf{Log\text{-}Distance\ Prior,\ }\psi(\mathbf{x})$", fontsize=10.5, fontweight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=10.0)

    ax.plot(NSUB_BT, NPCH_BT, marker="*", color="#fbbf24", mec="#1e293b", mew=1.2, ms=16, zorder=6,
            label=r"$\mathbf{BT\ Singularity\ Vertex\ (}\mathbf{x}_{\mathrm{BT}}\mathbf{)}$")

    annot_text = (
        r"$\mathbf{Singular\ Distance\ Funnel}$" + "\n" +
        r"$\bullet\ \mathbf{x}_{\mathrm{BT}} = (14.14,\ 20.59)$" + "\n" +
        r"$\bullet\ \psi(\mathbf{x}) \to \ln(\varepsilon)\ \mathrm{at\ vertex}$" + "\n" +
        r"$\bullet\ \mathrm{Radial\ geometric\ prior}$"
    )
    ax.annotate(
        annot_text, xy=(NSUB_BT, NPCH_BT), xytext=(11.5, 23.2),
        fontsize=9.6, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["gold"], lw=1.2),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.6, shrinkA=3, shrinkB=5),
        zorder=7
    )

    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.03, 0.04, info_text, transform=ax.transAxes,
            fontsize=9.8, color="#1e293b", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["border_gray"], lw=1.0), zorder=8)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_ylabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_xlim(10.5, 15.0)
    ax.set_ylim(16.0, 25.5)
    ax.tick_params(labelsize=10.2, direction="in", length=3.0)
    ax.grid(True, linestyle=":", alpha=0.22, color="white", zorder=1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=9.6)

# ==============================================================================
# 3. DRAW PANEL (c) — ADAPTIVE ACTIVE LEARNING
# ==============================================================================
def draw_panel_c(ax, config=CONFIG):
    pal = config["palette"]
    np.random.seed(42)
    
    nsub_init = np.random.uniform(10.6, 14.9, 90)
    npch_init = np.random.uniform(16.3, 25.2, 90)
    
    N_adapt = 150
    nsub_adapt = np.random.uniform(11.3, NSUB_BT, N_adapt)
    d_adapt = NSUB_BT - nsub_adapt
    branch = np.random.choice([0, 1], size=N_adapt, p=[0.5, 0.5])
    npch_adapt = np.where(
        branch == 0,
        NPCH_BT - 1.45 * d_adapt + np.random.normal(0, 0.20, N_adapt),
        NPCH_BT - 0.42 * d_adapt + np.random.normal(0, 0.20, N_adapt)
    )

    ax.set_title(r"$\mathbf{(c)\ Misfit\text{-}Driven\ Adaptive\ Active\ Learning\ Allocation:\ }\mathbf{x}_{k+1} = \arg\max\ [U(\mathbf{x}) \cdot \mathbf{I}(|g| \leq \delta_w)]$",
                 fontsize=11.8, fontweight="bold", loc="left", pad=10)

    ns_curve = np.linspace(10.5, NSUB_BT, 300)
    d_c = NSUB_BT - ns_curve
    npch_fold = NPCH_BT - 1.45 * d_c
    npch_hopf = NPCH_BT - 0.42 * d_c

    ax.fill_between(ns_curve, npch_fold - 0.50, npch_fold + 0.50, color="#cbd5e1", alpha=0.35, zorder=1)
    ax.fill_between(ns_curve, npch_hopf - 0.50, npch_hopf + 0.50, color="#cbd5e1", alpha=0.35, zorder=1)
    ax.fill_between(ns_curve, npch_fold, npch_hopf, color="#f1f5f9", alpha=0.4, zorder=1)

    ax.plot(ns_curve, npch_fold, color=pal["navy"], lw=2.0, ls="--", zorder=3, label=r"$\mathbf{Lower\ Fold\ Boundary\ (}g=0\mathbf{)}$")
    ax.plot(ns_curve, npch_hopf, color=pal["teal"], lw=2.0, ls="-.", zorder=3, label=r"$\mathbf{Upper\ Hopf\ Boundary\ (}g=0\mathbf{)}$")

    ax.scatter(nsub_init, npch_init, s=28, color=pal["slate"], alpha=0.65, marker="o",
               edgecolors="white", linewidths=0.5, zorder=4, label=r"$\mathbf{Initial\ LHS\ Points}$")
    ax.scatter(nsub_adapt, npch_adapt, s=40, color=pal["crimson"], alpha=0.90, marker="o",
               edgecolors="black", linewidths=0.6, zorder=5, label=r"$\mathbf{Adaptive\ Active\ Points}$")

    ax.plot(NSUB_BT, NPCH_BT, marker="*", color="#fbbf24", mec="#1e293b", mew=1.2, ms=16, zorder=6,
            label=r"$\mathbf{BT\ Singularity\ Vertex\ (}\mathbf{x}_{\mathrm{BT}}\mathbf{)}$")

    annot_text = (
        r"$\mathbf{Boundary\ Corridor\ Surge}$" + "\n" +
        r"$\bullet\ \mathrm{High\ epistemic\ uncertainty\ }U(\mathbf{x})$" + "\n" +
        r"$\bullet\ \mathrm{Samples\ concentrated\ along\ }\partial\Omega$" + "\n" +
        r"$\bullet\ \mathrm{Resolves\ sharp\ cusp\ transition}$"
    )
    ax.annotate(
        annot_text, xy=(13.6, 19.8), xytext=(13.1, 16.5),
        fontsize=9.6, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["crimson"], lw=1.2),
        arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.5, shrinkA=3, shrinkB=5),
        zorder=7
    )

    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.97, 0.95, info_text, transform=ax.transAxes,
            fontsize=9.8, color="#1e293b", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["border_gray"], lw=1.0), zorder=8)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_ylabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_xlim(10.5, 15.0)
    ax.set_ylim(16.0, 25.5)
    ax.tick_params(labelsize=10.2, direction="in", length=3.0)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=1)
    ax.legend(loc="upper left", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=9.4, handlelength=1.8)

# ==============================================================================
# 4. DRAW PANEL (d) — 1D LOSS WEIGHT FUNCTION
# ==============================================================================
def draw_panel_d(ax, config=CONFIG):
    pal = config["palette"]
    g_vals = np.linspace(-1.5, 1.5, 601)
    epsilons = [0.02, 0.05, 0.10, 0.25]
    colors = [pal["crimson"], pal["gold"], pal["teal"], pal["slate"]]
    linestyles = ["-", "--", "-.", ":"]
    linewidths = [2.4, 1.9, 1.8, 1.7]

    ax.set_title(r"$\mathbf{(d)\ Inverse\text{-}Distance\ Boundary\ Loss\ Weighting:\ }w(g) = \frac{1}{|g| + \varepsilon_w}$",
                 fontsize=11.8, fontweight="bold", loc="left", pad=10)

    ax.axvline(0, color="#475569", ls="--", lw=1.3, zorder=2, label=r"$\mathbf{Safety\ Boundary\ (}g=0\mathbf{)}$")
    w_opt = 1.0 / (np.abs(g_vals) + 0.02)
    ax.fill_between(g_vals, 0, w_opt, color=pal["crimson"], alpha=0.10, zorder=1)

    for eps, col, ls, lw in zip(epsilons, colors, linestyles, linewidths):
        w = 1.0 / (np.abs(g_vals) + eps)
        ax.plot(g_vals, w, color=col, ls=ls, lw=lw, zorder=4,
                label=rf"$\mathbf{{Regularizer\ }}\varepsilon_w = {eps:.2f}$")

    annot_text = (
        r"$\mathbf{Hyperbolic\ Gradient\ Surge}$" + "\n" +
        r"$\bullet\ w_{\max} = 1/\varepsilon_w = 50.0\ \mathrm{at\ boundary}$" + "\n" +
        r"$\bullet\ \mathrm{Focuses\ backpropagation\ on\ }g=0$" + "\n" +
        r"$\bullet\ \mathrm{Decays\ to\ }w \approx 1.0\ \mathrm{in\ far\ field}$"
    )
    ax.annotate(
        annot_text, xy=(0.0, 50.0), xytext=(-1.38, 22.0),
        fontsize=9.6, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["crimson"], lw=1.2),
        arrowprops=dict(arrowstyle="->", color=pal["crimson"], lw=1.5, shrinkA=3, shrinkB=5),
        zorder=7
    )

    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.04, 0.95, info_text, transform=ax.transAxes,
            fontsize=9.8, color="#1e293b", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["border_gray"], lw=1.0), zorder=8)

    ax.set_xlabel(r"$\mathbf{Limit\text{-}State\ Margin,\ }g(\mathbf{x})$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_ylabel(r"$\mathbf{Loss\ Weight\ Multiplier,\ }w(g)$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 56)
    ax.tick_params(labelsize=10.2, direction="in", length=3.0)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=9.4, handlelength=2.0)

# ==============================================================================
# 5. DRAW PANEL (e) — 2D SPATIAL LOSS WEIGHT FIELD
# ==============================================================================
def draw_panel_e(ax, config=CONFIG):
    pal = config["palette"]
    nsub = np.linspace(10.5, 15.0, 250)
    npch = np.linspace(16.0, 25.5, 250)
    NS, NP = np.meshgrid(nsub, npch)
    
    d = np.maximum(0, NSUB_BT - NS)
    lower_b = NPCH_BT - 1.45 * d
    upper_b = NPCH_BT - 0.42 * d
    
    dist_lower = np.abs(NP - lower_b)
    dist_upper = np.abs(NP - upper_b)
    r_bt = np.sqrt((NS - NSUB_BT)**2 + (NP - NPCH_BT)**2)
    dist_boundary = np.where(NS <= NSUB_BT, np.minimum(dist_lower, dist_upper), r_bt)
    w_field = 1.0 / (dist_boundary + EPSILON_W)

    ax.set_title(r"$\mathbf{(e)\ 2D\ Spatial\ Boundary\ Loss\ Weight\ Field:\ }w(N_{\mathrm{sub}}, N_{\mathrm{pch}})$",
                 fontsize=11.8, fontweight="bold", loc="left", pad=10)

    levels = np.linspace(0.05, 20.0, 50)
    cf = ax.contourf(NS, NP, w_field, levels=levels, cmap="plasma", extend="max")
    
    iso_levels = [2.0, 5.0, 10.0, 15.0]
    cs = ax.contour(NS, NP, w_field, levels=iso_levels, colors="white", alpha=0.45, linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8.8, fmt=r"$w=%.0f$", manual=False)

    cbar = plt.colorbar(cf, ax=ax, pad=0.025, aspect=20, shrink=0.96, ticks=[2.0, 5.0, 10.0, 15.0, 20.0])
    cbar.set_label(r"$\mathbf{Loss\ Weight\ Multiplier,\ }w(\mathbf{x})$", fontsize=10.5, fontweight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=10.0)

    ns_curve = np.linspace(10.5, NSUB_BT, 300)
    d_c = NSUB_BT - ns_curve
    npch_fold = NPCH_BT - 1.45 * d_c
    npch_hopf = NPCH_BT - 0.42 * d_c

    stroke = [pe.Stroke(linewidth=3.0, foreground="#0f172a"), pe.Normal()]
    ax.plot(ns_curve, npch_fold, color="#ffffff", lw=2.0, ls="--", zorder=3,
            path_effects=stroke, label=r"$\mathbf{Lower\ Fold\ Boundary\ (}g=0\mathbf{)}$")
    ax.plot(ns_curve, npch_hopf, color="#38bdf8", lw=2.0, ls="-.", zorder=3,
            path_effects=stroke, label=r"$\mathbf{Upper\ Hopf\ Boundary\ (}g=0\mathbf{)}$")

    ax.plot(NSUB_BT, NPCH_BT, marker="*", color="#fbbf24", mec="#0f172a", mew=1.2, ms=16, zorder=6,
            label=r"$\mathbf{BT\ Singularity\ Vertex\ (}\mathbf{x}_{\mathrm{BT}}\mathbf{)}$")

    annot_text = (
        r"$\mathbf{Targeted\ Loss\ Ridge}$" + "\n" +
        r"$\bullet\ w_{\max} = 1/\varepsilon_w = 20.0\ \mathrm{along\ }\partial\Omega$" + "\n" +
        r"$\bullet\ \mathrm{Automatic\ spatial\ lens\ along\ }g=0$" + "\n" +
        r"$\bullet\ \mathrm{Focuses\ surrogate\ capacity}$"
    )
    ax.annotate(
        annot_text, xy=(12.4, 19.8), xytext=(10.8, 22.8),
        fontsize=9.6, color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["gold"], lw=1.2),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.6, shrinkA=3, shrinkB=5),
        zorder=7
    )

    info_text = (
        r"$\mathbf{Operating\ Conditions}$:" + "\n" +
        r"$\bullet\ \Lambda = 4.0,\ Fr = 0.01$" + "\n" +
        r"$\bullet\ k_{\mathrm{in}} = 1.0,\ k_{\mathrm{ex}} = 1.5$"
    )
    ax.text(0.04, 0.04, info_text, transform=ax.transAxes,
            fontsize=9.8, color="#1e293b", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.30", fc="#ffffff", ec=pal["border_gray"], lw=1.0), zorder=8)

    ax.set_xlabel(r"$\mathbf{Subcooling\ Number,\ }N_{\mathrm{sub}}$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_ylabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=11.2, fontweight="bold", labelpad=5)
    ax.set_xlim(10.5, 15.0)
    ax.set_ylim(16.0, 25.5)
    ax.tick_params(labelsize=10.2, direction="in", length=3.0)
    ax.grid(True, linestyle=":", alpha=0.18, color="white", zorder=1)
    ax.legend(loc="lower right", frameon=True, framealpha=0.96, facecolor="white",
              edgecolor=pal["border_gray"], fontsize=9.4, handlelength=2.0)

# ==============================================================================
# MASTER MULTI-PANEL ASSEMBLY
# ==============================================================================
def assemble_master_figure_6():
    plt.rcParams.update(CONFIG["typography"])
    fig = plt.figure(figsize=CONFIG["figure"]["figsize"], dpi=CONFIG["figure"]["dpi"])

    # Outer Grid: 2 columns
    # Column 0: Panel (a) [top] and Panel (c) [bottom]
    # Column 1: Panel (b) [top], Panel (d) [middle], Panel (e) [bottom]
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.18,
                                 left=0.06, right=0.98, top=0.96, bottom=0.05)

    # Column 0 Grid: 2 rows (Panel a height = 1.35, Panel c height = 1.0)
    col0_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, 0],
                                               height_ratios=[1.35, 1.0], hspace=0.25)
    draw_panel_a(fig, col0_gs[0, 0], config=CONFIG)
    ax_c = fig.add_subplot(col0_gs[1, 0])
    draw_panel_c(ax_c, config=CONFIG)

    # Column 1 Grid: 3 rows (Panel b, Panel d, Panel e)
    col1_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0, 1],
                                               height_ratios=[1.0, 0.85, 1.0], hspace=0.28)
    ax_b = fig.add_subplot(col1_gs[0, 0])
    draw_panel_b(ax_b, config=CONFIG)

    ax_d = fig.add_subplot(col1_gs[1, 0])
    draw_panel_d(ax_d, config=CONFIG)

    ax_e = fig.add_subplot(col1_gs[2, 0])
    draw_panel_e(ax_e, config=CONFIG)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    out_png_local = os.path.join(output_dir, "fig6_remediation_taxonomy_master.png")
    out_pdf_local = os.path.join(output_dir, "fig6_remediation_taxonomy_master.pdf")
    
    ms_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "manuscript", "figures"))
    os.makedirs(ms_dir, exist_ok=True)
    out_png_ms = os.path.join(ms_dir, "fig6_remediation_taxonomy.png")
    out_pdf_ms = os.path.join(ms_dir, "fig6_remediation_taxonomy.pdf")

    fig.savefig(out_png_local, dpi=300)
    fig.savefig(out_pdf_local)
    fig.savefig(out_png_ms, dpi=300)
    fig.savefig(out_pdf_ms)
    plt.close(fig)

    print(f"[SUCCESS] Master Figure 6 generated and saved to:")
    print(f"  - {out_png_local}")
    print(f"  - {out_pdf_local}")
    print(f"  - {out_png_ms}")
    print(f"  - {out_pdf_ms}")

if __name__ == "__main__":
    assemble_master_figure_6()
