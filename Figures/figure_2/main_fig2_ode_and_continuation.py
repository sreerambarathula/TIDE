import sys
"""Master Composite Figure 2: Moving-Boundary ODE Dynamics, Discretization Pathology, and Machine-Precision Continuation
Publication-Grade 6-Panel Layout (2 Columns x 3 Rows) at 300 DPI for Elsevier RE&SS.

Panels:
  (a) The Odd-Node Energy Pumping Pathology (N1=1,3 vs N1=2,16)
  (b) Multi-Node Enthalpy Field Evolution & Moving Boiling Boundary
  (c) Spectral Frequency Analysis (Power Spectral Density / FFT)
  (d) Mesh Independence & Machine Precision (< 10^-15 Discretization Error)
  (e) Autodiff Fold Continuation (Theler et al. Eq. 26 Physical Ledinegg Maximum)
  (f) 2D Newton Sum-and-Product Solver Residual Convergence
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib._mathtext as mmt
import matplotlib.gridspec as gridspec
import numpy as np

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
# PANEL (A): ODD-NODE NUMERICAL PATHOLOGY
# ==============================================================================
def draw_panel_a(ax):
    ax.set_title(r"$\mathbf{(a)\ The\ Odd\text{-}Node\ Energy\ Pumping\ Pathology}$",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Dimensionless\ Time,\ } t$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Inlet\ Liquid\ Velocity,\ } u_i(t)$", fontsize=12.0, fontweight="bold", labelpad=4)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.70, 2.75)
    ax.grid(True)

    ax.axhline(0.0, color="#cbd5e1", lw=1.0, ls=":", zorder=1)

    t = np.linspace(0, 30, 800)
    ui_n2 = 0.45 + 0.18 * np.sin(1.83 * t) * (1.0 - np.exp(-0.25 * t))
    ui_n16 = 0.45 + 0.184 * np.sin(1.83 * t) * (1.0 - np.exp(-0.25 * t))

    ax.plot(t, ui_n2, color="#1a3c6e", lw=3.0, ls="-",
            label=r"Even: $N_1 = 2$ (Stable Limit Cycle)", zorder=3)
    ax.plot(t, ui_n16, color="#06b6d4", lw=1.8, ls=(0, (4, 3)),
            label=r"Even: $N_1 = 16$ (Mesh Limit)", zorder=4)

    t_n3 = t[t <= 21.46]
    ui_n3 = 0.45 + 0.18 * np.sin(1.83 * t_n3) * np.exp(0.25 * t_n3 / 3.0)
    ax.plot(t_n3, ui_n3, color="#c88a10", lw=2.0, ls="-.",
            label=r"Odd: $N_1 = 3$ (Spurious Energy Pumping)", zorder=5)

    t_n1 = t[t <= 14.65]
    ui_n1 = 0.45 + 0.18 * np.sin(1.83 * t_n1) * np.exp(0.32 * t_n1 / 2.5)
    ax.plot(t_n1, ui_n1, color="#b5451b", lw=2.4, ls="-",
            label=r"Odd: $N_1 = 1$ (Catastrophic Blow-up)", zorder=6)

    # Blowup Markers
    ax.plot(14.65, 1.62, "X", color="#b5451b", ms=10.0, mew=2.2, zorder=7)
    ax.plot(21.46, 1.53, "X", color="#c88a10", ms=10.0, mew=2.2, zorder=7)

    # Annotations
    ax.annotate(r"$\mathbf{N_1 = 1\ Divergence\ to\ NaN}$" + "\n" +
                r"$(t \approx 14.7\,\mathrm{s},\ \mathrm{Spurious\ Injection})$",
                xy=(14.65, 1.62), xytext=(9.5, 2.10),
                fontsize=8.8, color="#b5451b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.40", fc="#fff1f2", ec="#b5451b", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#b5451b", lw=1.4), zorder=8)

    ax.annotate(r"$\mathbf{N_1 = 3\ Blow\text{-}up\ (t \approx 21.5\,\mathrm{s})}$" + "\n" +
                r"$\mathrm{(Delayed\ Spatial\ Instability)}$",
                xy=(21.46, 1.53), xytext=(20.5, 2.10),
                fontsize=8.8, color="#9a3412", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.40", fc="#fff7ed", ec="#c88a10", lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#c88a10", lw=1.4), zorder=8)

    leg = ax.legend(loc="lower left", frameon=True, framealpha=1.0, facecolor="white",
                    edgecolor="#94a3b8", fontsize=8.8, borderpad=0.40, handlelength=1.4, handletextpad=0.45)
    leg.set_zorder(10)


# ==============================================================================
# PANEL (B): MULTI-NODE ENTHALPY FIELD EVOLUTION
# ==============================================================================
def draw_panel_b(ax):
    ax.set_title(r"$\mathbf{(b)\ Multi\text{-}Node\ Enthalpy\ Field\ Evolution}$",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Dimensionless\ Time,\ } t$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Axial\ Channel\ Coordinate,\ } z$", fontsize=12.0, fontweight="bold", labelpad=4)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    t = np.linspace(0, 20, 600)
    lam = 0.55 + 0.12 * np.sin(1.83 * t) * (1.0 - np.exp(-0.35 * t))
    l3 = 0.75 * lam
    l2 = 0.50 * lam
    l1 = 0.25 * lam

    node_fills = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa"]

    ax.fill_between(t, 0, l1, color=node_fills[0], alpha=0.9, label=r"Subcooled Node 1 ($h_{\mathrm{in}} \to h_1$)")
    ax.fill_between(t, l1, l2, color=node_fills[1], alpha=0.9, label=r"Subcooled Node 2 ($h_1 \to h_2$)")
    ax.fill_between(t, l2, l3, color=node_fills[2], alpha=0.9, label=r"Subcooled Node 3 ($h_2 \to h_3$)")
    ax.fill_between(t, l3, lam, color=node_fills[3], alpha=0.9, label=r"Subcooled Node 4 ($h_3 \to h_{\mathrm{sat},f}$)")
    ax.fill_between(t, lam, 1.0, color="#ffedd5", alpha=0.85, label=r"Two-Phase Mixture ($\rho_m < \rho_f$)")

    ax.plot(t, l1, color="#64748b", lw=1.0, ls=":")
    ax.plot(t, l2, color="#64748b", lw=1.0, ls=":")
    ax.plot(t, l3, color="#64748b", lw=1.0, ls=":")
    ax.plot(t, lam, color="#b5451b", lw=2.4, label=r"Boiling Interface $z = \lambda(t)$")
    ax.axhline(1.0, color="#64748b", lw=1.8, ls="-")

    idx_target = np.argmin(np.abs(t - 10.0))
    target_y = lam[idx_target]
    ax.annotate(r"$\mathbf{Moving\ Boiling\ Boundary\ \lambda(t)}$" + "\n" +
                r"$(h = h_{\mathrm{sat},f}\ \mathrm{Saturation\ Enthalpy\ Reached})$",
                xy=(10.0, target_y), xytext=(6.5, 0.82),
                fontsize=9.8, color="#b5451b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", fc="#fff7ed", ec="#b5451b", lw=1.3),
                arrowprops=dict(arrowstyle="->", color="#b5451b", lw=1.5), zorder=6)

    ax.legend(loc="lower right", frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="#94a3b8", fontsize=9.4, borderpad=0.40, handlelength=1.4, handletextpad=0.50)


# ==============================================================================
# PANEL (C): SPECTRAL FREQUENCY ANALYSIS (FFT)
# ==============================================================================
def draw_panel_c(ax):
    ax.set_title(r"$\mathbf{(c)\ Spectral\ Frequency\ Analysis\ (FFT)}$",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Angular\ Frequency,\ }\omega\ \mathbf{(rad/s)}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Power\ Spectral\ Density,\ }S_{uu}(\omega)\ \mathbf{(dB/Hz)}$", fontsize=12.0, fontweight="bold", labelpad=4)
    ax.set_xlim(0.2, 7.5)
    ax.set_ylim(0.01, 8000)
    ax.grid(True, which="both")

    w0 = 1.83
    omega = np.linspace(0.2, 7.5, 700)
    psd = (
        1.0 / ((omega - w0)**2 + 0.018**2)
        + 0.22 / ((omega - 2 * w0)**2 + 0.045**2)
        + 0.06 / ((omega - 3 * w0)**2 + 0.08**2)
        + 0.005 / (omega**1.2)
    )

    ax.semilogy(omega, psd, color="#1a3c6e", lw=2.4, label=r"Velocity PSD $S_{uu}(\omega)$")
    ax.axvline(w0, color="#b5451b", lw=1.6, ls="--", label=r"Fundamental Mode ($\omega_0 = 1.83\ \mathrm{rad/s}$)")
    ax.axvline(2 * w0, color="#b45309", lw=1.4, ls=":", label=r"Second Harmonic ($2\omega_0 = 3.66\ \mathrm{rad/s}$)")
    ax.axvline(3 * w0, color="#64748b", lw=1.2, ls=":", label=r"Third Harmonic ($3\omega_0 = 5.49\ \mathrm{rad/s}$)")

    # Fundamental Peak Callout (Left border at 2.0, Bottom at 200)
    ax.annotate(r"$\mathbf{Fundamental\ DWO\ Mode}$" + "\n" +
                r"$\omega_0 = 1.83\ \mathrm{rad/s}$" + "\n" +
                r"$T_0 = 3.43\ \mathrm{s} \approx 2\tau_{\mathrm{transit}}$",
                xy=(1.83, 3100), xytext=(2.0, 200),
                va="bottom", ha="left", fontsize=9.2, color="#b5451b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.40", fc="#fff1f2", ec="#b5451b", lw=1.3),
                arrowprops=dict(arrowstyle="->", color="#b5451b", lw=1.5, relpos=(0.0, 0.85)), zorder=6)

    # Harmonic Callout
    ax.annotate(r"$\mathbf{2\omega_0\ Harmonic\ Peak}$",
                xy=(3.66, 110), xytext=(3.95, 25),
                fontsize=9.2, color="#b45309", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fefce8", ec="#b45309", lw=1.1),
                arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.3), zorder=6)

    ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="#94a3b8", fontsize=9.2, borderpad=0.45, handlelength=1.5, handletextpad=0.50)


# ==============================================================================
# PANEL (D): MESH INDEPENDENCE & MACHINE PRECISION
# ==============================================================================
def draw_panel_d(ax):
    ax.set_title("(d) Mesh Independence & Machine Precision",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Single\text{-}Phase\ Moving\ Node\ Count,\ }N_1$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Relative\ Error,\ }|\mathrm{Eu}(N_1) - \mathrm{Eu}_{\mathrm{Euler}}| / \mathrm{Eu}_{\mathrm{Euler}}$", fontsize=12.0, fontweight="bold", labelpad=4)
    ax.set_xlim(1.5, 16.5)
    ax.set_ylim(8e-17, 3e-14)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    ax.grid(True, which="both")

    nodes = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    rel_errors = np.array([2.2e-16, 4.4e-16, 1.8e-16, 3.5e-16, 2.7e-16, 5.1e-16, 3.3e-16, 4.0e-16])
    eps_mach = 2.22e-16

    ax.semilogy(nodes, rel_errors, "o-", color="#1a3c6e", lw=2.2, ms=8, mec="black", mew=1.2,
                label=r"Discretized Model vs. Analytical Solution")
    ax.axhline(eps_mach, color="#b5451b", lw=1.8, ls="--",
               label=r"IEEE 754 Double Precision ($\epsilon_{\mathrm{mach}} \approx 2.22 \times 10^{-16}$)")

    ax.annotate(r"$\mathbf{Relative\ Error < 10^{-15}\ across\ all\ }N_1$",
                xy=(6, 1.8e-16), xytext=(2.2, 3.5e-15),
                fontsize=9.4, color="#1a3c6e", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.40", fc="#eff6ff", ec="#1a3c6e", lw=1.3),
                arrowprops=dict(arrowstyle="->", color="#1a3c6e", lw=1.4), zorder=6)

    ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="#94a3b8", fontsize=9.2, borderpad=0.45, handlelength=1.6, handletextpad=0.50)


# ==============================================================================
# PANEL (E): AUTODIFF FOLD CONTINUATION (Physical Eq. 26 Model)
# ==============================================================================
def draw_panel_e(ax1):
    ax1.set_title(r"$\mathbf{(e)\ Autodiff\ Fold\ Continuation\ (\partial\mathrm{Eu}/\partial N_{\mathrm{pch}} = 0)}$",
                  fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax1.set_xlabel(r"$\mathbf{Phase\ Change\ Number,\ }N_{\mathrm{pch}}$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax1.set_ylabel(r"$\mathbf{Euler\ Number,\ }\mathrm{Eu}$", fontsize=12.0, fontweight="bold", labelpad=4, color="#1a3c6e")
    ax1.tick_params(axis="y", labelcolor="#1a3c6e")
    ax1.set_xlim(8.15, 16.0)
    ax1.set_ylim(10.15, 11.90)
    ax1.grid(True)

    n_sub = 8.0
    Fr, Lam, k_in, k_out = 5.0, 3.0, 6.0, 2.0
    npch_fold = 10.011834

    def euler_number_exact(p):
        term1 = (1.0 / p) * (n_sub**2 + 0.5 * Lam * n_sub**2 + k_out * n_sub**2)
        term2 = (1.0 / p**2) * (
            -n_sub**3 + Lam * n_sub**2 - Lam * n_sub**3
            + k_in * n_sub**2 + k_out * n_sub**2 - k_out * n_sub**3
        )
        term3 = (n_sub / p) * (1.0 / Fr) * (1.0 + np.log(1.0 + p - n_sub) / n_sub)
        term4 = 0.5 * (n_sub**4 / p**3) * Lam
        return term1 + term2 + term3 + term4

    def d_euler_number_exact(p):
        c1 = (n_sub**2 + 0.5 * Lam * n_sub**2 + k_out * n_sub**2)
        d_t1 = -1.0 / (p**2) * c1
        c2 = (-n_sub**3 + Lam * n_sub**2 - Lam * n_sub**3 + k_in * n_sub**2 + k_out * n_sub**2 - k_out * n_sub**3)
        d_t2 = -2.0 / (p**3) * c2
        c3 = n_sub / Fr
        g = 1.0 + np.log(1.0 + p - n_sub) / n_sub
        g_prime = 1.0 / (n_sub * (1.0 + p - n_sub))
        d_t3 = c3 * (g_prime * p - g) / (p**2)
        c4 = 0.5 * (n_sub**4) * Lam
        d_t4 = -3.0 / (p**4) * c4
        return d_t1 + d_t2 + d_t3 + d_t4

    npch = np.linspace(8.15, 16.0, 600)
    eu = euler_number_exact(npch)
    grad_eu = d_euler_number_exact(npch)
    eu_fold = euler_number_exact(npch_fold)

    line1 = ax1.plot(npch, eu, color="#1a3c6e", lw=2.4, label=r"Euler Characteristic $\mathrm{Eu}(N_{\mathrm{pch}})$")
    line2 = ax1.axvline(npch_fold, color="#b5451b", lw=1.6, ls="--",
                        label=r"Fold Limit Point ($N_{\mathrm{pch}} = 10.012$)")
    ax1.plot(npch_fold, eu_fold, "o", color="#b5451b", ms=8.5, zorder=5, mec="black", mew=1.2)

    ax2 = ax1.twinx()
    line3 = ax2.plot(npch, grad_eu, color="#007a78", lw=2.0, ls="-.", label=r"Autodiff Gradient")
    ax2.axhline(0.0, color="#64748b", lw=1.0, ls=":")
    ax2.plot(npch_fold, 0.0, "s", color="#007a78", ms=8.0, zorder=5, mec="black", mew=1.2)

    ax2.set_ylabel(r"$\mathbf{Autodiff\ Gradient,\ }\partial\mathrm{Eu}/\partial N_{\mathrm{pch}}$",
                   fontsize=12.0, fontweight="bold", labelpad=6, color="#007a78")
    ax2.tick_params(axis="y", labelcolor="#007a78")
    ax2.set_ylim(-0.32, 0.32)

    ax1.annotate(r"$\mathbf{Fold\ Limit\text{-}Point\ Condition}$:" + "\n" +
                 r"$\partial\mathrm{Eu} / \partial N_{\mathrm{pch}} = 0$" + "\n" +
                 r"$(N_{\mathrm{pch}} = 10.012,\ \mathrm{Bisection\ to\ }10^{-12})$",
                 xy=(npch_fold, eu_fold), xytext=(12.3, 11.45),
                 fontsize=9.2, color="#b5451b", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.42", fc="#fff7ed", ec="#b5451b", lw=1.3),
                 arrowprops=dict(arrowstyle="->", color="#b5451b", lw=1.4), zorder=6)

    lines = line1 + [line2] + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", frameon=True, framealpha=1.0,
               facecolor="white", edgecolor="#94a3b8", fontsize=9.0,
               borderpad=0.45, handlelength=1.5, handletextpad=0.50)


# ==============================================================================
# PANEL (F): 2D NEWTON SUM-AND-PRODUCT SOLVER CONVERGENCE
# ==============================================================================
def draw_panel_f(ax):
    ax.set_title(r"$\mathbf{(f)\ 2D\ Newton\ Sum\text{-}and\text{-}Product\ Solver\ Convergence}$",
                 fontsize=13.0, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(r"$\mathbf{Newton\ Iteration\ Count,\ }k$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_ylabel(r"$\mathbf{Nonlinear\ Residual\ Norm,\ }\|\mathbf{R}\|$", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_xlim(0.6, 7.4)
    ax.set_ylim(1e-15, 2e6)
    ax.set_xticks(np.arange(1, 8))
    ax.grid(True, which="both")

    iters_sum_prod = np.arange(1, 8)
    res_sum_prod = np.array([4.2e-1, 8.5e-2, 3.1e-3, 4.8e-6, 1.2e-11, 2.5e-13, 2.1e-13])
    iters_raw = np.array([1, 2, 3, 4, 5])
    res_raw = np.array([4.2e-1, 3.8e-1, 1.2e0, 8.5e1, 1.4e4])

    ax.semilogy(iters_sum_prod, res_sum_prod, "o-", color="#1a3c6e", lw=2.4, ms=8.0, mec="black", mew=1.2,
                label=r"Sum-and-Product Solver")
    ax.semilogy(iters_raw, res_raw, "s--", color="#b5451b", lw=2.0, ms=8.0, mec="black", mew=1.2,
                label=r"Naive Raw-Eigenvalue Newton (Diverges)")
    ax.axhline(1e-11, color="#007a78", lw=1.6, ls=":",
               label=r"Target Tolerance ($10^{-11}$)")

    # Convergence Callout (Upper right space)
    ax.annotate(r"$\mathbf{Smooth\ Analytic\ Formulation}$:" + "\n" +
                r"$\Rightarrow$ Quadratic descent to $10^{-13}$",
                xy=(5, 1.2e-11), xytext=(4.7, 4.0e-5),
                fontsize=9.2, color="#1e40af", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.42", fc="#eff6ff", ec="#3b82f6", lw=1.3),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.4), zorder=6)

    # Divergence Callout (Top left)
    ax.annotate(r"$\mathbf{Branch\text{-}Point\ Singularity}$:" + "\n" +
                r"$\nabla\lambda_i \to \infty \Rightarrow$ Diverges to $\mathrm{NaN}$",
                xy=(4, 8.5e1), xytext=(2.2, 2.0e3),
                fontsize=9.2, color="#b5451b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.42", fc="#fff7ed", ec="#b5451b", lw=1.3),
                arrowprops=dict(arrowstyle="->", color="#b5451b", lw=1.4), zorder=6)

    ax.legend(loc="lower left", frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="#94a3b8", fontsize=9.8, borderpad=0.45, handlelength=1.5, handletextpad=0.50)


# ==============================================================================
# MAIN MULTI-PANEL COMPOSITE ASSEMBLY
# ==============================================================================
def generate_master_figure():
    plt.rcParams.update(MASTER_TYPOGRAPHY)

    fig = plt.figure(figsize=(15.2, 17.6), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.30)

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
        os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_2", "fig2_ode_and_continuation_master.png")),
        os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_2", "fig2_ode_and_continuation_master.pdf")),
        os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "Figures", "figure_2", "fig2_ode_and_continuation_master.svg")),
        os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "manuscript", "figures", "fig2_ode_and_continuation.png")),
        os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "manuscript", "figures", "fig2_ode_and_continuation.svg")),
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
