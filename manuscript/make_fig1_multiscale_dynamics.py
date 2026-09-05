"""Figure 1: Multiscale Thermal-Hydraulic Dynamics and Genesis of Flow Boiling Instabilities
Publication-grade, 6-panel figure connecting physical zonation, thermodynamic phase-change
non-linearity, force balance decomposition, static Ledinegg excursions, dynamic acoustic
transit delays, and non-linear limit-cycle phase orbits.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.gridspec as gridspec

os.makedirs("d:/AGravity/Tide_Tutor/manuscript/figures", exist_ok=True)

# --- Publication Styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.8,
    "axes.linewidth": 1.1,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
})

NAVY = "#1a3c6e"
CRIMSON = "#b5451b"
TEAL = "#007a78"
GOLD = "#c88a10"
SLATE = "#4a5568"

fig = plt.figure(figsize=(16.8, 10.2), dpi=300)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.30)

# ==============================================================================
# PANEL (a): Physical Channel Zonation & Moving Boiling Boundary
# ==============================================================================
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(-0.6, 4.6)
ax_a.set_ylim(-0.2, 5.8)
ax_a.axis("off")
ax_a.set_title(r"$\mathbf{(a)}$ Physical Zonation & Moving Boundary $\lambda(t)$", pad=12, loc="left", fontweight="bold")

# Plenums
ax_a.add_patch(FancyBboxPatch((0.5, -0.1), 3.0, 0.5, boxstyle="round,pad=0.05", fc="#d0dbe5", ec=NAVY, lw=1.5))
ax_a.text(2.0, 0.15, r"Lower Plenum ($\Delta P_{\mathrm{ext}}$ Header)", ha="center", va="center", fontsize=9.2, fontweight="bold", color=NAVY)

ax_a.add_patch(FancyBboxPatch((0.5, 5.0), 3.0, 0.5, boxstyle="round,pad=0.05", fc="#d0dbe5", ec=NAVY, lw=1.5))
ax_a.text(2.0, 5.25, r"Upper Plenum (Constant $\Delta P_{\mathrm{ext}}$)", ha="center", va="center", fontsize=9.2, fontweight="bold", color=NAVY)

# Channel walls
ax_a.plot([1.0, 1.0], [0.4, 5.0], color=SLATE, lw=3.0)
ax_a.plot([3.0, 3.0], [0.4, 5.0], color=SLATE, lw=3.0)

# Single phase region (0 < z < lambda)
z_lambda = 2.3
ax_a.add_patch(Rectangle((1.0, 0.4), 2.0, z_lambda - 0.4, fc="#dbeafe", ec="none", alpha=0.85))
ax_a.text(2.0, 1.35, "Single-Phase Liquid\n" + r"($0 \leq z < \lambda(t)$)" + "\n" + r"$T < T_{\mathrm{sat}}, \; \rho(z) = \rho_f$", 
          ha="center", va="center", fontsize=8.8, color="#1e40af")

# Moving boundary line
ax_a.plot([0.8, 3.2], [z_lambda, z_lambda], color=CRIMSON, lw=2.2, ls="--")
ax_a.text(3.3, z_lambda, r"$\mathbf{z = \lambda(t)}$" + "\n" + r"($h = h_f$ Saturation)", ha="left", va="center", fontsize=8.8, color=CRIMSON, fontweight="bold")

# Two-phase region (lambda < z < 1)
ax_a.add_patch(Rectangle((1.0, z_lambda), 2.0, 5.0 - z_lambda, fc="#ffedd5", ec="none", alpha=0.85))
ax_a.text(2.0, 3.55, "Two-Phase Mixture\n" + r"($\lambda(t) \leq z \leq 1$)" + "\n" + r"$\rho_m(z) \ll \rho_f, \; u(z) \uparrow$", 
          ha="center", va="center", fontsize=8.8, color="#9a3412")

# Bubbles in two-phase region
np.random.seed(42)
for _ in range(25):
    bx = np.random.uniform(1.2, 2.8)
    by = np.random.uniform(z_lambda + 0.15, 4.4)
    br = np.random.uniform(0.04, 0.11) * ((by - z_lambda)/2.7 + 0.6)
    ax_a.add_patch(Circle((bx, by), br, fc="#fed7aa", ec="#ea580c", lw=0.8, alpha=0.9))

# Heat flux arrows entering walls
for y_pos in np.linspace(0.8, 4.6, 7):
    ax_a.annotate("", xy=(0.98, y_pos), xytext=(0.4, y_pos),
                  arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4))
    ax_a.annotate("", xy=(3.02, y_pos), xytext=(3.6, y_pos),
                  arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4))
ax_a.text(0.12, 2.7, r"Uniform Heat Flux $q''$", ha="center", va="center", rotation=90, fontsize=9.2, color=CRIMSON, fontweight="bold")

# Inlet and exit velocities
ax_a.annotate("", xy=(2.0, 0.75), xytext=(2.0, 0.35),
              arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.0))
ax_a.text(1.9, 0.55, r"$u_i(t)$", ha="right", va="center", fontsize=9.5, color=NAVY, fontweight="bold")

ax_a.annotate("", xy=(2.0, 4.9), xytext=(2.0, 4.3),
              arrowprops=dict(arrowstyle="->", color="#9a3412", lw=2.5))
ax_a.text(1.9, 4.6, r"$u_e(t) \gg u_i$", ha="right", va="center", fontsize=9.5, color="#9a3412", fontweight="bold")


# ==============================================================================
# PANEL (b): Phase-Change Density Disparity & Quality Paradox
# ==============================================================================
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_title(r"$\mathbf{(b)}$ Thermodynamic Void Non-Linearity", pad=12, loc="left", fontweight="bold")

x = np.linspace(0.0, 1.0, 500)
rho_f_70 = 740.0
rho_g_70 = 36.0
gamma_70 = rho_g_70 / rho_f_70
alpha_70 = x / (x + (1.0 - x) * gamma_70)

gamma_1 = 1.0 / 1600.0
alpha_1 = x / (x + (1.0 - x) * gamma_1)

ax_b.plot(x * 100, alpha_70 * 100, color=CRIMSON, lw=2.4, label=r"BWR Core ($70\,\mathrm{bar}, \; \rho_f/\rho_g \approx 20.5$)")
ax_b.plot(x * 100, alpha_1 * 100, color=SLATE, lw=1.8, ls="--", label=r"Atmospheric ($1\,\mathrm{bar}, \; \rho_f/\rho_g \approx 1600$)")

x_target = 0.05
alpha_target = x_target / (x_target + (1.0 - x_target) * gamma_70) * 100

ax_b.plot([5.0, 5.0], [0, alpha_target], color=GOLD, ls=":", lw=1.6)
ax_b.plot([0, 5.0], [alpha_target, alpha_target], color=GOLD, ls=":", lw=1.6)
ax_b.plot(5.0, alpha_target, "o", color=CRIMSON, ms=8, zorder=5, mec="black", mew=1.2)

ax_b.annotate(r"$\mathbf{5\%}$ Mass Quality ($x=0.05$)" + "\n" + rf"$\Rightarrow \mathbf{{\alpha = {alpha_target:.1f}\%}}$ Channel Volume!",
             xy=(5.0, alpha_target), xytext=(22, 36),
             fontsize=8.8, color="#9a3412", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff7ed", ec="#ea580c", lw=1.1),
             arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.3))

ax_b.set_xlabel(r"Thermodynamic Mass Quality, $x \; (\%)$")
ax_b.set_ylabel(r"Cross-Sectional Void Fraction, $\alpha \; (\%)$")
ax_b.set_xlim(-2, 102)
ax_b.set_ylim(-2, 102)
ax_b.grid(True)
ax_b.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0")


# ==============================================================================
# PANEL (c): Internal S-Curve Force Decomposition
# ==============================================================================
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_title(r"$\mathbf{(c)}$ Hydrodynamic Force Decomposition", pad=12, loc="left", fontweight="bold")

G = np.linspace(0.1, 3.2, 500)
dp_1phi_fric = 0.22 * G**2 + 0.05 * G
dp_2phi_fric_accel = 2.4 / (1.0 + 3.0 * G**1.8)
dp_grav = 0.55 * (1.0 - 0.75 / (1.0 + 2.0 * G**1.5))
dp_total = dp_1phi_fric + dp_2phi_fric_accel + dp_grav

ax_c.plot(G, dp_1phi_fric, color=TEAL, lw=1.8, ls="-.", label=r"Single-Phase Friction ($\propto G^2$)")
ax_c.plot(G, dp_2phi_fric_accel, color=GOLD, lw=1.8, ls="--", label=r"Two-Phase Vapor Expansion ($\propto 1/\rho_m$)")
ax_c.plot(G, dp_grav, color=SLATE, lw=1.6, ls=":", label=r"Gravitational Head ($\propto \rho_m g$)")
ax_c.plot(G, dp_total, color=NAVY, lw=2.6, label=r"$\mathbf{Total \; Internal \; \Delta P}$ (S-Curve)")

idx_neg = np.where(np.gradient(dp_total, G) < 0)[0]
ax_c.axvspan(G[idx_neg[0]], G[idx_neg[-1]], color="#fee2e2", alpha=0.55, label=r"Negative Slope ($\frac{\partial \Delta P}{\partial G} < 0$)")

ax_c.set_xlabel(r"Mass Flow Rate, $G \; (\mathrm{kg/m^2 s})$")
ax_c.set_ylabel(r"Channel Pressure Drop, $\Delta P$")
ax_c.set_xlim(0.1, 3.2)
ax_c.set_ylim(0.0, 3.2)
ax_c.grid(True)
ax_c.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=8.4)


# ==============================================================================
# PANEL (d): Static Ledinegg Excursion & Dynamic Flow Jumps
# ==============================================================================
ax_d = fig.add_subplot(gs[1, 0])
ax_d.set_title(r"$\mathbf{(d)}$ Static Ledinegg Instability & Flow Excursion", pad=12, loc="left", fontweight="bold")

def s_curve(g):
    x = g - 1.6
    return x**3 - 1.15 * x + 1.45

G_d = np.linspace(0.2, 3.0, 500)
dP_d = s_curve(G_d)

pump_head = 1.35
pump_line = np.full_like(G_d, pump_head)

diff = dP_d - pump_head
idx_cross = np.where(np.diff(np.sign(diff)) != 0)[0]
g_roots = [G_d[i] for i in idx_cross]

ax_d.plot(G_d, dP_d, color=NAVY, lw=2.4, label=r"Channel Characteristic $\Delta P_{\mathrm{channel}}(G)$")
ax_d.plot(G_d, pump_line, color=CRIMSON, lw=2.0, ls="--", label=r"External Pump Supply $\Delta P_{\mathrm{ext}}$")

colors = [TEAL, CRIMSON, TEAL]
for gr, col in zip(g_roots, colors):
    ax_d.plot(gr, pump_head, "o", color=col, ms=8, zorder=5, mec="black", mew=1.2)

# Neatly separated point labels
ax_d.annotate(r"$\mathbf{A}$ (Vapor-Rich)", (g_roots[0], pump_head), xytext=(0.7, pump_head + 0.72),
             fontsize=8.6, ha="center", color=TEAL, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))

ax_d.annotate(r"$\mathbf{B}$ (Unstable Saddle)" + "\n" + r"$\mathbf{\partial \Delta P / \partial G < 0}$", 
             (g_roots[1], pump_head), xytext=(g_roots[1], pump_head - 0.68),
             fontsize=8.6, ha="center", color=CRIMSON, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.2))

ax_d.annotate(r"$\mathbf{C}$ (Liquid-Rich)", (g_roots[2], pump_head), xytext=(2.4, pump_head + 0.72),
             fontsize=8.6, ha="center", color=TEAL, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))

# Excursion jump arrow
ax_d.annotate("", xy=(g_roots[0] + 0.12, pump_head + 0.15), xytext=(g_roots[1] - 0.12, pump_head + 0.15),
             arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=2.0, ls=":", connectionstyle="arc3,rad=-0.28"))
ax_d.text(1.08, 1.85, "Runaway Flow Collapse\n" + r"$\mathbf{B \rightarrow A}$ (Burnout / CHF)", fontsize=8.4, color=CRIMSON, fontweight="bold", ha="center")

ax_d.set_xlabel(r"Mass Flow Rate, $G$")
ax_d.set_ylabel(r"Pressure Drop, $\Delta P$")
ax_d.set_xlim(0.2, 3.0)
ax_d.set_ylim(0.2, 2.5)
ax_d.grid(True)
ax_d.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=8.6)


# ==============================================================================
# PANEL (e): Density-Wave Transit Delay & Acoustic Feedback
# ==============================================================================
ax_e = fig.add_subplot(gs[1, 1])
ax_e.set_title(r"$\mathbf{(e)}$ Dynamic Density-Wave Transit Delay $\tau$", pad=12, loc="left", fontweight="bold")

t = np.linspace(0, 12, 500)
delta_ui = 0.25 * np.cos(2 * np.pi * t / 4.0)
tau = 2.0
delta_dp_exit = 0.35 * np.cos(2 * np.pi * (t - tau) / 4.0)

ax_e.plot(t, delta_ui, color=NAVY, lw=2.2, label=r"Inlet Flow Perturbation $\delta u_i(t)$")
ax_e.plot(t, delta_dp_exit, color=CRIMSON, lw=2.2, ls="--", label=r"Exit Pressure Feedback $\delta \Delta P_{\mathrm{exit}}(t)$")

t_peak1 = 4.0
t_peak2 = 6.0
ax_e.plot([t_peak1, t_peak1], [-0.52, 0.42], color=SLATE, ls=":", lw=1.3)
ax_e.plot([t_peak2, t_peak2], [-0.52, 0.42], color=SLATE, ls=":", lw=1.3)

ax_e.annotate("", xy=(t_peak2, 0.38), xytext=(t_peak1, 0.38),
             arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2.0))
ax_e.text((t_peak1 + t_peak2)/2, 0.45, r"$\mathbf{\tau = T/2 \; (180^\circ \; Lag)}$", 
          ha="center", va="bottom", fontsize=9.0, color="#9a3412", fontweight="bold")

ax_e.text(6.0, -0.42, r"$\mathbf{Acoustic \; Feedback:}$ Delayed steam packet exits channel" + "\n" + 
          r"$\Rightarrow \Delta P$ pressure spike $\Rightarrow$ Further chokes inlet velocity $u_i$", 
          ha="center", va="center", fontsize=8.4,
          bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec="#cbd5e1", lw=1.0))

ax_e.set_xlabel(r"Dimensionless Time, $t$")
ax_e.set_ylabel(r"Perturbation Amplitude")
ax_e.set_xlim(0, 12)
ax_e.set_ylim(-0.55, 0.65)
ax_e.grid(True)
ax_e.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=8.4)


# ==============================================================================
# PANEL (f): Nonlinear Phase-Space Limit Cycle Attractor
# ==============================================================================
ax_f = fig.add_subplot(gs[1, 2])
ax_f.set_title(r"$\mathbf{(f)}$ Supercritical Hopf Limit-Cycle Orbit", pad=12, loc="left", fontweight="bold")

theta = np.linspace(0, 2*np.pi, 200)
u_center, lam_center = 0.45, 0.52
u_orbit = u_center + 0.28 * np.cos(theta) - 0.08 * np.sin(2*theta)
lam_orbit = lam_center + 0.18 * np.sin(theta) + 0.05 * np.cos(2*theta)

t_spiral = np.linspace(0, 15, 300)
r_spiral = 0.02 * np.exp(0.22 * t_spiral)
r_spiral = np.minimum(r_spiral, 1.0)
u_spiral = u_center + r_spiral * (0.28 * np.cos(t_spiral) - 0.08 * np.sin(2*t_spiral))
lam_spiral = lam_center + r_spiral * (0.18 * np.sin(t_spiral) + 0.05 * np.cos(2*t_spiral))

ax_f.plot(u_spiral[:200], lam_spiral[:200], color=GOLD, lw=1.4, ls=":", label=r"Unstable Spiral ($\mathrm{Re}(\mu) > 0$)")
ax_f.plot(u_orbit, lam_orbit, color=CRIMSON, lw=2.6, label=r"Stable Limit Cycle $\Gamma$")
ax_f.plot(u_center, lam_center, "x", color="black", ms=9, mew=2.2, label=r"Unstable Focus $\mathbf{x}^*$")

ax_f.annotate("", xy=(u_orbit[50], lam_orbit[50]), xytext=(u_orbit[45], lam_orbit[45]),
             arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=2.2))
ax_f.annotate("", xy=(u_orbit[150], lam_orbit[150]), xytext=(u_orbit[145], lam_orbit[145]),
             arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=2.2))

# Inset: Time series of sustained velocity oscillations
inset_ax = ax_f.inset_axes([0.52, 0.10, 0.44, 0.34])
t_sim = np.linspace(0, 20, 300)
u_sim = u_center + 0.28 * np.cos(1.83 * t_sim) * (1.0 - np.exp(-0.35 * t_sim))
inset_ax.plot(t_sim, u_sim, color=CRIMSON, lw=1.5)
inset_ax.set_title(r"Sustained $u_i(t)$ Oscillations", fontsize=7.8, pad=2)
inset_ax.set_xlabel(r"$t$", fontsize=7.2, labelpad=1)
inset_ax.set_ylabel(r"$u_i$", fontsize=7.2, labelpad=1)
inset_ax.tick_params(labelsize=6.8)
inset_ax.grid(True, alpha=0.3)

ax_f.set_xlabel(r"Inlet Liquid Velocity, $u_i(t)$")
ax_f.set_ylabel(r"Boiling Boundary Position, $\lambda(t)$")
ax_f.set_xlim(0.1, 0.82)
ax_f.set_ylim(0.25, 0.8)
ax_f.grid(True)
ax_f.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#e2e8f0", fontsize=8.4)

# Output saving
out_path = "d:/AGravity/Tide_Tutor/manuscript/figures/fig1_multiscale_dynamics.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
fig.savefig("d:/AGravity/Tide_Tutor/manuscript/figures/fig1_scurve_schematic.png", dpi=300, bbox_inches="tight")
print("Saved Figure 1 to:", out_path)
