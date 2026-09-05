"""Figure 2: real fold/Hopf-type boundary curves near (a) the transversal
codim-2 point (Point A) and (b) the genuine Bogdanov-Takens point (Point
B), both from this project's own ground-truth continuation tools -- not a
schematic.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from tide.continuation.codim2_convergence import _fold, hopf_npch_general
from tide.surrogates.bt_point_data import (FR_BT, KE_BT, KI_BT, LAM_BT, NPCH_BT,
                                            NSUB_BT, wedge_boundaries_true)

plt.rcParams.update({"font.size": 12, "font.family": "serif", "axes.linewidth": 1.1})

N1 = 16

# --- Point A: transversal crossing (real facility parameters) ---
FR_A, LAM_A, KI_A, KE_A = 0.035, 5.90, 6.55, 2.03
NSUB_A = 29.886

nsub_a = np.linspace(NSUB_A - 4, NSUB_A + 4, 60)
fold_a, hopf_a = [], []
for ns in nsub_a:
    f = _fold(ns, FR_A, LAM_A, KI_A, KE_A)
    fold_a.append(f)
    if f is None:
        hopf_a.append(None)
        continue
    h = hopf_npch_general(ns, FR_A, LAM_A, KI_A, KE_A, N1, bracket=(f * 0.85, f * 1.4))
    hopf_a.append(h)

fold_a = np.array([v if v is not None else np.nan for v in fold_a])
hopf_a = np.array([v if v is not None else np.nan for v in hopf_a])

# --- Point B: genuine Bogdanov-Takens point ---
nsub_b = np.linspace(NSUB_BT - 3.0, NSUB_BT, 80)
lower_b, upper_b = [], []
for ns in nsub_b:
    lo, hi = wedge_boundaries_true(ns, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    lower_b.append(lo)
    upper_b.append(hi)
lower_b = np.array([v if v is not None else np.nan for v in lower_b])
upper_b = np.array([v if v is not None else np.nan for v in upper_b])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

ax = axes[0]
ax.plot(nsub_a, fold_a, color="#1a3c6e", lw=2.2, label="Fold (Ledinegg) curve")
ax.plot(nsub_a, hopf_a, color="#b5451b", lw=2.2, label="Hopf (DWO) curve")
ax.axvline(NSUB_A, color="#888888", ls=":", lw=1.2)
ax.set_title("(a) Point A — transversal crossing\n(curves cross at an angle)", fontsize=11.5)
ax.set_xlabel("$N_{sub}$")
ax.set_ylabel("$N_{pch}$")
ax.legend(frameon=False, fontsize=9.5, loc="upper left")

ax = axes[1]
ax.plot(nsub_b, lower_b, color="#1a3c6e", lw=2.2, label="Lower boundary (fold-type)")
ax.plot(nsub_b, upper_b, color="#b5451b", lw=2.2, label="Upper boundary (Hopf-type)")
ax.fill_between(nsub_b, lower_b, upper_b, color="#ffe8b3", alpha=0.6, label="Stable window")
ax.axvline(NSUB_BT, color="#888888", ls=":", lw=1.2)
ax.plot([NSUB_BT], [NPCH_BT], marker="*", color="black", ms=16, zorder=5)
ax.annotate("BT point\n(window closes\nto zero width)", (NSUB_BT, NPCH_BT),
            xytext=(NSUB_BT - 2.85, 22.3), fontsize=9.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
ax.set_title("(b) Point B — genuine Bogdanov-Takens point\n(curves meet tangentially)", fontsize=11.5)
ax.set_xlabel("$N_{sub}$")
ax.set_ylabel("$N_{pch}$")
ax.legend(frameon=False, fontsize=9.5, loc="upper left")

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig2_bifurcation_maps.png")
fig.savefig(out, dpi=300)
print("saved", out)
print("Point A fold/hopf sample:", fold_a[:3], hopf_a[:3])
print("Point B lower/upper sample near BT:", lower_b[-5:], upper_b[-5:])
