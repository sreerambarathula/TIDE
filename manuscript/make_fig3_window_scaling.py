"""Figure 3: log-log plot of stable-window width vs. distance from the BT
point (Point B), the direct empirical measurement that ruled out the
naive quadratic-tangency assumption and motivated (then refuted, for
this problem) the log-distance-feature fix attempt.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from tide.surrogates.bt_point_data import FR_BT, KE_BT, KI_BT, LAM_BT, wedge_boundaries_true

plt.rcParams.update({"font.size": 12, "font.family": "serif", "axes.linewidth": 1.1})

# CORRECTED (post-audit): the true double-zero point, from double_zero.py's
# 2D Newton solve, not the original fold-curve-constrained NSUB_BT; and
# wedge_boundaries_true (true g=0 roots), not the old fold-as-lower-boundary
# wedge_boundaries. See PROJECT_LOG.md for the full correction record.
NSUB_BT_CORRECTED = 14.142794816

N1 = 16
deltas = np.geomspace(1e-4, 2.0, 40)
widths, ds = [], []
for d in deltas:
    nsub = NSUB_BT_CORRECTED - d
    lo, hi = wedge_boundaries_true(nsub, FR_BT, LAM_BT, KI_BT, KE_BT, N1)
    if lo is None or hi is None:
        continue
    widths.append(hi - lo)
    ds.append(d)
ds, widths = np.array(ds), np.array(widths)

logd, logw = np.log(ds), np.log(widths)
slope, intercept = np.polyfit(logd, logw, 1)
resid = logw - (slope * logd + intercept)
r2 = 1 - np.sum(resid**2) / np.sum((logw - logw.mean()) ** 2)

fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.loglog(ds, widths, "o", color="#1a3c6e", ms=5, label="Measured window width")
fit_d = np.geomspace(ds.min(), ds.max(), 50)
fit_w = np.exp(intercept) * fit_d**slope
ax.loglog(fit_d, fit_w, color="#b5451b", lw=2,
          label=f"Power-law fit: width $\\propto \\delta^{{{slope:.2f}}}$\n($R^2$={r2:.3f})")
ref_w = np.exp(intercept) * fit_d**2  # what quadratic tangency would predict, same intercept
ax.loglog(fit_d, ref_w, color="#888888", lw=1.5, ls="--",
          label="Naive quadratic prediction\n(textbook BT tangency), for reference")

ax.set_xlabel("Distance from BT point,  $\\delta = N_{sub,BT} - N_{sub}$")
ax.set_ylabel("Stable window width  (in $N_{pch}$)")
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig3_window_scaling.png")
fig.savefig(out, dpi=300)
print("saved", out, "slope=", slope, "R2=", r2)
