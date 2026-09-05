"""Figure 4: absolute near-BT vs. far-BT boundary error across baseline and
every Phase 4 fix attempt -- the figure that makes the ratio-metric lesson
visible (a fix can shrink the far-error denominator without helping, or
even while hurting, the near-BT numerator that actually matters).
Values are the seed-averaged results already computed and recorded in
PROJECT_LOG.md Sec. 41-43 (baseline/capacity: 20 seeds; Fourier/log-distance/
boundary-weighted single-architecture: 6 seeds; combined fix: 20 seeds,
Sec. 43 final).
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11.5, "font.family": "serif", "axes.linewidth": 1.1})

rows = [
    ("Baseline\n(64,64)", 0.02411, 0.00681, 3.54),
    ("Capacity\n(128,128,128)", 0.01817, 0.01509, 1.20),
    ("Capacity\n(128,128,128,128)", 0.01686, 0.00964, 1.75),
    ("Capacity\n(256,256,256)", 0.02426, 0.01463, 1.66),
    ("Fourier\nfeatures", 0.02421, 0.00559, 4.33),
    ("Log-distance\nfeature", 0.03583, 0.00635, 5.64),
    ("Boundary-\nweighted loss", 0.02088, 0.00276, 7.56),
    ("Combined:\ncapacity+BW loss", 0.01034, 0.00579, 1.785),
]

names = [r[0] for r in rows]
near = [r[1] for r in rows]
far = [r[2] for r in rows]

x = np.arange(len(rows))
w = 0.36

fig, ax = plt.subplots(figsize=(10.5, 5.4))
finite_mask = [n is not None for n in near]
xn = x[np.array(finite_mask)]
near_v = [n for n in near if n is not None]
far_v = [f for f in far if f is not None]

bars_near = ax.bar(xn - w / 2, near_v, width=w, color="#b5451b", label="Near-BT error (|dist| $\\leq$ 0.5)")
bars_far = ax.bar(xn + w / 2, far_v, width=w, color="#1a3c6e", label="Far-from-BT error (|dist| $\\geq$ 2.0)")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Mean absolute upper-boundary error\n(in $N_{pch}$ units)")
ax.legend(frameon=False, fontsize=10.5, loc="upper right")
ax.set_title("Absolute error is the metric that matters for \"did the fix help\";\n"
              "the near/far ratio can mislead when a fix changes both regions unevenly", fontsize=11)
fig.tight_layout()
out = "/Users/sreerambarathula/Codes/Claude_Code/Flow_Boiling_Instability/manuscript/figures/fig4_fix_comparison.png"
fig.savefig(out, dpi=300)
print("saved", out, "-- NOTE: combined-fix bar pending final 20-seed numbers")
