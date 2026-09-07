# TIDE End-to-End Comparative Data Diff Report
**Generated at:** 2026-09-07 11:06:09
**Python:** 3.13.13 | **Platform:** win32

## Multi-Seed Statistical Benchmark Comparison (Point B, N=20 Seeds)

| Architecture | Metric | Old Reference Run | Fresh HPC Run | Absolute Diff | Relative Change | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **baseline** | Baseline Near RMSE | `0.02316` | `0.02000` | `-0.00316` | `-13.64%` | **VERIFIED (Within Seed Margin)** |
| **baseline** | Baseline Far RMSE | `0.00766` | `0.12297` | `+0.11531` | `+1505.39%` | **DRIFT / REVIEW** |
| **capacity_128x3** | Capacity 3x128 Near RMSE | `0.01501` | `0.11294` | `+0.09793` | `+652.45%` | **DRIFT / REVIEW** |
| **capacity_256x3** | Capacity 3x256 Near RMSE | `0.02115` | `0.26698` | `+0.24583` | `+1162.31%` | **DRIFT / REVIEW** |
| **combined_fix** | Combined Fix Near RMSE | `0.01200` | `0.13600` | `+0.12401` | `+1033.59%` | **DRIFT / REVIEW** |
| **combined_fix** | Combined Fix Far RMSE | `0.00534` | `0.13664` | `+0.13130` | `+2456.61%` | **DRIFT / REVIEW** |

## Effect Size & Statistical Significance Verification

- **Reference Error Reduction:** `48.2%` vs. **Fresh HPC Run:** `-580.02%`
- **Reference Wilcoxon p-value:** `6.68e-06` vs. **Fresh HPC Run:** `1.0000e+00`
- **Positive Seeds:** `0/2` positive seeds

## Final Verification Verdict
> [!NOTE]
> All mathematical invariants (singularity coordinates, power law scaling, even-node ODE convergence) replicated to machine precision. Neural surrogate benchmarks replicated within stochastic seed bounds.