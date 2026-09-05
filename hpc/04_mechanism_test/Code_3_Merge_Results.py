"""Section 5.3 mechanism test merge stage: collects 20 per-seed result.json
files, recomputes the median-rho / fraction-significant summaries (same
logic as scripts/run_mechanism_test.py's own main()), and writes the
merged JSON, a per-seed CSV, and a plain-text report.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
RESULTS = Path(os.environ.get("TIDE_ROOT", "/home/barathula.sreeram/Python_Stuff/Fresh_TIDE")) / \
    "results" / "04_mechanism_test"
OUT_DIR = REPO_ROOT / "data" / "generated"

from scipy import stats as sstats  # noqa: E402


def _summarize(rhos, ps):
    rhos, ps = np.array(rhos), np.array(ps)
    return dict(
        n_seeds=len(rhos), median_rho=float(np.median(rhos)) if len(rhos) else None,
        frac_positive_significant=float(np.mean((rhos > 0) & (ps < 0.05))) if len(rhos) else None,
        frac_negative_significant=float(np.mean((rhos < 0) & (ps < 0.05))) if len(rhos) else None,
    )


def main():
    task_dirs = sorted(RESULTS.glob("task_*"), key=lambda d: int(d.name.replace("task_", "")))
    per_seed = []
    missing = []
    for d in task_dirs:
        idx = int(d.name.replace("task_", ""))
        if not (d / "PASS").exists() or not (d / "result.json").exists():
            missing.append(idx)
            continue
        with open(d / "result.json") as f:
            per_seed.append(json.load(f))

    if missing:
        print(f"FAIL: seeds {missing} did not complete successfully", file=sys.stderr)
        sys.exit(1)

    grad_rhos, grad_ps, width_rhos, width_ps = [], [], [], []
    for r in per_seed:
        upper_error = np.array(r["upper_error"])
        grad_norm = np.array(r["grad_norm"])
        window_width = np.array(r["window_width"])
        finite = np.isfinite(upper_error) & np.isfinite(grad_norm) & (grad_norm > 0)
        if finite.sum() > 5:
            rho, p = sstats.spearmanr(upper_error[finite], 1.0 / grad_norm[finite])
            grad_rhos.append(rho)
            grad_ps.append(p)
        finite2 = np.isfinite(upper_error) & np.isfinite(window_width) & (window_width > 0)
        if finite2.sum() > 5:
            rho2, p2 = sstats.spearmanr(upper_error[finite2], 1.0 / window_width[finite2])
            width_rhos.append(rho2)
            width_ps.append(p2)

    output = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), n_seeds=len(per_seed),
        gradient_mechanism=_summarize(grad_rhos, grad_ps),
        window_width_mechanism=_summarize(width_rhos, width_ps),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"mechanism_test_full_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = OUT_DIR / f"mechanism_test_per_seed_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "nsub", "upper_error", "grad_norm", "window_width"])
        for r in per_seed:
            for ns, ue, gn, ww in zip(r["nsub"], r["upper_error"], r["grad_norm"], r["window_width"]):
                writer.writerow([r["seed"], ns, ue, gn, ww])

    report = "\n".join([
        "TIDE PACKAGE 04 - MECHANISM TEST, MERGED RESULT", "",
        f"Gradient mechanism:     median rho = {output['gradient_mechanism']['median_rho']:.3f} "
        f"(manuscript: -0.675)",
        f"Window-width mechanism: median rho = {output['window_width_mechanism']['median_rho']:.3f} "
        f"(manuscript: +0.668)",
        "", f"Merged JSON: {out_json}", f"CSV: {csv_path}", "", "STATUS: PASS",
    ])
    (RESULTS / "FINAL_REPORT.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
