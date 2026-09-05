"""Point A (Section 5.1) merge stage: collects the 5 config task outputs
into one point_a_results_full_<timestamp>.json (same schema
run_point_a_experiments.py's own main() would produce), a CSV of every
per-seed value, and a plain-text report comparing against PROJECT_LOG.md's
documented numbers (Sec. 31/32/34) for each config.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
RESULTS = Path(os.environ.get("TIDE_ROOT", "/home/barathula.sreeram/Python_Stuff/Fresh_TIDE")) / \
    "results" / "03_point_a_analysis"
OUT_DIR = REPO_ROOT / "data" / "generated"

CONFIG_NAMES = ["boundary_band_64x64", "uniform_64x64", "misfit_adaptive_64x64",
                "boundary_band_32x32", "boundary_band_128x128"]

# PROJECT_LOG.md's documented numbers, for comparison only.
PROJECT_LOG_HOPF = {
    "boundary_band_64x64": dict(ratio_mean=2.32, ratio_median=2.16, p=2.4e-4),
    "uniform_64x64": dict(ratio_mean=0.72, ratio_median=0.71, p=0.998),
    "misfit_adaptive_64x64": dict(ratio_mean=0.69, ratio_median=0.69, p=0.98),
    "boundary_band_32x32": dict(ratio_mean=0.92, ratio_median=0.68, p=0.88),
    "boundary_band_128x128": dict(ratio_mean=0.84, ratio_median=1.58, p=0.31),
}


def main():
    task_dirs = sorted(RESULTS.glob("task_*"))
    results = {}
    missing = []
    for d in task_dirs:
        idx = int(d.name.replace("task_", ""))
        if not (d / "PASS").exists() or not (d / "result.json").exists():
            missing.append(idx)
            continue
        with open(d / "result.json") as f:
            results[CONFIG_NAMES[idx]] = json.load(f)

    if missing:
        print(f"FAIL: tasks {missing} did not complete successfully", file=sys.stderr)
        sys.exit(1)

    output = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), mode="full", configs=results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"point_a_results_full_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    # CSV: one row per (config, seed, metric)
    csv_path = OUT_DIR / f"point_a_per_seed_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "dataset", "hidden_sizes", "metric", "seed_index", "near", "far"])
        for cfg_name, r in results.items():
            for metric, d in r["per_seed"].items():
                for i, (n, fa) in enumerate(zip(d["near"], d["far"])):
                    writer.writerow([cfg_name, r["dataset"], r["hidden_sizes"], metric, i, n, fa])

    report_lines = ["TIDE PACKAGE 03 - POINT A ANALYSIS, MERGED RESULT", ""]
    report_lines.append(f"{'config':<24}{'hopf_ratio_mean':<18}{'hopf_ratio_median':<20}{'p':<12}{'PROJECT_LOG (mean/median/p)'}")
    for cfg_name, r in results.items():
        hopf = r["metrics"]["hopf_error"]
        pl = PROJECT_LOG_HOPF.get(cfg_name)
        pl_str = f"{pl['ratio_mean']}/{pl['ratio_median']}/{pl['p']:.2g}" if pl else "n/a"
        report_lines.append(
            f"{cfg_name:<24}{hopf['ratio_mean']:<18.3f}{hopf['ratio_median']:<20.3f}"
            f"{hopf['p_value']:<12.3g}{pl_str}")

    report_lines += ["", f"Merged JSON: {out_json}", f"CSV: {csv_path}", "", "STATUS: PASS"]
    report_text = "\n".join(report_lines)
    (RESULTS / "FINAL_REPORT.txt").write_text(report_text)
    print(report_text)


if __name__ == "__main__":
    main()
