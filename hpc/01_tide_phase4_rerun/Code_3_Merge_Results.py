"""TIDE Phase-4 rerun -- merge stage.

Reads all 22 per-seed-batch result.json files written by Code_2_Run_Batch.sh
(one per PBS array task), regroups them by (point, config), and recomputes
the exact same summary statistic run_phase4_experiments.py's own
_run_single_worker would have produced for the full, un-split seed set --
same mean/ratio/Wilcoxon procedure, same formula, just applied to seeds
gathered from several small processes instead of one long-lived one.

Writes:
  - data/generated/phase4_results_full_<timestamp>.json, in the exact
    schema run_phase4_experiments.py's own main() produces (so it slots in
    next to the previous, partially-shortcut-based file of the same name
    for direct comparison).
  - FINAL_REPORT.txt: a plain-text PASS/FAIL summary plus a side-by-side
    comparison against the manuscript's Table 3/4 numbers, so it's readable
    without re-opening the JSON.

Run this on the login node after all 22 array tasks show PASS (or as the
tail of a PBS job dependent on the whole array via
afterokarray:<array_job_id>[] -- see submit_all.sh).
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sstats

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
RESULTS = Path(
    __import__("os").environ.get("TIDE_ROOT", "/home/barathula.sreeram/Python_Stuff/Fresh_TIDE")
) / "results" / "01_tide_phase4_rerun"
OUT_DIR = REPO_ROOT / "data" / "generated"

# Manuscript's reported numbers (Table 3, Table 4) for comparison only --
# does not gate PASS/FAIL. The whole point of this rerun is to find out,
# empirically and without any hand-copied shortcut, whether these still
# hold on a from-scratch execution.
MANUSCRIPT_TABLE3 = {
    ("B", "baseline"): dict(near=0.0241, far=0.0068, ratio=3.54, p=9.5e-7),
    ("B", "capacity_128x3"): dict(near=0.0182, far=0.0151, ratio=1.20, p=0.095),
    ("B", "capacity_128x4"): dict(near=0.0169, far=0.0096, ratio=1.75, p=0.016),
    ("B", "capacity_256x3"): dict(near=0.0243, far=0.0146, ratio=1.66, p=0.027),
    ("B", "combined_fix"): dict(near=0.0103, far=0.0058, ratio=1.79, p=6.7e-5),
    ("C", "baseline"): dict(near=0.0154, far=0.0040, ratio=3.83, p=0.031),
    ("C", "combined_fix"): dict(near=0.0070, far=0.0058, ratio=1.20, p=0.031),
}
EXPECTED_N_SEEDS = {"B": 20, "C": 6}


def _provenance():
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    return dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_commit=git_hash,
        lockfile="requirements-lock.txt",
        note=("Every config below was freshly executed on HPC via a PBS array "
              "job (22 seed-batch tasks), then merged here -- no value in this "
              "file was hand-copied from a prior run's log."),
    )


def main():
    task_dirs = sorted(RESULTS.glob("task_*"))
    if not task_dirs:
        print(f"ERROR: no task_* directories found under {RESULTS}", file=sys.stderr)
        sys.exit(1)

    batches = []
    missing_or_failed = []
    for d in task_dirs:
        idx = d.name.replace("task_", "")
        result_path = d / "result.json"
        pass_path = d / "PASS"
        if not pass_path.exists() or not result_path.exists():
            missing_or_failed.append(idx)
            continue
        with open(result_path) as f:
            batches.append(json.load(f))

    if missing_or_failed:
        print("FAIL: the following task indices did not complete successfully: "
              + ", ".join(sorted(missing_or_failed, key=int)), file=sys.stderr)
        print("Do not trust a merge run with missing tasks. Investigate those "
              "tasks' error.log / UPLOAD_THIS.txt before rerunning them and "
              "re-merging.", file=sys.stderr)
        sys.exit(1)

    grouped = {}
    for b in batches:
        key = (b["point"], b["config_name"])
        grouped.setdefault(key, []).append(b)

    output = dict(provenance=_provenance(), mode="full")
    output["point_B"] = {}
    output["point_C"] = {}
    report_lines = []
    overall_pass = True

    report_lines.append("TIDE PACKAGE 01 - PHASE4 RERUN, MERGED RESULT")
    report_lines.append(f"Generated: {output['provenance']['timestamp']}")
    report_lines.append(f"Git commit: {output['provenance']['git_commit']}")
    report_lines.append("")
    report_lines.append(f"{'point':<6}{'config':<18}{'n':<4}{'near':<10}{'far':<10}"
                         f"{'ratio':<8}{'p':<12}{'manuscript near/far/ratio/p'}")

    for (point, config_name), batch_list in sorted(grouped.items()):
        seed_near = {}
        seed_far = {}
        for b in batch_list:
            for s, n, f in zip(b["seeds"], b["per_seed_near"], b["per_seed_far"]):
                seed_near[s] = n
                seed_far[s] = f
        seeds_sorted = sorted(seed_near)
        near_list = [seed_near[s] for s in seeds_sorted]
        far_list = [seed_far[s] for s in seeds_sorted]

        expected_n = EXPECTED_N_SEEDS[point]
        if len(seeds_sorted) != expected_n or seeds_sorted != list(range(expected_n)):
            overall_pass = False
            report_lines.append(
                f"WARNING: {point}/{config_name} has seeds {seeds_sorted}, "
                f"expected exactly 0..{expected_n - 1}. Check for a missing or "
                f"duplicated array task before trusting this row.")

        near = np.array(near_list)
        far = np.array(far_list)
        valid = np.isfinite(near) & np.isfinite(far)
        near_v, far_v = near[valid], far[valid]
        stat, p = sstats.wilcoxon(near_v - far_v, alternative="greater")

        result = dict(
            config_name=config_name,
            n_seeds=len(seeds_sorted),
            n_valid=int(valid.sum()),
            per_seed_near=near_list,
            per_seed_far=far_list,
            mean_near=float(near_v.mean()),
            mean_far=float(far_v.mean()),
            ratio=float(near_v.mean() / far_v.mean()),
            p_value=float(p),
            elapsed_seconds=round(sum(b["elapsed_seconds"] for b in batch_list), 1),
            train_kwargs=batch_list[0]["train_kwargs"],
        )
        output[f"point_{point}"][config_name] = result

        m = MANUSCRIPT_TABLE3.get((point, config_name))
        m_str = (f"{m['near']}/{m['far']}/{m['ratio']}/{m['p']:.2g}" if m else "n/a")
        report_lines.append(
            f"{point:<6}{config_name:<18}{result['n_valid']:<4}"
            f"{result['mean_near']:<10.4f}{result['mean_far']:<10.4f}"
            f"{result['ratio']:<8.3f}{result['p_value']:<12.3g}{m_str}")

    out_path = OUT_DIR / f"phase4_results_full_{time.strftime('%Y%m%d_%H%M%S')}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    report_lines.append("")
    report_lines.append(f"Merged JSON written to: {out_path}")
    report_lines.append("")
    if overall_pass:
        report_lines.append("STATUS: PASS")
        report_lines.append("All 7 configurations (Point B x5, Point C x2) completed with "
                             "the full expected seed count, freshly executed end to end.")
    else:
        report_lines.append("STATUS: FAIL")
        report_lines.append("See WARNING lines above -- one or more configs did not get "
                             "the expected, complete seed set.")

    report_text = "\n".join(report_lines)
    report_path = RESULTS / "FINAL_REPORT.txt"
    report_path.write_text(report_text)
    print(report_text)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
