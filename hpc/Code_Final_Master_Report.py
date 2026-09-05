"""THE consolidated report: runs only after all 4 analysis streams
(Table 3/4, point verification, Point A analysis, mechanism test) have
succeeded. Reads each stream's own merged output, and writes:

  - $TIDE_ROOT/results/MASTER_REPORT.txt: every quantitative result this
    entire pipeline produced, next to the manuscript's currently-printed
    value, with a per-stage and one overall PASS/FAIL.
  - $TIDE_ROOT/results/MASTER_SUMMARY.csv: the same comparisons as rows,
    for spreadsheet/plotting use.

Does not recompute anything -- pure aggregation of what the 4 streams
already produced and saved.
"""
import csv
import glob
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("TIDE_ROOT", "/home/barathula.sreeram/Python_Stuff/Fresh_TIDE"))
REPO = ROOT / "repo"
DATA_GEN = REPO / "data" / "generated"
RESULTS = ROOT / "results"


def _latest(pattern):
    matches = sorted(glob.glob(str(DATA_GEN / pattern)))
    return Path(matches[-1]) if matches else None


def _load(path):
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


rows = []  # (stage, item, ours, manuscript, note)


def add(stage, item, ours, manuscript, note=""):
    rows.append(dict(stage=stage, item=item, ours=ours, manuscript=manuscript, note=note))


def main():
    stage_status = {}

    # --- Stage: Point verification (Table 2) ---
    pv_path = RESULTS / "02_point_verification" / "result.json"
    pv = _load(pv_path)
    if pv:
        stage_status["02_point_verification"] = all(
            pv[k]["status"] == "PASS" for k in ("point_A", "point_B", "point_C"))
        add("Point verification", "Point A Nsub*", pv["point_A"]["found_nsub"], 29.886)
        add("Point verification", "Point A slope diff", pv["point_A"]["slope_diff"], 0.92)
        add("Point verification", "Point B Nsub*", pv["point_B"]["nsub"], 14.142794816)
        add("Point verification", "Point B Npch*", pv["point_B"]["npch"], 20.597778032)
        add("Point verification", "Point C Nsub*", pv["point_C"]["nsub"], 7.630101792)
        add("Point verification", "Point C Npch*", pv["point_C"]["npch"], 10.913202566)
        sw = pv.get("transversality_sweep", {})
        add("Point verification", "Transversality sweep: near-tangent count", sw.get("near_tangent_count"), 0,
            note="qualitative match; quantitative slope-diff range may differ from the historical sweep's exact grid -- see REPRODUCIBILITY.md")
    else:
        stage_status["02_point_verification"] = False

    # --- Stage: Point A analysis (Section 5.1) ---
    pa = _load(_latest("point_a_results_full_*.json"))
    if pa:
        stage_status["03_point_a_analysis"] = True
        pl = {
            "boundary_band_64x64": (2.32, 2.16, 2.4e-4),
            "uniform_64x64": (0.72, 0.71, 0.998),
            "misfit_adaptive_64x64": (0.69, 0.69, 0.98),
            "boundary_band_32x32": (0.92, 0.68, 0.88),
            "boundary_band_128x128": (0.84, 1.58, 0.31),
        }
        for cfg_name, r in pa["configs"].items():
            hopf = r["metrics"]["hopf_error"]
            m = pl.get(cfg_name)
            add("Point A", f"{cfg_name} hopf ratio (mean)", round(hopf["ratio_mean"], 3),
                m[0] if m else None)
            add("Point A", f"{cfg_name} hopf ratio (median)", round(hopf["ratio_median"], 3),
                m[1] if m else None)
            add("Point A", f"{cfg_name} p-value", hopf["p_value"], m[2] if m else None)
    else:
        stage_status["03_point_a_analysis"] = False

    # --- Stage: Mechanism test (Section 5.3) ---
    mech = _load(_latest("mechanism_test_full_*.json"))
    if mech:
        stage_status["04_mechanism_test"] = True
        add("Mechanism test", "Gradient mechanism median rho",
            round(mech["gradient_mechanism"]["median_rho"], 3), -0.675)
        add("Mechanism test", "Window-width mechanism median rho",
            round(mech["window_width_mechanism"]["median_rho"], 3), 0.668)
    else:
        stage_status["04_mechanism_test"] = False

    # --- Stage: Table 3/4 ---
    t34 = _load(_latest("phase4_results_full_*.json"))
    if t34:
        stage_status["01_tide_phase4_rerun"] = True
        manuscript_t3 = {
            "baseline": (0.0241, 0.0068, 3.54, 9.5e-7),
            "capacity_128x3": (0.0182, 0.0151, 1.20, 0.095),
            "capacity_128x4": (0.0169, 0.0096, 1.75, 0.016),
            "capacity_256x3": (0.0243, 0.0146, 1.66, 0.027),
            "combined_fix": (0.0103, 0.0058, 1.79, 6.7e-5),
        }
        manuscript_t4 = {
            "baseline": (0.0154, 0.0040, 3.83, 0.031),
            "combined_fix": (0.0070, 0.0058, 1.20, 0.031),
        }
        for cfg_name, r in t34.get("point_B", {}).items():
            m = manuscript_t3.get(cfg_name)
            add("Table 3 (Point B)", f"{cfg_name} near/far/ratio/p",
                f"{r['mean_near']:.4f}/{r['mean_far']:.4f}/{r['ratio']:.3f}/{r['p_value']:.3g}",
                f"{m[0]}/{m[1]}/{m[2]}/{m[3]:.2g}" if m else None)
        for cfg_name, r in t34.get("point_C", {}).items():
            m = manuscript_t4.get(cfg_name)
            add("Table 4 (Point C)", f"{cfg_name} near/far/ratio/p",
                f"{r['mean_near']:.4f}/{r['mean_far']:.4f}/{r['ratio']:.3f}/{r['p_value']:.3g}",
                f"{m[0]}/{m[1]}/{m[2]}/{m[3]:.2g}" if m else None)
    else:
        stage_status["01_tide_phase4_rerun"] = False

    overall_pass = all(stage_status.values())

    # --- Write MASTER_REPORT.txt ---
    lines = ["TIDE -- MASTER REPRODUCIBILITY REPORT", f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}", ""]
    lines.append("Stage status:")
    for stage, ok in stage_status.items():
        lines.append(f"  {stage:<28} {'PASS' if ok else 'FAIL/MISSING'}")
    lines.append("")
    lines.append(f"{'stage':<22}{'item':<45}{'ours':<32}{'manuscript'}")
    for r in rows:
        lines.append(f"{r['stage']:<22}{r['item']:<45}{str(r['ours']):<32}{str(r['manuscript'])}")
        if r["note"]:
            lines.append(f"{'':<22}  note: {r['note']}")
    lines.append("")
    lines.append(f"OVERALL STATUS: {'PASS' if overall_pass else 'FAIL'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "MASTER_REPORT.txt").write_text("\n".join(lines))

    with open(RESULTS / "MASTER_SUMMARY.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "item", "ours", "manuscript", "note"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n".join(lines))
    print(f"\nWritten: {RESULTS / 'MASTER_REPORT.txt'}")
    print(f"Written: {RESULTS / 'MASTER_SUMMARY.csv'}")
    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
