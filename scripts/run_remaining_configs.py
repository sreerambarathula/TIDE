"""One-off: run just the 4 configs that hadn't completed before the
original full run was killed (PROJECT_LOG.md Sec. 46), using the fixed
subprocess-per-config worker interface, and merge with the 3 already-valid
Point B results into one complete, final output file.
"""
import json
import time
from pathlib import Path

from run_phase4_experiments import (CONFIGS_B, CONFIGS_C, OUT_DIR,
                                     _provenance, _run_config_subprocess)

# Already valid, from the killed run's log (PROJECT_LOG.md Sec. 46) --
# not re-run, since these completed successfully before the kill.
ALREADY_DONE_B = {
    "baseline": dict(config_name="baseline", n_seeds=20, n_valid=20,
                      mean_near=0.02316, mean_far=0.00766, ratio=3.023, p_value=0.0001307,
                      note="from killed run's log, not re-run"),
    "capacity_128x3": dict(config_name="capacity_128x3", n_seeds=20, n_valid=20,
                            mean_near=0.01501, mean_far=0.00691, ratio=2.171, p_value=1.335e-05,
                            note="from killed run's log, not re-run"),
    "capacity_128x4": dict(config_name="capacity_128x4", n_seeds=20, n_valid=20,
                            mean_near=0.01851, mean_far=0.01508, ratio=1.227, p_value=0.01812,
                            note="from killed run's log, not re-run"),
}

output = dict(provenance=_provenance(), mode="full", note="Point B configs 1-3 reused from killed run (Sec. 46); configs 4-5 and all of Point C freshly run here with the fixed subprocess-per-config worker.")

print("=== Point B (remaining) ===", flush=True)
results_B = dict(ALREADY_DONE_B)
for idx, (name, fn_name, kwargs) in enumerate(CONFIGS_B):
    if name in ALREADY_DONE_B:
        continue
    results_B[name] = _run_config_subprocess("B", idx, 20, name)
output["point_B"] = results_B

print("=== Point C ===", flush=True)
results_C = {}
for idx, (name, fn_name, kwargs) in enumerate(CONFIGS_C):
    results_C[name] = _run_config_subprocess("C", idx, 6, name)
output["point_C"] = results_C

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"phase4_results_full_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
