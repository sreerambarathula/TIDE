"""Reproducible experiment runner for the manuscript's Table 3 and Table 4
(PROJECT_LOG.md Sec. 44, external-audit finding 5: "the repository does
not specify an executable workflow that recreates the reported results").

Runs every configuration in Table 3 (Point B: baseline, three capacity
architectures, and the winning combined fix) and Table 4 (Point C:
baseline and the combined fix), using the CORRECTED tools (double_zero.py,
wedge_boundaries_true) confirmed in Sec. 44 not to change the reported
upper-boundary numbers. Saves per-seed raw results and a summary to
data/generated/ with provenance metadata, so every number in the
manuscript traces back to a committed artifact, not only to prose in this
log.

**Each configuration runs in its OWN subprocess**, not all eight inside
one long-lived Python process. Found necessary the hard way (PROJECT_LOG.md
Sec. 46): a first version ran everything in one process and progressively
slowed down 1.6x, then 5.2x per successive config -- traced to JAX/XLA's
per-shape compilation cache accumulating across configs with no release
within the process, pushing the system into heavy swap use (confirmed via
`vm.swapusage`, not guessed -- 78% of swap in use, with the offending
process's own resident memory a modest 300MB, meaning the pressure was
structural/cumulative, not a simple leak in this script's own data).
Every config here is a fresh `python` process, matching the pattern used
successfully throughout this project's other long runs.

Usage:
    python scripts/run_phase4_experiments.py [--point B|C|both] [--quick]

--quick uses 6 seeds everywhere (a fast smoke-check that the pipeline
still works end to end) instead of the full 20/6-seed split the
manuscript reports; the output file records which mode was used.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"
THIS_FILE = Path(__file__).resolve()

POINT_B = dict(Fr=0.5, Lam=0.001, ki=11.0, ke=3.0, Nsub_bt=14.142794816,
               nsub_range=(11.0, 14.10), n_nsub=60,
               nsub_grid=(11.0, 13.0, 10, 13.0, 14.12, 20))  # two linspace segments, flattened
POINT_C = dict(Fr=0.3, Lam=0.001, ki=6.0, ke=5.0, Nsub_bt=7.630101792,
               nsub_range=(3.5, 7.5), n_nsub=60,
               nsub_grid=(4.7, 6.5, 8, 6.5, 7.55, 10))

# (config_name, train_fn_name, train_kwargs) -- shared by both the
# orchestrator (to know what to launch) and the worker (to know what to run).
CONFIGS_B = [
    ("baseline", "train_surrogate", dict(hidden_sizes=(64, 64), n_epochs=10000)),
    ("capacity_128x3", "train_surrogate", dict(hidden_sizes=(128, 128, 128), n_epochs=10000)),
    ("capacity_128x4", "train_surrogate", dict(hidden_sizes=(128, 128, 128, 128), n_epochs=10000)),
    ("capacity_256x3", "train_surrogate", dict(hidden_sizes=(256, 256, 256), n_epochs=10000)),
    ("combined_fix", "train_surrogate_boundary_weighted",
     dict(hidden_sizes=(128, 128, 128), eps_w=0.1, n_epochs=10000)),
]
CONFIGS_C = [
    ("baseline", "train_surrogate", dict(hidden_sizes=(64, 64), n_epochs=10000)),
    ("combined_fix", "train_surrogate_boundary_weighted",
     dict(hidden_sizes=(128, 128, 128), eps_w=0.1, n_epochs=10000)),
]


def _nsub_grid_from_tuple(t):
    a, b, na, c, d, nb = t
    return np.concatenate([np.linspace(a, b, na), np.linspace(c, d, nb)])


def _provenance():
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=THIS_FILE.parents[1],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "not-a-git-repo"
    return dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_commit=git_hash,
        lockfile="requirements-lock.txt",
    )


def _run_single_worker(point_name, config_idx, n_seeds):
    """Runs INSIDE its own fresh subprocess -- imports happen here, not at
    module level, so each invocation gets a clean JAX/XLA state."""
    from scipy import stats as sstats

    from tide.eval.bt_decoupling import run_bt_decoupling_analysis
    from tide.surrogates.boundary_weighted_mlp import train_surrogate_boundary_weighted
    from tide.surrogates.bt_point_data import generate_bt_dataset
    from tide.surrogates.mlp import train_surrogate, train_test_split

    train_fns = dict(train_surrogate=train_surrogate,
                      train_surrogate_boundary_weighted=train_surrogate_boundary_weighted)

    point_cfg = POINT_B if point_name == "B" else POINT_C
    configs = CONFIGS_B if point_name == "B" else CONFIGS_C
    name, fn_name, kwargs = configs[config_idx]
    train_fn = train_fns[fn_name]
    nsub_grid = _nsub_grid_from_tuple(point_cfg["nsub_grid"])

    data = generate_bt_dataset(point_cfg["Fr"], point_cfg["Lam"], point_cfg["ki"],
                                point_cfg["ke"], N1=16, nsub_range=point_cfg["nsub_range"],
                                n_nsub=point_cfg["n_nsub"])
    near_list, far_list = [], []
    t0 = time.time()
    for seed in range(n_seeds):
        train, _ = train_test_split(data, seed=seed)
        surrogate, _ = train_fn(train, seed=seed, **kwargs)
        _, summary = run_bt_decoupling_analysis(
            surrogate, point_cfg["Fr"], point_cfg["Lam"], point_cfg["ki"], point_cfg["ke"],
            N1=16, nsub_bt=point_cfg["Nsub_bt"], nsub_grid=nsub_grid)
        near_list.append(summary["upper_error_near"])
        far_list.append(summary["upper_error_far"])
    elapsed = time.time() - t0

    near, far = np.array(near_list), np.array(far_list)
    valid = np.isfinite(near) & np.isfinite(far)
    near, far = near[valid], far[valid]
    stat, p = sstats.wilcoxon(near - far, alternative="greater")
    result = dict(
        config_name=name,
        n_seeds=n_seeds,
        n_valid=int(valid.sum()),
        per_seed_near=near_list,
        per_seed_far=far_list,
        mean_near=float(near.mean()),
        mean_far=float(far.mean()),
        ratio=float(near.mean() / far.mean()),
        p_value=float(p),
        elapsed_seconds=round(elapsed, 1),
        train_kwargs={k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
    )
    # Print a single JSON line prefixed with a marker, so the orchestrating
    # parent process can find it amid this worker's own stdout without
    # relying on it being the only thing printed.
    print("RESULT_JSON:" + json.dumps(result))


def _run_config_subprocess(point_name, config_idx, n_seeds, config_display_name):
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(THIS_FILE), "--_worker", "--point", point_name,
         "--config-idx", str(config_idx), "--n-seeds", str(n_seeds)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  {config_display_name}: SUBPROCESS FAILED (exit {proc.returncode})", flush=True)
        print(proc.stderr[-4000:], flush=True)
        raise RuntimeError(f"worker subprocess failed for {point_name}/{config_display_name}")
    result_line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:"))
    result = json.loads(result_line[len("RESULT_JSON:"):])
    elapsed = time.time() - t0
    print(f"  {config_display_name}: near={result['mean_near']:.5f} far={result['mean_far']:.5f} "
          f"ratio={result['ratio']:.3f} p={result['p_value']:.4g} ({elapsed:.0f}s wall)", flush=True)
    return result


def run_point(point_name, n_seeds_full, n_seeds_quick, quick):
    print(f"=== Point {point_name} ===", flush=True)
    n = n_seeds_quick if quick else n_seeds_full
    configs = CONFIGS_B if point_name == "B" else CONFIGS_C
    results = {}
    for idx, (name, fn_name, kwargs) in enumerate(configs):
        results[name] = _run_config_subprocess(point_name, idx, n, name)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", choices=["B", "C", "both"], default="both")
    parser.add_argument("--quick", action="store_true",
                         help="6 seeds everywhere, for a fast pipeline smoke-check")
    # Internal flags used only when this script re-invokes itself as a worker:
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config-idx", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-seeds", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _run_single_worker(args.point, args.config_idx, args.n_seeds)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = dict(provenance=_provenance(), mode="quick" if args.quick else "full")

    if args.point in ("B", "both"):
        output["point_B"] = run_point("B", n_seeds_full=20, n_seeds_quick=6, quick=args.quick)
    if args.point in ("C", "both"):
        output["point_C"] = run_point("C", n_seeds_full=6, n_seeds_quick=6, quick=args.quick)

    suffix = "quick" if args.quick else "full"
    out_path = OUT_DIR / f"phase4_results_{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
