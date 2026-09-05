"""Stage 2-4 for Point A (manuscript Section 5.1): the transversal-crossing
negative control. Mirrors run_phase4_experiments.py's structure exactly
(subprocess-per-config, --_worker mode) so it slots into the same HPC
pattern, but for Point A's three sampling schemes and architecture sweep
instead of Point B/C's fix comparison.

Every dataset/training/statistics call here is pre-existing, already-real
code (src/tide/surrogates/data_generation.py, adaptive_sampling.py,
src/tide/eval/decoupling.py, stats.py). This script only orchestrates them,
using the exact experimental design documented in PROJECT_LOG.md Sec.
31/32/34: boundary-band dataset as the primary/architecture-sweep dataset,
uniform and misfit-driven-adaptive as the two sampling-scheme robustness
checks, all at Point A's real-facility parameters (Table 2).

Configurations (name, dataset, hidden_sizes, n_seeds) -- seed counts match
what PROJECT_LOG.md actually reports for each (20 for the three
sampling-scheme comparisons, 10 for the two additional architectures,
per Sec. 31/32):
  1. boundary_band_64x64   (primary)         20 seeds
  2. uniform_64x64         (sampling check)   20 seeds
  3. misfit_adaptive_64x64 (sampling check)   20 seeds
  4. boundary_band_32x32   (architecture)     10 seeds
  5. boundary_band_128x128 (architecture)     10 seeds

Usage:
    python scripts/run_point_a_experiments.py [--quick]
--quick uses 4 seeds everywhere, for a fast end-to-end pipeline check.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"
THIS_FILE = Path(__file__).resolve()

# Point A's real-facility parameters, Table 2.
POINT_A = dict(Fr=0.035, Lam=5.90, ki=6.55, ke=2.03, Nsub_star=29.886,
                nsub_range=(24.0, 36.0))

CONFIGS_A = [
    ("boundary_band_64x64", "boundary_band", (64, 64), 20),
    ("uniform_64x64", "uniform", (64, 64), 20),
    ("misfit_adaptive_64x64", "misfit_adaptive", (64, 64), 20),
    ("boundary_band_32x32", "boundary_band", (32, 32), 10),
    ("boundary_band_128x128", "boundary_band", (128, 128), 10),
]


def _provenance():
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=THIS_FILE.parents[1],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "not-a-git-repo"
    return dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                python_version=platform.python_version(),
                platform=platform.platform(), git_commit=git_hash,
                lockfile="requirements-lock.txt")


def _build_dataset(dataset_name, seed=0):
    """Builds one of the three named Point A datasets from scratch. Every
    call regenerates deterministically from seed=0 -- no caching across
    configs, so each config's subprocess is fully self-contained (same
    reasoning as run_phase4_experiments.py's fresh-subprocess-per-config)."""
    from tide.surrogates.adaptive_sampling import (_sample_candidate_pool,
                                                     measure_misfit_and_augment)
    from tide.surrogates.data_generation import (_leading_real_part,
                                                   generate_stability_dataset)
    from tide.physics.ledinegg_curve import euler_number
    import numpy as np

    p = POINT_A
    if dataset_name == "boundary_band":
        return generate_stability_dataset(p["Fr"], p["Lam"], p["ki"], p["ke"],
                                           N1=16, nsub_range=p["nsub_range"], seed=seed)

    if dataset_name == "uniform":
        # Same plain-uniform candidate pool adaptive_sampling.py uses as its
        # Stage-1 base, used directly as the training set (no misfit-driven
        # augmentation) -- this IS the "uniform (non-window-aware) sampling"
        # scheme from PROJECT_LOG.md Sec. 32.
        n_points = 4800  # comparable total size to generate_stability_dataset's default
        nsub, npch = _sample_candidate_pool(p["nsub_range"], n_points, p["Fr"], p["Lam"],
                                             p["ki"], p["ke"], seed=seed)
        eu = np.array([float(euler_number(pc, ns, p["Fr"], p["Lam"], p["ki"], p["ke"]))
                       for ns, pc in zip(nsub, npch)])
        g_list = [_leading_real_part(ns, pc, p["Fr"], p["Lam"], p["ki"], p["ke"], 16)
                  for ns, pc in zip(nsub, npch)]
        valid = np.array([v is not None for v in g_list])
        g = np.array([v if v is not None else np.nan for v in g_list])
        n_valid = int(valid.sum())
        # fold_at_nsub/hopf_at_nsub aren't meaningful for a plain uniform
        # pool (no boundary-band structure to report per-row) -- NaN-filled,
        # matching measure_misfit_and_augment's own convention for rows it
        # adds (adaptive_sampling.py line ~84), since it requires these keys
        # to exist on base_data even when it never reads their values for
        # uniform-pool rows.
        return dict(nsub=nsub[valid], npch=npch[valid], Eu=eu[valid], g=g[valid],
                    fold_at_nsub=np.full(n_valid, np.nan),
                    hopf_at_nsub=np.full(n_valid, np.nan))

    if dataset_name == "misfit_adaptive":
        # Stage 1: plain uniform base (same as the "uniform" scheme above,
        # smaller, since Stage 2 adds to it) -- per adaptive_sampling.py's
        # own docstring, this is the two-stage template from PROJECT_LOG.md
        # Sec. 33/34.
        base = _build_dataset("uniform", seed=seed)
        augmented, diagnostics = measure_misfit_and_augment(
            base, p["Fr"], p["Lam"], p["ki"], p["ke"], N1=16,
            nsub_range=p["nsub_range"], seed=seed)
        return augmented

    raise ValueError(f"unknown dataset {dataset_name}")


def _run_single_worker(config_idx, n_seeds):
    """Runs INSIDE its own fresh subprocess, same reasoning as
    run_phase4_experiments.py (clean JAX/XLA state per config)."""
    from tide.eval.stats import run_and_summarize

    name, dataset_name, hidden_sizes, _default_seeds = CONFIGS_A[config_idx]
    p = POINT_A
    data = _build_dataset(dataset_name, seed=0)

    nsub_grid = None
    import numpy as np
    nsub_grid = np.linspace(p["nsub_range"][0], p["nsub_range"][1], 60)

    t0 = time.time()
    per_metric, per_seed = run_and_summarize(
        data, p["Fr"], p["Lam"], p["ki"], p["ke"], N1=16,
        nsub_star=p["Nsub_star"], nsub_grid=nsub_grid,
        n_seeds=n_seeds, n_epochs=3000, hidden_sizes=hidden_sizes)
    elapsed = time.time() - t0

    result = dict(
        config_name=name, dataset=dataset_name,
        hidden_sizes=list(hidden_sizes), n_seeds=n_seeds,
        metrics={m: v for m, v in per_metric.items()},
        per_seed={m: {k: [float(x) for x in v] for k, v in d.items()}
                  for m, d in per_seed.items()},
        elapsed_seconds=round(elapsed, 1),
    )
    print("RESULT_JSON:" + json.dumps(result))


def _run_config_subprocess(config_idx, n_seeds):
    name = CONFIGS_A[config_idx][0]
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(THIS_FILE), "--_worker",
         "--config-idx", str(config_idx), "--n-seeds", str(n_seeds)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  {name}: SUBPROCESS FAILED (exit {proc.returncode})", flush=True)
        print(proc.stderr[-4000:], flush=True)
        raise RuntimeError(f"worker subprocess failed for {name}")
    result_line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:"))
    result = json.loads(result_line[len("RESULT_JSON:"):])
    elapsed = time.time() - t0
    hopf = result["metrics"]["hopf_error"]
    print(f"  {name}: hopf_ratio_mean={hopf['ratio_mean']:.3f} "
          f"hopf_ratio_median={hopf['ratio_median']:.3f} p={hopf['p_value']:.4g} "
          f"({elapsed:.0f}s wall)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config-idx", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-seeds", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _run_single_worker(args.config_idx, args.n_seeds)
        return

    output = dict(provenance=_provenance(), mode="quick" if args.quick else "full")
    output["configs"] = {}
    for idx, (name, dataset_name, hidden_sizes, default_seeds) in enumerate(CONFIGS_A):
        n_seeds = 4 if args.quick else default_seeds
        output["configs"][name] = _run_config_subprocess(idx, n_seeds)

    suffix = "quick" if args.quick else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"point_a_results_{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
