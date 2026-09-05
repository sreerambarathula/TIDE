"""HPC seed-batch worker for the TIDE Phase-4 rerun.

Runs an explicit SUBSET of seeds for one (point, config) pair from Table
3/4, reusing the exact same CONFIGS_B / CONFIGS_C / POINT_B / POINT_C
definitions and per-seed procedure as scripts/run_phase4_experiments.py's
own _run_single_worker -- the only generalization is "range(n_seeds)" ->
an arbitrary explicit seed list, so seeds can be split across many small
HPC array tasks and merged afterward (Code_3_Merge_Results.py) into
exactly the statistics a single un-split run would have produced.

Deliberately does NOT modify run_phase4_experiments.py -- imports its
config tables directly so the two can never silently drift apart. Each
invocation is its own fresh process (same reasoning as the original
script: JAX/XLA per-shape compilation cache growth across configs inside
one long-lived process, PROJECT_LOG.md Sec. 46).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # hpc/01_tide_phase4_rerun/ -> repo root
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", choices=["B", "C"], required=True)
    parser.add_argument("--config-idx", type=int, required=True)
    parser.add_argument("--seeds", type=str, required=True,
                         help="comma-separated explicit seed list, e.g. 0,1,2,3,4")
    args = parser.parse_args()
    seed_list = [int(s) for s in args.seeds.split(",")]

    from run_phase4_experiments import (POINT_B, POINT_C, CONFIGS_B, CONFIGS_C,
                                         _nsub_grid_from_tuple)
    from tide.eval.bt_decoupling import run_bt_decoupling_analysis
    from tide.surrogates.boundary_weighted_mlp import train_surrogate_boundary_weighted
    from tide.surrogates.bt_point_data import generate_bt_dataset
    from tide.surrogates.mlp import train_surrogate, train_test_split

    train_fns = dict(train_surrogate=train_surrogate,
                      train_surrogate_boundary_weighted=train_surrogate_boundary_weighted)

    point_cfg = POINT_B if args.point == "B" else POINT_C
    configs = CONFIGS_B if args.point == "B" else CONFIGS_C
    name, fn_name, kwargs = configs[args.config_idx]
    train_fn = train_fns[fn_name]
    nsub_grid = _nsub_grid_from_tuple(point_cfg["nsub_grid"])

    data = generate_bt_dataset(point_cfg["Fr"], point_cfg["Lam"], point_cfg["ki"],
                                point_cfg["ke"], N1=16, nsub_range=point_cfg["nsub_range"],
                                n_nsub=point_cfg["n_nsub"])

    near_list, far_list = [], []
    t0 = time.time()
    for seed in seed_list:
        train, _ = train_test_split(data, seed=seed)
        surrogate, _ = train_fn(train, seed=seed, **kwargs)
        _, summary = run_bt_decoupling_analysis(
            surrogate, point_cfg["Fr"], point_cfg["Lam"], point_cfg["ki"], point_cfg["ke"],
            N1=16, nsub_bt=point_cfg["Nsub_bt"], nsub_grid=nsub_grid)
        near_list.append(summary["upper_error_near"])
        far_list.append(summary["upper_error_far"])
    elapsed = time.time() - t0

    result = dict(
        point=args.point,
        config_name=name,
        config_idx=args.config_idx,
        seeds=seed_list,
        per_seed_near=near_list,
        per_seed_far=far_list,
        elapsed_seconds=round(elapsed, 1),
        train_kwargs={k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
    )
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
