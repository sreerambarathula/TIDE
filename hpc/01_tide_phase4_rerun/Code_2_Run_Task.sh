#!/usr/bin/env bash
# TIDE Phase-4 rerun -- one array task = one (point, config) pair from
# Table 3 / Table 4, run with run_phase4_experiments.py's own --_worker
# mode (the exact code path the manuscript's numbers are supposed to come
# from -- no hand-typed "reused from a previous run" shortcuts this time,
# see PROJECT_LOG.md Sec. 46 and scripts/run_remaining_configs.py for what
# went wrong last time).
set -Eeuo pipefail

PACKAGE="01_tide_phase4_rerun"
ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/tide}"
VENDOR="$ROOT/repo"
ENV="$ROOT/envs/tide_env"
PYTHON="$ENV/bin/python"
RESULTS="$ROOT/results/$PACKAGE"
TASKDIR="$RESULTS/task_${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX not set -- run this only as a PBS array subjob}"

mkdir -p "$TASKDIR"
rm -f "$TASKDIR/PASS" "$TASKDIR/FAIL" "$TASKDIR/UPLOAD_THIS.txt" "$TASKDIR/result.json"
: > "$TASKDIR/console.log"
: > "$TASKDIR/error.log"

on_error() {
    status=$?
    printf 'Task %s failed with exit status %s\n' "$PBS_ARRAY_INDEX" "$status" | tee -a "$TASKDIR/error.log"
    printf 'FAIL\n' > "$TASKDIR/FAIL"
    {
        echo "TIDE PACKAGE 01 - TASK $PBS_ARRAY_INDEX"
        echo "STATUS: FAIL"
        echo "ERROR: See error.log and console.log in $TASKDIR"
        echo "NEXT: Send this text and error.log. Do not re-merge until reviewed."
    } > "$TASKDIR/UPLOAD_THIS.txt"
    exit "$status"
}
trap on_error ERR

# index -> (point, config_idx, config_name, n_seeds), matching CONFIGS_B /
# CONFIGS_C in scripts/run_phase4_experiments.py exactly. If that file's
# config lists ever change, this mapping must be updated to match.
POINTS=(B B B B B C C)
CONFIG_IDXS=(0 1 2 3 4 0 1)
CONFIG_NAMES=(baseline capacity_128x3 capacity_128x4 capacity_256x3 combined_fix baseline combined_fix)
N_SEEDS=(20 20 20 20 20 6 6)

i="$PBS_ARRAY_INDEX"
POINT="${POINTS[$i]}"
CFG_IDX="${CONFIG_IDXS[$i]}"
CFG_NAME="${CONFIG_NAMES[$i]}"
SEEDS="${N_SEEDS[$i]}"

export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${NCPUS:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
    echo "TIDE Package 01, task $i: Point $POINT / $CFG_NAME ($SEEDS seeds)"
    echo "Host: $(hostname)"
    echo "PBS job: ${PBS_JOBID:-interactive}"
    "$PYTHON" --version

    cd "$VENDOR"
    # --_worker prints one line "RESULT_JSON:{...}" to stdout; this IS the
    # same code path scripts/run_phase4_experiments.py's own orchestrator
    # uses for each config, just invoked directly per array task instead
    # of via that script's sequential for-loop.
    "$PYTHON" scripts/run_phase4_experiments.py \
        --_worker --point "$POINT" --config-idx "$CFG_IDX" --n-seeds "$SEEDS" \
        > "$TASKDIR/worker_stdout.log"

    RESULT_LINE="$(grep '^RESULT_JSON:' "$TASKDIR/worker_stdout.log")"
    echo "${RESULT_LINE#RESULT_JSON:}" > "$TASKDIR/result.json"

    "$PYTHON" -c "
import json
with open('$TASKDIR/result.json') as f:
    r = json.load(f)
print(f\"point=$POINT config={r['config_name']} n_valid={r['n_valid']}/{r['n_seeds']} \"
      f\"near={r['mean_near']:.5f} far={r['mean_far']:.5f} ratio={r['ratio']:.3f} \"
      f\"p={r['p_value']:.4g} elapsed={r['elapsed_seconds']}s\")
"
} > >(tee -a "$TASKDIR/console.log") 2> >(tee -a "$TASKDIR/error.log" >&2)

printf 'PASS\n' > "$TASKDIR/PASS"
{
    echo "TIDE PACKAGE 01 - TASK $i"
    echo "STATUS: PASS"
    echo "POINT: $POINT"
    echo "CONFIG: $CFG_NAME (index $CFG_IDX)"
    echo "N_SEEDS: $SEEDS"
    echo "RESULT: $TASKDIR/result.json"
    echo "NEXT: once all 7 tasks show PASS, run Code_3_Merge_Results.py"
} > "$TASKDIR/UPLOAD_THIS.txt"
