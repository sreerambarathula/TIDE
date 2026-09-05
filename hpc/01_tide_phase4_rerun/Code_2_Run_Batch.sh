#!/usr/bin/env bash
# TIDE Phase-4 rerun -- one array task = one seed-batch of one (point,
# config) pair from Table 3 / Table 4. Splitting each 20-seed Point-B
# config into 4 batches of 5 seeds (and leaving Point C's 6-seed configs
# as single batches) turns 7 fully-serial configs into 22 independent,
# 1-core tasks -- efficient use of a 60-core allocation without
# oversubscribing any single task (each seed's MLP is small; more than
# 1 core per task would not speed up a single seed, only add thread
# contention across the ~22 concurrently-running tasks).
#
# Every batch is executed via Code_2b_run_seed_batch.py, which reuses
# run_phase4_experiments.py's own config tables and per-seed procedure
# unmodified -- this is the same code path the manuscript's numbers are
# supposed to come from, just fanned out across more, smaller processes
# instead of run_phase4_experiments.py's own sequential per-config loop.
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

# index -> (point, config_idx, config_name, seed_list). Point B's 5
# configs x 20 seeds each are split into 4 batches of 5 seeds (indices
# 0-19); Point C's 2 configs x 6 seeds are left as single batches
# (indices 20-21). Must be kept in sync with CONFIGS_B / CONFIGS_C in
# scripts/run_phase4_experiments.py if those ever change.
POINTS=(   B         B         B         B         \
           B         B         B         B         \
           B         B         B         B         \
           B         B         B         B         \
           B         B         B         B         \
           C  C )
CONFIG_IDXS=(0 0 0 0   1 1 1 1   2 2 2 2   3 3 3 3   4 4 4 4   0 1)
CONFIG_NAMES=(baseline baseline baseline baseline \
              capacity_128x3 capacity_128x3 capacity_128x3 capacity_128x3 \
              capacity_128x4 capacity_128x4 capacity_128x4 capacity_128x4 \
              capacity_256x3 capacity_256x3 capacity_256x3 capacity_256x3 \
              combined_fix combined_fix combined_fix combined_fix \
              baseline combined_fix)
SEED_LISTS=("0,1,2,3,4" "5,6,7,8,9" "10,11,12,13,14" "15,16,17,18,19" \
            "0,1,2,3,4" "5,6,7,8,9" "10,11,12,13,14" "15,16,17,18,19" \
            "0,1,2,3,4" "5,6,7,8,9" "10,11,12,13,14" "15,16,17,18,19" \
            "0,1,2,3,4" "5,6,7,8,9" "10,11,12,13,14" "15,16,17,18,19" \
            "0,1,2,3,4" "5,6,7,8,9" "10,11,12,13,14" "15,16,17,18,19" \
            "0,1,2,3,4,5" "0,1,2,3,4,5")

i="$PBS_ARRAY_INDEX"
POINT="${POINTS[$i]}"
CFG_IDX="${CONFIG_IDXS[$i]}"
CFG_NAME="${CONFIG_NAMES[$i]}"
SEEDS="${SEED_LISTS[$i]}"

# Single-core task: pin every threaded library to 1 so ~22 of these
# running concurrently on the same allocation don't fight each other for
# cores (matches the C2PD-HPC convention of OMP_NUM_THREADS=$NCPUS with
# OPENBLAS/MKL/NUMEXPR pinned to 1, here with NCPUS itself fixed at 1).
export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
    echo "TIDE Package 01, task $i: Point $POINT / $CFG_NAME, seeds [$SEEDS]"
    echo "Host: $(hostname)"
    echo "PBS job: ${PBS_JOBID:-interactive}"
    "$PYTHON" --version

    cd "$VENDOR"
    "$PYTHON" "$VENDOR/hpc/01_tide_phase4_rerun/Code_2b_run_seed_batch.py" \
        --point "$POINT" --config-idx "$CFG_IDX" --seeds "$SEEDS" \
        > "$TASKDIR/worker_stdout.log"

    RESULT_LINE="$(grep '^RESULT_JSON:' "$TASKDIR/worker_stdout.log")"
    echo "${RESULT_LINE#RESULT_JSON:}" > "$TASKDIR/result.json"

    "$PYTHON" -c "
import json
with open('$TASKDIR/result.json') as f:
    r = json.load(f)
print(f\"point={r['point']} config={r['config_name']} seeds={r['seeds']} \"
      f\"near={r['per_seed_near']} far={r['per_seed_far']} elapsed={r['elapsed_seconds']}s\")
"
} > >(tee -a "$TASKDIR/console.log") 2> >(tee -a "$TASKDIR/error.log" >&2)

printf 'PASS\n' > "$TASKDIR/PASS"
{
    echo "TIDE PACKAGE 01 - TASK $i"
    echo "STATUS: PASS"
    echo "POINT: $POINT"
    echo "CONFIG: $CFG_NAME (index $CFG_IDX)"
    echo "SEEDS: $SEEDS"
    echo "RESULT: $TASKDIR/result.json"
    echo "NEXT: once all 22 tasks show PASS, Code_3_Merge_Results.py runs automatically"
    echo "      (submitted as a dependent job -- see submit_all.sh)"
} > "$TASKDIR/UPLOAD_THIS.txt"
