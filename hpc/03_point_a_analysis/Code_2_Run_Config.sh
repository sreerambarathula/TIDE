#!/usr/bin/env bash
# Point A (Section 5.1) analysis -- one array task per configuration (5
# total: 3 sampling schemes at 20 seeds, 2 extra architectures at 10 seeds).
# Config-level parallelism (not seed-batched further, unlike the Table 3/4
# job) -- proportionate to this stage's smaller total compute (~80
# seed-trainings vs. Table 3/4's ~112), and simpler; each config's own
# internal seed loop still only takes a few minutes.
set -Eeuo pipefail

PACKAGE="03_point_a_analysis"
ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}"
VENDOR="$ROOT/repo"
ENV="$ROOT/envs/tide_env"
PYTHON="$ENV/bin/python"
RESULTS="$ROOT/results/$PACKAGE"
TASKDIR="$RESULTS/task_${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX not set}"

mkdir -p "$TASKDIR"
rm -f "$TASKDIR/PASS" "$TASKDIR/FAIL" "$TASKDIR/UPLOAD_THIS.txt" "$TASKDIR/result.json"
: > "$TASKDIR/console.log"
: > "$TASKDIR/error.log"

on_error() {
    status=$?
    printf 'Task %s failed with exit status %s\n' "$PBS_ARRAY_INDEX" "$status" | tee -a "$TASKDIR/error.log"
    printf 'FAIL\n' > "$TASKDIR/FAIL"
    { echo "TIDE PACKAGE 03 - TASK $PBS_ARRAY_INDEX"; echo "STATUS: FAIL";
      echo "ERROR: See error.log and console.log in $TASKDIR"; } > "$TASKDIR/UPLOAD_THIS.txt"
    exit "$status"
}
trap on_error ERR

# index -> n_seeds, matching CONFIGS_A in scripts/run_point_a_experiments.py
N_SEEDS=(20 20 20 10 10)
i="$PBS_ARRAY_INDEX"
SEEDS="${N_SEEDS[$i]}"

export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
    echo "TIDE Package 03, task $i (n_seeds=$SEEDS)"
    echo "Host: $(hostname)"
    "$PYTHON" --version
    cd "$VENDOR"
    "$PYTHON" scripts/run_point_a_experiments.py --_worker --config-idx "$i" --n-seeds "$SEEDS" \
        > "$TASKDIR/worker_stdout.log"
    RESULT_LINE="$(grep '^RESULT_JSON:' "$TASKDIR/worker_stdout.log")"
    echo "${RESULT_LINE#RESULT_JSON:}" > "$TASKDIR/result.json"
} > >(tee -a "$TASKDIR/console.log") 2> >(tee -a "$TASKDIR/error.log" >&2)

printf 'PASS\n' > "$TASKDIR/PASS"
{ echo "TIDE PACKAGE 03 - TASK $i"; echo "STATUS: PASS";
  echo "RESULT: $TASKDIR/result.json"; } > "$TASKDIR/UPLOAD_THIS.txt"
