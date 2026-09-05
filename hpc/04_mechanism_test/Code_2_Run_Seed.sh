#!/usr/bin/env bash
# Section 5.3 mechanism test -- one array task per seed (20 total). Each
# seed's worker (scripts/run_mechanism_test.py --_worker --seed N) already
# exists as a standalone unit, so per-seed array parallelism is the
# natural, simplest granularity here -- no batching needed.
set -Eeuo pipefail

PACKAGE="04_mechanism_test"
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
    { echo "TIDE PACKAGE 04 - TASK $PBS_ARRAY_INDEX"; echo "STATUS: FAIL";
      echo "ERROR: See error.log and console.log in $TASKDIR"; } > "$TASKDIR/UPLOAD_THIS.txt"
    exit "$status"
}
trap on_error ERR

export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SEED="$PBS_ARRAY_INDEX"
{
    echo "TIDE Package 04, seed $SEED"
    echo "Host: $(hostname)"
    "$PYTHON" --version
    cd "$VENDOR"
    "$PYTHON" scripts/run_mechanism_test.py --_worker --seed "$SEED" \
        > "$TASKDIR/worker_stdout.log"
    RESULT_LINE="$(grep '^RESULT_JSON:' "$TASKDIR/worker_stdout.log")"
    echo "${RESULT_LINE#RESULT_JSON:}" > "$TASKDIR/result.json"
} > >(tee -a "$TASKDIR/console.log") 2> >(tee -a "$TASKDIR/error.log" >&2)

printf 'PASS\n' > "$TASKDIR/PASS"
{ echo "TIDE PACKAGE 04 - TASK $SEED"; echo "STATUS: PASS";
  echo "RESULT: $TASKDIR/result.json"; } > "$TASKDIR/UPLOAD_THIS.txt"
