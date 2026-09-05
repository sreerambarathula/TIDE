#!/usr/bin/env bash
# Stage 1: locate/verify Points A, B, C from scratch. Single job, not an
# array -- this is fast (Point A/B/C solves are seconds; the 265-combination
# transversality sweep is the slow part, still a one-time deterministic
# scan, not seed-dependent training).
set -Eeuo pipefail

PACKAGE="02_point_verification"
ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}"
VENDOR="$ROOT/repo"
ENV="$ROOT/envs/tide_env"
PYTHON="$ENV/bin/python"
RESULTS="$ROOT/results/$PACKAGE"

mkdir -p "$RESULTS"
rm -f "$RESULTS/PASS" "$RESULTS/FAIL" "$RESULTS/UPLOAD_THIS.txt"
: > "$RESULTS/console.log"
: > "$RESULTS/error.log"

on_error() {
    status=$?
    printf 'Point verification failed with exit status %s\n' "$status" | tee -a "$RESULTS/error.log"
    printf 'FAIL\n' > "$RESULTS/FAIL"
    { echo "TIDE PACKAGE 02 - POINT VERIFICATION"; echo "STATUS: FAIL";
      echo "ERROR: See error.log and console.log in $RESULTS"; } > "$RESULTS/UPLOAD_THIS.txt"
    exit "$status"
}
trap on_error ERR

export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${NCPUS:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
    echo "TIDE Package 02: point verification"
    echo "Host: $(hostname)"
    "$PYTHON" --version
    cd "$VENDOR"
    "$PYTHON" scripts/find_and_verify_points.py
} > >(tee -a "$RESULTS/console.log") 2> >(tee -a "$RESULTS/error.log" >&2)

RESULT_FILE="$(ls -t "$VENDOR"/data/generated/point_verification_*.json | head -1)"
cp "$RESULT_FILE" "$RESULTS/result.json"

printf 'PASS\n' > "$RESULTS/PASS"
{ echo "TIDE PACKAGE 02 - POINT VERIFICATION"; echo "STATUS: PASS";
  echo "RESULT: $RESULTS/result.json"; } > "$RESULTS/UPLOAD_THIS.txt"
