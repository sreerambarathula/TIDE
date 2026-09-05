#!/usr/bin/env bash
# THE one-click entry point: submits all 4 analysis streams (Table 3/4,
# point verification, Point A analysis, mechanism test), each stream's own
# merge job (dependent on that stream's array finishing), and one final
# master-report job dependent on ALL FOUR streams succeeding. Nothing to
# run by hand after this.
#
# Prerequisite: run 01_tide_phase4_rerun/Code_1_Setup.sh once first (clones
# the repo, builds the shared venv, runs the base test suite). This script
# checks for that and refuses to proceed if it hasn't completed.
#
# Usage (from hpc/ inside the cloned repo, on the login node):
#   bash submit_everything.sh
set -Eeuo pipefail

ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}"
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SETUP_REPORT="$ROOT/results/01_tide_phase4_rerun/SETUP_THIS.txt"
if [[ ! -f "$SETUP_REPORT" ]] || ! grep -q "STATUS: PASS" "$SETUP_REPORT"; then
    echo "Setup has not completed successfully yet (checked $SETUP_REPORT)." >&2
    echo "Run this first:" >&2
    echo "  cd $HPC_DIR/01_tide_phase4_rerun && bash Code_1_Setup.sh" >&2
    exit 1
fi

echo "=== Submitting all 4 analysis streams ==="

cd "$HPC_DIR/01_tide_phase4_rerun"
ARRAY1="$(qsub submit_array.pbs)"
MERGE1="$(qsub -W "depend=afterokarray:${ARRAY1}[]" submit_merge.pbs)"
echo "01 Table 3/4 (22 tasks):       array=$ARRAY1  merge=$MERGE1"

cd "$HPC_DIR/02_point_verification"
JOB2="$(qsub submit.pbs)"
echo "02 Point verification (1 job): job=$JOB2"

cd "$HPC_DIR/03_point_a_analysis"
ARRAY3="$(qsub submit_array.pbs)"
MERGE3="$(qsub -W "depend=afterokarray:${ARRAY3}[]" submit_merge.pbs)"
echo "03 Point A analysis (5 tasks): array=$ARRAY3  merge=$MERGE3"

cd "$HPC_DIR/04_mechanism_test"
ARRAY4="$(qsub submit_array.pbs)"
MERGE4="$(qsub -W "depend=afterokarray:${ARRAY4}[]" submit_merge.pbs)"
echo "04 Mechanism test (20 tasks):  array=$ARRAY4  merge=$MERGE4"

cd "$HPC_DIR"
FINAL="$(qsub -W "depend=afterok:${MERGE1}:${JOB2}:${MERGE3}:${MERGE4}" submit_final_report.pbs)"

{
  echo "array1=$ARRAY1"; echo "merge1=$MERGE1"; echo "job2=$JOB2"
  echo "array3=$ARRAY3"; echo "merge3=$MERGE3"
  echo "array4=$ARRAY4"; echo "merge4=$MERGE4"; echo "final=$FINAL"
} > "$ROOT/results/.last_submission_ids"

echo ""
echo "Final master report job: $FINAL (runs only after ALL FOUR streams succeed)"
echo ""
echo "Total concurrent cores if scheduler runs everything at once: ~22+2+5+20 = 49"
echo "(well under your 60-core budget; 1 core per training/eval task to avoid"
echo " thread oversubscription, matching every other job in this pipeline)"
echo ""
echo "Monitor with:  bash Code_Watch_All.sh"
echo "Final output:  $ROOT/results/MASTER_REPORT.txt"
echo "               $ROOT/results/MASTER_SUMMARY.csv"
