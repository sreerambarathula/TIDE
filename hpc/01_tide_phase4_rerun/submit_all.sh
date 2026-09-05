#!/usr/bin/env bash
# Convenience wrapper: submits the 22-task array job, then submits the
# merge job with a dependency on the WHOLE array succeeding
# (afterokarray:<jobid>[]) so the final PASS/FAIL report is produced
# automatically -- no need to watch qstat and merge by hand.
#
# Run this from inside hpc/01_tide_phase4_rerun/ on the login node, after
# Code_1_Setup.sh has already completed with STATUS: PASS.
set -Eeuo pipefail

SETUP_REPORT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}/results/01_tide_phase4_rerun/SETUP_THIS.txt"
if [[ ! -f "$SETUP_REPORT" ]] || ! grep -q "STATUS: PASS" "$SETUP_REPORT"; then
    echo "Setup has not completed successfully yet (checked $SETUP_REPORT)." >&2
    echo "Run Code_1_Setup.sh first and confirm it reports STATUS: PASS." >&2
    exit 1
fi

ARRAY_JOB="$(qsub submit_array.pbs)"
echo "Array job submitted: $ARRAY_JOB (22 tasks, indices 0-21)"

MERGE_JOB="$(qsub -W "depend=afterokarray:${ARRAY_JOB}" submit_merge.pbs)"
echo "Merge job submitted: $MERGE_JOB (runs only after all 22 array tasks succeed)"

echo ""
echo "Monitor with: qstat -t $ARRAY_JOB"
echo "Final report will appear at:"
echo "  \$TIDE_ROOT/results/01_tide_phase4_rerun/FINAL_REPORT.txt"
