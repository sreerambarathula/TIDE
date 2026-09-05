#!/usr/bin/env bash
# TIDE Phase-4 rerun -- progress dashboard. Run this any time after
# submit_all.sh to see, in one glance: each array subjob's PBS state,
# a PASS/FAIL/running count across all 22 tasks, and the tail of whichever
# task's console.log was written to most recently (i.e. whatever is
# actively running right now).
#
# Usage:
#   bash Code_4_Watch_Progress.sh            # one-shot snapshot
#   watch -n 30 bash Code_4_Watch_Progress.sh  # auto-refresh every 30s
set -Eeuo pipefail

ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}"
RESULTS="$ROOT/results/01_tide_phase4_rerun"

echo "=== PBS queue state (your jobs) ==="
qstat -u "$(whoami)" -t 2>/dev/null || echo "(qstat not available or no jobs queued)"
echo ""

echo "=== Task summary (0-21) ==="
pass=0; fail=0; running=0; not_started=0
for i in $(seq 0 21); do
    d="$RESULTS/task_$i"
    if [[ -f "$d/PASS" ]]; then
        status="PASS"; pass=$((pass+1))
    elif [[ -f "$d/FAIL" ]]; then
        status="FAIL"; fail=$((fail+1))
    elif [[ -d "$d" ]]; then
        status="RUNNING (or queued)"; running=$((running+1))
    else
        status="not started"; not_started=$((not_started+1))
    fi
    printf "  task_%-3s %s\n" "$i" "$status"
done
echo ""
echo "Totals: $pass PASS / $fail FAIL / $running running-or-queued / $not_started not started (out of 22)"
echo ""

if [[ -f "$RESULTS/FINAL_REPORT.txt" ]]; then
    echo "=== FINAL_REPORT.txt already exists ==="
    cat "$RESULTS/FINAL_REPORT.txt"
    echo ""
fi

LATEST_LOG="$(find "$RESULTS" -maxdepth 2 -name console.log -newermt '-1 hour' 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1 || true)"
if [[ -n "${LATEST_LOG:-}" ]]; then
    echo "=== Tail of most recently updated log: $LATEST_LOG ==="
    tail -n 20 "$LATEST_LOG"
else
    echo "(no console.log updated in the last hour -- nothing actively running, or run just started)"
fi
