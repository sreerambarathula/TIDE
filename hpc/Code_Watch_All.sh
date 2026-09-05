#!/usr/bin/env bash
# One-shot (or watch-able) dashboard across ALL FOUR analysis streams plus
# the final master report. Run any time after submit_everything.sh.
#
#   bash Code_Watch_All.sh              # snapshot
#   watch -n 30 bash Code_Watch_All.sh  # auto-refresh
set -Eeuo pipefail

ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}"
RESULTS="$ROOT/results"

echo "=== PBS queue state (your jobs) ==="
qstat -u "$(whoami)" -t 2>/dev/null || echo "(qstat not available or no jobs queued)"
echo ""

_count_tasks() {
    local dir="$1" total="$2"
    local pass=0 fail=0 other=0
    for i in $(seq 0 $((total - 1))); do
        d="$dir/task_$i"
        if [[ -f "$d/PASS" ]]; then pass=$((pass+1))
        elif [[ -f "$d/FAIL" ]]; then fail=$((fail+1))
        else other=$((other+1)); fi
    done
    echo "$pass PASS / $fail FAIL / $other pending (of $total)"
}

echo "=== 01 Table 3/4 rerun (22 tasks) ==="
if [[ -d "$RESULTS/01_tide_phase4_rerun" ]]; then
    _count_tasks "$RESULTS/01_tide_phase4_rerun" 22
    [[ -f "$RESULTS/01_tide_phase4_rerun/FINAL_REPORT.txt" ]] && echo "  Merged: yes"
else
    echo "  not started"
fi

echo ""
echo "=== 02 Point verification (1 job) ==="
if [[ -f "$RESULTS/02_point_verification/PASS" ]]; then
    echo "  PASS"
elif [[ -f "$RESULTS/02_point_verification/FAIL" ]]; then
    echo "  FAIL"
else
    echo "  pending / not started"
fi

echo ""
echo "=== 03 Point A analysis (5 tasks) ==="
if [[ -d "$RESULTS/03_point_a_analysis" ]]; then
    _count_tasks "$RESULTS/03_point_a_analysis" 5
    [[ -f "$RESULTS/03_point_a_analysis/FINAL_REPORT.txt" ]] && echo "  Merged: yes"
else
    echo "  not started"
fi

echo ""
echo "=== 04 Mechanism test (20 tasks) ==="
if [[ -d "$RESULTS/04_mechanism_test" ]]; then
    _count_tasks "$RESULTS/04_mechanism_test" 20
    [[ -f "$RESULTS/04_mechanism_test/FINAL_REPORT.txt" ]] && echo "  Merged: yes"
else
    echo "  not started"
fi

echo ""
if [[ -f "$RESULTS/MASTER_REPORT.txt" ]]; then
    echo "=== MASTER_REPORT.txt is ready ==="
    cat "$RESULTS/MASTER_REPORT.txt"
else
    echo "=== MASTER_REPORT.txt not written yet (waits for all 4 streams) ==="
fi
