#!/bin/bash
# Interactive or Local Runner for TIDE Master Pipeline
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CORES=${1:-60}
MODE=${2:-full}

echo "Running TIDE Master Pipeline with $CORES cores in '$MODE' mode..."
python3 -u run_master_pipeline.py --mode "$MODE" --cores "$CORES" 2>&1 | tee -a TIDE_MASTER_RUN.log
