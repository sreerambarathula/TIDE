#!/usr/bin/env bash
# TIDE environment build -- run ONLY as a submitted job (submit_setup.pbs),
# never on the login node. Matches C2PD-HPC's 02_C2PD_Env_Setup shape: no
# git operations here at all (that's Code_0_Clone.sh, run separately on
# the login node first, since it's lightweight network I/O, not compute).
set -Eeuo pipefail

PACKAGE="01_tide_phase4_rerun"
ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}"
RESULTS="$ROOT/results/$PACKAGE"
VENDOR="$ROOT/repo"
ENV="$ROOT/envs/tide_env"

mkdir -p "$RESULTS" "$ROOT/envs"
LOG="$RESULTS/setup.log"
REPORT="$RESULTS/SETUP_THIS.txt"
exec > >(tee "$LOG") 2>&1

fail() {
  rc=$?
  {
    echo "TIDE PACKAGE 01 - ENVIRONMENT SETUP"
    echo "STATUS: FAIL"
    echo "EXIT_CODE: $rc"
    echo "FAILED_LINE: ${BASH_LINENO[0]:-UNKNOWN}"
    echo "NEXT: Paste this file here. Do not submit the array job or rerun setup."
    echo
    echo "LOG TAIL"
    echo "--------"
    tail -120 "$LOG" 2>/dev/null || true
  } > "$REPORT"
  exit "$rc"
}
trap fail ERR

echo "P01 setup started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname)"

if [[ ! -d "$VENDOR/.git" ]]; then
  echo "FATAL: $VENDOR does not exist. Run Code_0_Clone.sh on the login node first." >&2
  false
fi
COMMIT="$(git -C "$VENDOR" rev-parse HEAD)"
echo "Using repo at $VENDOR, commit $COMMIT"

PYTHON=""
for candidate in \
  "$(command -v python3.13 2>/dev/null || true)" \
  "$(command -v python3.11 2>/dev/null || true)" \
  "$(command -v python3 2>/dev/null || true)"; do
  if [[ -x "$candidate" ]] && "$candidate" -c 'import sys; raise SystemExit(not sys.version_info[:2] >= (3, 11))'; then
    PYTHON="$candidate"
    break
  fi
done
test -n "$PYTHON"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Creating venv at $ENV ..."
  "$PYTHON" -m venv "$ENV"
fi
echo "Upgrading pip/setuptools/wheel ..."
"$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
echo "Installing tide package (pip install -e .) ..."
"$ENV/bin/python" -m pip install -e "$VENDOR"
echo "Installing locked dependencies (JAX, NumPy, SciPy, Optax -- this is the"
echo "slow step, several minutes with no output is normal) ..."
"$ENV/bin/python" -m pip install -r "$VENDOR/requirements-lock.txt"
echo "Installing pytest ..."
"$ENV/bin/python" -m pip install pytest
echo "Dependency installation complete."

"$ENV/bin/python" - <<'PY'
import jax, numpy, scipy, optax
print("JAX", jax.__version__)
print("JAX_BACKEND", jax.default_backend())
print("NUMPY", numpy.__version__)
print("SCIPY", scipy.__version__)
print("OPTAX", optax.__version__)
PY

"$ENV/bin/python" -c "
import sys; sys.path.insert(0, '$VENDOR/src')
from tide.surrogates.bt_point_data import wedge_boundaries_true
from tide.continuation.double_zero import find_double_zero_point
print('post-audit tools present: wedge_boundaries_true, find_double_zero_point')
" || echo "WARNING: could not import expected post-audit symbols -- check VENDOR checkout"

echo "Running full test suite (this should take ~15 min on one core) ..."
cd "$VENDOR"
"$ENV/bin/python" -m pytest tests/ -q | tee "$RESULTS/pytest.log"

{
  echo "TIDE PACKAGE 01 - ENVIRONMENT SETUP"
  echo "STATUS: PASS"
  echo "REPOSITORY: https://github.com/sreerambarathula/Fresh_TIDE"
  echo "COMMIT: $COMMIT"
  echo "PYTHON: $("$ENV/bin/python" -V 2>&1)"
  echo "ENVIRONMENT: $ENV"
  echo "TEST_SUITE: $(tail -1 "$RESULTS/pytest.log")"
  echo "NEXT: Submit hpc/01_tide_phase4_rerun/submit_array.pbs"
} > "$REPORT"

echo "P01 setup completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
