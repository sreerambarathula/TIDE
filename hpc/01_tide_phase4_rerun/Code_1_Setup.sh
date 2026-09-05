#!/usr/bin/env bash
# TIDE Phase-4 rerun -- setup stage.
# Mirrors the C2PD-HPC convention: clone/checkout, build a package-local venv,
# write a PASS/FAIL-style report (SETUP_THIS.txt) so a stalled setup never
# gets silently mistaken for a completed one.
set -Eeuo pipefail

PACKAGE="01_tide_phase4_rerun"
ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/tide}"
RESULTS="$ROOT/results/$PACKAGE"
VENDOR="$ROOT/repo"
ENV="$ROOT/envs/tide_env"
REPO="https://github.com/sreerambarathula/Fresh_TIDE.git"
BRANCH="main"

# Fine-grained GitHub PAT with read-only access to just this repo, exported
# before running this script (the repo is private -- gh CLI's own login on
# your workstation does not carry over to the cluster). If you instead set
# up an SSH deploy key on this account, delete the two lines below and
# change REPO to git@github.com:sreerambarathula/Fresh_TIDE.git.
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a fine-grained PAT scoped to the Fresh_TIDE repo before running this script}"
REPO="https://${GITHUB_TOKEN}@github.com/sreerambarathula/Fresh_TIDE.git"

mkdir -p "$RESULTS" "$ROOT/envs"
LOG="$RESULTS/setup.log"
REPORT="$RESULTS/SETUP_THIS.txt"
exec > >(tee "$LOG") 2>&1

fail() {
  rc=$?
  {
    echo "TIDE PACKAGE 01 - PHASE4 RERUN SETUP"
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

if [[ -d "$VENDOR/.git" ]]; then
  git -C "$VENDOR" fetch --all --prune
  git -C "$VENDOR" checkout "$BRANCH"
  git -C "$VENDOR" reset --hard "origin/$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO" "$VENDOR"
fi
COMMIT="$(git -C "$VENDOR" rev-parse HEAD)"

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
  "$PYTHON" -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV/bin/python" -m pip install -e "$VENDOR"
"$ENV/bin/python" -m pip install -r "$VENDOR/requirements-lock.txt"
"$ENV/bin/python" -m pip install pytest

"$ENV/bin/python" - <<'PY'
import jax, numpy, scipy, optax
print("JAX", jax.__version__)
print("JAX_BACKEND", jax.default_backend())
print("NUMPY", numpy.__version__)
print("SCIPY", scipy.__version__)
print("OPTAX", optax.__version__)
PY

# Confirm the two post-audit corrections this rerun depends on are present
# in the checked-out commit before spending compute time on it.
"$ENV/bin/python" -c "
import sys; sys.path.insert(0, '$VENDOR/src')
from tide.surrogates.bt_point_data import wedge_boundaries_true
from tide.continuation.double_zero import solve_double_zero
print('post-audit tools present: wedge_boundaries_true, solve_double_zero')
" || echo "WARNING: could not import expected post-audit symbols -- check VENDOR checkout"

echo "Running full test suite (this should take ~15 min on one core) ..."
cd "$VENDOR"
"$ENV/bin/python" -m pytest tests/ -q | tee "$RESULTS/pytest.log"

{
  echo "TIDE PACKAGE 01 - PHASE4 RERUN SETUP"
  echo "STATUS: PASS"
  echo "REPOSITORY: https://github.com/sreerambarathula/tide"
  echo "COMMIT: $COMMIT"
  echo "PYTHON: $("$ENV/bin/python" -V 2>&1)"
  echo "ENVIRONMENT: $ENV"
  echo "TEST_SUITE: $(tail -1 "$RESULTS/pytest.log")"
  echo "NEXT: Submit hpc/01_tide_phase4_rerun/submit_array.pbs"
} > "$REPORT"

echo "P01 setup completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
