#!/usr/bin/env bash
# Gets the repo onto the cluster. Lightweight (network I/O only, no
# compute) -- run this directly on the login node, same as your C2PD-HPC
# pattern where getting code onto the cluster is separate from, and does
# not happen inside, the job that builds the environment.
set -Eeuo pipefail

ROOT="${TIDE_ROOT:-/home/barathula.sreeram/Python_Stuff/Fresh_TIDE}"
VENDOR="$ROOT/repo"
BRANCH="main"
REPO="${GIT_REPO_URL:-git@github.com:sreerambarathula/Fresh_TIDE.git}"

mkdir -p "$ROOT"
if [[ -d "$VENDOR/.git" ]]; then
  git -C "$VENDOR" fetch --all --prune
  git -C "$VENDOR" checkout "$BRANCH"
  git -C "$VENDOR" reset --hard "origin/$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO" "$VENDOR"
fi
echo "Repo ready at $VENDOR, commit $(git -C "$VENDOR" rev-parse HEAD)"
