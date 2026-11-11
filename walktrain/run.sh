#!/usr/bin/env bash
# Launch headless Spaz walk training.
# Works when this package lives in the standalone ballistica-walktrain repo
# (BALLISTICA_ROOT → engine checkout) or inside a ballistica tree.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# If walktrain lives inside a ballistica checkout, parent is the engine root.
if [[ -f "$REPO_ROOT/Makefile" && -d "$REPO_ROOT/src/ballistica" ]]; then
  BALLISTICA_ROOT="${BALLISTICA_ROOT:-$REPO_ROOT}"
  # PYTHONPATH should still include REPO_ROOT only if walktrain is importable;
  # when nested as ballistica/walktrain, parent already on path via cwd tricks —
  # prefer adding REPO_ROOT's parent… Actually package is REPO_ROOT/walktrain
  # only if structure is repo/walktrain/*.py. Nested ballistica/walktrain means
  # REPO_ROOT=ballistica and import walktrain works with PYTHONPATH=ballistica.
  PYTHON_ROOT="$REPO_ROOT"
else
  BALLISTICA_ROOT="${BALLISTICA_ROOT:-$REPO_ROOT/../ballistica}"
  PYTHON_ROOT="$REPO_ROOT"
fi

BALLISTICA_ROOT="$(cd "$BALLISTICA_ROOT" 2>/dev/null && pwd)" || {
  echo "BALLISTICA_ROOT not found: set it to your efroemling/ballistica checkout" >&2
  exit 1
}

export WALKTRAIN_BRAIN="${WALKTRAIN_BRAIN:-$REPO_ROOT/output/spaz_walk.brain}"
export WALKTRAIN_SPEED="${WALKTRAIN_SPEED:-3}"
export WALKTRAIN_SAVE_EVERY="${WALKTRAIN_SAVE_EVERY:-5}"
mkdir -p "$(dirname "$WALKTRAIN_BRAIN")"

BIN=""
while IFS= read -r candidate; do
  if [[ -x "$candidate" ]]; then
    BIN="$candidate"
    break
  fi
done < <(find "$BALLISTICA_ROOT/build" -type f -name ballisticakit_headless 2>/dev/null | sort)

if [[ -z "$BIN" ]]; then
  echo "No headless binary under $BALLISTICA_ROOT/build — building prefab server (debug)…"
  export PATH="${HOME}/.local/bin:${PATH}"
  export BA_APP_RUN_USE_PREFAB="${BA_APP_RUN_USE_PREFAB:-1}"
  make -C "$BALLISTICA_ROOT" prefab-server-debug-build
  while IFS= read -r candidate; do
    if [[ -x "$candidate" ]]; then
      BIN="$candidate"
      break
    fi
  done < <(find "$BALLISTICA_ROOT/build" -type f -name ballisticakit_headless 2>/dev/null | sort)
fi

if [[ -z "${BIN}" || ! -x "$BIN" ]]; then
  echo "Could not locate ballisticakit_headless under $BALLISTICA_ROOT" >&2
  exit 1
fi

# Staged layout: …/dist/ballisticakit_headless with ba_data beside dist or inside dist.
BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
if [[ -d "$BIN_DIR/ba_data" ]]; then
  RUN_CWD="$BIN_DIR"
elif [[ -d "$BIN_DIR/../ba_data" ]]; then
  RUN_CWD="$(cd "$BIN_DIR/.." && pwd)"
else
  RUN_CWD="$BIN_DIR"
fi

echo "Binary: $BIN"
echo "Run cwd: $RUN_CWD"
echo "Brain:  $WALKTRAIN_BRAIN"
echo "Speed:  2^${WALKTRAIN_SPEED}x wall clock (sim step still 8ms)"
echo "Ctrl-C stops; brain saves every ${WALKTRAIN_SAVE_EVERY} episodes and ~30s wall."

export PYTHONPATH="${PYTHON_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
EXEC="import walktrain.boot; walktrain.boot.main()"
cd "$RUN_CWD"
exec "$BIN" --exec "$EXEC"
