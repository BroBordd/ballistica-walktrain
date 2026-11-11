#!/usr/bin/env bash
# Multi-worker WalkTrain: N headless Ballistica sims sharing one .brain
# (Isaac-Gym-style parallelism via processes — ODE is CPU, not GPU).
#
# Usage:
#   export BALLISTICA_ROOT=../ballistica
#   ./walktrain/run_parallel.sh
#
# Env:
#   WALKTRAIN_WORKERS   default: nproc (capped at 8)
#   WALKTRAIN_SPEED     default: 8  → 2^8 = 256× wall clock (debug builds)
#   WALKTRAIN_BRAIN     shared checkpoint path
#   WALKTRAIN_SAVE_EVERY  default: 2
#   WALKTRAIN_MAX_EP    default: 20
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

if [[ -f "$REPO_ROOT/Makefile" && -d "$REPO_ROOT/src/ballistica" ]]; then
  BALLISTICA_ROOT="${BALLISTICA_ROOT:-$REPO_ROOT}"
  PYTHON_ROOT="$REPO_ROOT"
else
  BALLISTICA_ROOT="${BALLISTICA_ROOT:-$REPO_ROOT/../ballistica}"
  PYTHON_ROOT="$REPO_ROOT"
fi

BALLISTICA_ROOT="$(cd "$BALLISTICA_ROOT" 2>/dev/null && pwd)" || {
  echo "BALLISTICA_ROOT not found" >&2
  exit 1
}

NPROC="$(nproc 2>/dev/null || echo 4)"
WORKERS="${WALKTRAIN_WORKERS:-$NPROC}"
if [[ "$WORKERS" -gt 8 ]]; then
  WORKERS=8
fi
if [[ "$WORKERS" -lt 1 ]]; then
  WORKERS=1
fi

export WALKTRAIN_BRAIN="${WALKTRAIN_BRAIN:-$REPO_ROOT/output/spaz_walk.brain}"
export WALKTRAIN_SPEED="${WALKTRAIN_SPEED:-8}"
export WALKTRAIN_SAVE_EVERY="${WALKTRAIN_SAVE_EVERY:-1}"
export WALKTRAIN_MAX_EP="${WALKTRAIN_MAX_EP:-20}"
export WALKTRAIN_SHARED="${WALKTRAIN_SHARED:-1}"
mkdir -p "$(dirname "$WALKTRAIN_BRAIN")"
LOG_DIR="${WALKTRAIN_LOG_DIR:-$REPO_ROOT/output/logs}"
mkdir -p "$LOG_DIR"

_ensure_libpython() {
  if ldconfig -p 2>/dev/null | grep -q 'libpython3\.14\.so\.1\.0'; then
    return 0
  fi
  local cand prefix
  for cand in \
    "${LIBPYTHON_DIR:-}" \
    /opt/conda/lib \
    "${CONDA_PREFIX:-}/lib" \
    /usr/local/python/3.14.*/lib; do
    [[ -z "$cand" || "$cand" == "/lib" ]] && continue
    if [[ -f "$cand/libpython3.14.so.1.0" || -f "$cand/libpython3.14.so" ]]; then
      export LD_LIBRARY_PATH="${cand}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      prefix="$(cd "$cand/.." && pwd)"
      if [[ -d "$prefix/lib/python3.14" || -d "$prefix/bin" ]]; then
        export PYTHONHOME="${PYTHONHOME:-$prefix}"
      fi
      return 0
    fi
  done
}
_ensure_libpython

BIN=""
while IFS= read -r candidate; do
  if [[ -x "$candidate" ]]; then
    BIN="$candidate"
    break
  fi
done < <(find "$BALLISTICA_ROOT/build" -type f -name ballisticakit_headless 2>/dev/null | sort)

if [[ -z "${BIN}" || ! -x "$BIN" ]]; then
  echo "Could not locate ballisticakit_headless under $BALLISTICA_ROOT" >&2
  exit 1
fi

BIN_DIR="$(cd "$(dirname "$BIN")" && pwd)"
if [[ -d "$BIN_DIR/ba_data" ]]; then
  RUN_CWD="$BIN_DIR"
elif [[ -d "$BIN_DIR/../ba_data" ]]; then
  RUN_CWD="$(cd "$BIN_DIR/.." && pwd)"
else
  RUN_CWD="$BIN_DIR"
fi

export PYTHONPATH="${PYTHON_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Plugin install copies a playback-only walktrain/ into ~/.ballisticakit/mods.
# Ballistica imports that first (no boot.py), which shadows this repo's package.
export WALKTRAIN_PYTHON_ROOT="$PYTHON_ROOT"
EXEC='import os,sys; r=os.environ["WALKTRAIN_PYTHON_ROOT"]; sys.path.insert(0,r); [sys.modules.pop(k) for k in list(sys.modules) if k=="walktrain" or k.startswith("walktrain.")]; import walktrain.boot as _wtb; _wtb.main()'

echo "=== WalkTrain parallel ==="
echo "Workers: $WORKERS"
echo "Speed:   2^${WALKTRAIN_SPEED}x wall (sim step still 8ms)"
echo "Brain:   $WALKTRAIN_BRAIN (shared+locked)"
echo "Binary:  $BIN"
echo "Logs:    $LOG_DIR/worker-*.log"
echo "Ctrl-C stops all workers."

PIDS=()
cleanup() {
  echo ""
  echo "Stopping workers: ${PIDS[*]:-}"
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

: >"$LOG_DIR/combined.log"

for ((i = 0; i < WORKERS; i++)); do
  log="$LOG_DIR/worker-$i.log"
  : >"$log"
  (
    # Stagger boots so session hijacks don't collide.
    if [[ "$i" -gt 0 ]]; then
      sleep "$i"
    fi
    cd "$RUN_CWD"
    # Unique UDP port per worker (0 = ephemeral) avoids 43210 clashes.
    export BA_UDP_PORT=0
    export WALKTRAIN_WORKER_ID="$i"
    "$BIN" --exec "$EXEC" 2>&1 | tee -a "$log" "$LOG_DIR/combined.log" | sed -u "s/^/[w$i] /"
  ) &
  PIDS+=("$!")
  echo "started worker $i pid=${PIDS[$i]}"
done

# Wait until all exit (or Ctrl-C).
ec=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    ec=1
  fi
done
trap - INT TERM EXIT
exit "$ec"
