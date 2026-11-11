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

export WALKTRAIN_BRAIN="${WALKTRAIN_BRAIN:-$REPO_ROOT/output/spaz_fight.brain}"
export WALKTRAIN_SPEED="${WALKTRAIN_SPEED:-3}"
export WALKTRAIN_SAVE_EVERY="${WALKTRAIN_SAVE_EVERY:-5}"
mkdir -p "$(dirname "$WALKTRAIN_BRAIN")"

# Prefab headless dynamically links libpython3.14.so.1.0. Codespaces/MS
# CPython is often static-only; conda (and some uv builds) ship the .so.
# LD_LIBRARY_PATH alone is not enough: PYTHONHOME must match that lib's
# prefix or the interpreter segfaults mixing stdlibs.
_ensure_libpython() {
  if ldconfig -p 2>/dev/null | grep -q 'libpython3\.14\.so\.1\.0'; then
    return 0
  fi
  local cand prefix
  for cand in \
    "${LIBPYTHON_DIR:-}" \
    /opt/conda/lib \
    "${CONDA_PREFIX:-}/lib" \
    /usr/local/python/3.14.*/lib \
    "${HOME}/.local/share/uv/python"/*/lib; do
    [[ -z "$cand" || "$cand" == "/lib" ]] && continue
    if [[ -f "$cand/libpython3.14.so.1.0" || -f "$cand/libpython3.14.so" ]]; then
      export LD_LIBRARY_PATH="${cand}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      prefix="$(cd "$cand/.." && pwd)"
      # Only set PYTHONHOME when this tree looks like a real prefix.
      if [[ -d "$prefix/lib/python3.14" || -d "$prefix/bin" ]]; then
        export PYTHONHOME="${PYTHONHOME:-$prefix}"
      fi
      echo "Using libpython from: $cand (PYTHONHOME=${PYTHONHOME:-unset})"
      return 0
    fi
  done
  cand="$(find /opt/conda /usr/local /home -name 'libpython3.14.so.1.0' 2>/dev/null | head -1 || true)"
  if [[ -n "$cand" ]]; then
    export LD_LIBRARY_PATH="$(dirname "$cand")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    prefix="$(cd "$(dirname "$cand")/.." && pwd)"
    if [[ -d "$prefix/lib/python3.14" ]]; then
      export PYTHONHOME="${PYTHONHOME:-$prefix}"
    fi
    echo "Using libpython from: $(dirname "$cand") (PYTHONHOME=${PYTHONHOME:-unset})"
    return 0
  fi
  echo "WARNING: libpython3.14.so.1.0 not found; binary may fail to start." >&2
  echo "Set LIBPYTHON_DIR (and matching PYTHONHOME) to a shared CPython 3.14." >&2
}

_ensure_libpython

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
# Plugin install copies a playback-only walktrain/ into ~/.ballisticakit/mods.
# Ballistica imports that first (no boot.py), which shadows this repo's package.
export WALKTRAIN_PYTHON_ROOT="$PYTHON_ROOT"
EXEC='import os,sys; r=os.environ["WALKTRAIN_PYTHON_ROOT"]; sys.path.insert(0,r); [sys.modules.pop(k) for k in list(sys.modules) if k=="walktrain" or k.startswith("walktrain.")]; import walktrain.boot as _wtb; _wtb.main()'
cd "$RUN_CWD"
exec "$BIN" --exec "$EXEC"
