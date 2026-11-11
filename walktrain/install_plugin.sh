#!/usr/bin/env bash
# Install the standalone WalkTrain brain plugin + brain into BombSquad mods.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODS="${1:-${BALLISTICA_MODS:-$HOME/.ballisticakit/mods}}"
BRAIN_SRC="${WALKTRAIN_BRAIN:-$ROOT/output/spaz_fight.brain}"
PLUGIN_SRC="$ROOT/walktrain/walktrain_brain.py"

mkdir -p "$MODS/brains"

# Remove the old multi-file package install if present (avoids shadowing /
# duplicate plugin entries).
if [[ -d "$MODS/walktrain" ]]; then
  rm -rf "$MODS/walktrain"
  echo "Removed old package install → $MODS/walktrain"
fi

# Drop legacy walk brain if present.
rm -f "$MODS/brains/spaz_walk.brain"

cp -f "$PLUGIN_SRC" "$MODS/walktrain_brain.py"
echo "Installed plugin → $MODS/walktrain_brain.py"

if [[ -f "$BRAIN_SRC" ]]; then
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m walktrain.stamp_meta "$BRAIN_SRC" || true
  cp -f "$BRAIN_SRC" "$MODS/brains/spaz_fight.brain"
  echo "Installed brain → $MODS/brains/spaz_fight.brain"
else
  echo "WARNING: no brain at $BRAIN_SRC — train first or copy one to $MODS/brains/"
fi

echo "Enable: Settings → Plugins → walktrain_brain.WalkTrainBrainPlugin"
echo "Play: chat /brainbot (flat Football Stadium duel from brain meta)"
