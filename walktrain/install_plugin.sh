#!/usr/bin/env bash
# Install walktrain + brain file into BombSquad / BallisticaKit mods.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODS="${1:-${BALLISTICA_MODS:-$HOME/.ballisticakit/mods}}"
BRAIN_SRC="${WALKTRAIN_BRAIN:-$ROOT/output/spaz_walk.brain}"

mkdir -p "$MODS/brains"

# Package (plugin lives at walktrain.plugin.WalkTrainBrainPlugin).
# Only ship playback pieces into mods — not headless trainer boot/run helpers.
rm -rf "$MODS/walktrain"
mkdir -p "$MODS/walktrain"
for f in __init__.py brain.py player.py plugin.py; do
  cp -a "$ROOT/walktrain/$f" "$MODS/walktrain/"
done

if [[ -f "$BRAIN_SRC" ]]; then
  # Ensure metadata is present before copying.
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m walktrain.stamp_meta "$BRAIN_SRC" || true
  cp -f "$BRAIN_SRC" "$MODS/brains/spaz_walk.brain"
  echo "Installed brain → $MODS/brains/spaz_walk.brain"
else
  echo "WARNING: no brain at $BRAIN_SRC — train first or copy one to $MODS/brains/"
fi

echo "Installed plugin package → $MODS/walktrain"
echo "Enable: Settings → Plugins → walktrain.plugin.WalkTrainBrainPlugin"
echo "Play: host a game on The Pad, then chat /brainbot"
