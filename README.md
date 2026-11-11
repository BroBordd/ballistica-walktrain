# ballistica-walktrain

Headless trainer: spawn a **Spaz** on **The Pad** and learn to walk without falling, using **real Ballistica physics** (same ODE math as the game). Progress saves to a resumable **`.brain`** checkpoint (LLM-style continue).

> **Agents / continuing work:** start at [`docs/HANDOFF.md`](docs/HANDOFF.md), then [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Status

Headless training against real Ballistica physics is **working**: episodes log, falls occur (The Pad railing removed), and `output/spaz_walk.brain` saves/resumes. Policy quality is still being trained/tuned — see [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Requirements

- [Ballistica](https://github.com/efroemling/ballistica) source tree (for headless binary / assets tooling)
- Prefab or cmake **headless** binary: `ballisticakit_headless`
- This repo on `PYTHONPATH`

## Quick start

```bash
# 1) Engine (sibling folder)
git clone --depth 1 https://github.com/efroemling/ballistica.git
cd ballistica && make prefab-server-debug-build   # needs uv + clang-format; see docs/NEXT_STEPS.md

# 2) Trainer — single worker
cd ../ballistica-walktrain
export BALLISTICA_ROOT=../ballistica
./walktrain/run.sh

# Or: parallel workers (recommended) — N headless sims, one shared .brain
./walktrain/run_parallel.sh
```

Checkpoint default: `output/spaz_walk.brain`. Rerun the same command to **resume**.

Parallel mode runs several `ballisticakit_headless` processes (CPU ODE — not a GPU Isaac Gym clone) that Hogwild-update one locked `.brain`, at high debug speed (`2^WALKTRAIN_SPEED`).

| Env var | Default | Meaning |
|---------|---------|---------|
| `WALKTRAIN_BRAIN` | `output/spaz_walk.brain` | Checkpoint path |
| `WALKTRAIN_SPEED` | `3` single / `8` parallel | Wall-clock `2**n` via debug speed exponent |
| `WALKTRAIN_WORKERS` | `nproc` (max 8) | Parallel headless processes (`run_parallel.sh`) |
| `WALKTRAIN_SAVE_EVERY` | `5` / `2` parallel | Episodes between saves |
| `WALKTRAIN_SHARED` | `1` | Locked shared-brain pull/push between workers |
| `BALLISTICA_ROOT` | `../ballistica` | Engine checkout for `run.sh` |

## Package layout

```
walktrain/brain.py              # MLP + Adam + .brain I/O (pure Python)
walktrain/activity.py           # Session/Activity, Spaz, rewards
walktrain/boot.py               # --exec entrypoint
walktrain/walktrain_brain.py    # Standalone BombSquad API 9 playback plugin
walktrain/run.sh                # Launch helper
```

## Play a brain in BombSquad 1.8 (API 9)

`.brain` files now store **metadata** (map trained on, obs scales, task, etc.).
`/brainbot` starts a teams Death Match forced to that map, then spawns the bot.

```bash
# Install one plugin file + brain into BallisticaKit mods
./walktrain/install_plugin.sh          # default: ~/.ballisticakit/mods
# or: ./walktrain/install_plugin.sh /path/to/BombSquad/mods
```

1. Enable **walktrain_brain.WalkTrainBrainPlugin** under Settings → Plugins (restart if needed).
2. Chat **`/brainbot`** — starts a teams Death Match on the brain’s map and spawns the bot.
3. **`/brainbot stop`** removes the bot; **`/brainbot info`** shows brain vs current map.

Default brain path: `mods/brains/spaz_walk.brain`. Override in the plugin settings UI, or stamp an older checkpoint with:

```bash
PYTHONPATH=. python3 -m walktrain.stamp_meta output/spaz_walk.brain
```

## License

Trainer code in this repository: MIT (same spirit as Ballistica’s MIT sources).  
Ballistica engine/assets: follow [efroemling/ballistica](https://github.com/efroemling/ballistica) licensing — do not redistribute proprietary assets.
