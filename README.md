# ballistica-walktrain

Headless trainer: spawn **two Spazzes** on a **flat Football Stadium** (no
powerups, no flags) and learn to **fight** with **real Ballistica physics**.
Each agent sees the other's **3D node position**; more **momentum** → more
punch damage (stock Spaz `HitMessage` / impulse API). Progress saves to a
resumable **`.brain`** checkpoint.

> **Agents / continuing work:** start at [`docs/HANDOFF.md`](docs/HANDOFF.md), then [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Status

Walk-no-fall training on The Pad was retired. Current task is **self-play
duel** on Football Stadium (`task=duel_fight`, brain v2). Checkpoint default:
`output/spaz_fight.brain`.

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

Checkpoint default: `output/spaz_fight.brain`. Rerun the same command to **resume**.

| Env var | Default | Meaning |
|---------|---------|---------|
| `WALKTRAIN_BRAIN` | `output/spaz_fight.brain` | Checkpoint path |
| `WALKTRAIN_SPEED` | `3` single / `8` parallel | Wall-clock `2**n` via debug speed exponent |
| `WALKTRAIN_WORKERS` | `nproc` (max 8) | Parallel headless processes (`run_parallel.sh`) |
| `WALKTRAIN_SAVE_EVERY` | `5` / `2` parallel | Episodes between saves |
| `WALKTRAIN_SHARED` | `1` | Locked shared-brain pull/push between workers |
| `BALLISTICA_ROOT` | `../ballistica` | Engine checkout for `run.sh` |

## Package layout

```
walktrain/brain.py              # MLP + Adam + .brain I/O (pure Python)
walktrain/activity.py           # Session/Activity, 2 Spazzes, duel rewards
walktrain/boot.py               # --exec entrypoint
walktrain/walktrain_brain.py    # Standalone BombSquad API 9 playback plugin
walktrain/run.sh                # Launch helper
```

## Play a brain in BombSquad 1.8 (API 9)

`.brain` files store **flat-map metadata** (`flat_map`, `no_powerups`,
`no_flags`, map name). `/brainbot` reads those rules, starts a teams Death
Match on Football Stadium without powerup rain, and spawns **two** fighting
bots on opposite sides.

```bash
./walktrain/install_plugin.sh          # default: ~/.ballisticakit/mods
```

1. Enable **walktrain_brain.WalkTrainBrainPlugin** under Settings → Plugins.
2. Chat **`/brainbot`** — flat stadium duel from brain meta.
3. **`/brainbot stop`** / **`/brainbot info`**.

Default brain path: `mods/brains/spaz_fight.brain`.

## License

Trainer code in this repository: MIT (same spirit as Ballistica’s MIT sources).  
Ballistica engine/assets: follow [efroemling/ballistica](https://github.com/efroemling/ballistica) licensing — do not redistribute proprietary assets.
