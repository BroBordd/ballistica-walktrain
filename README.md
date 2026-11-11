# ballistica-walktrain

Headless trainer: spawn a **Spaz** on **The Pad** and learn to walk without falling, using **real Ballistica physics** (same ODE math as the game). Progress saves to a resumable **`.brain`** checkpoint (LLM-style continue).

> **Agents / continuing work:** start at [`docs/HANDOFF.md`](docs/HANDOFF.md), then [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Status

Core Python trainer is implemented. **First in-engine training run was not fully verified** before this repo was published — see handoff docs for the remaining smoke-test checklist.

## Requirements

- [Ballistica](https://github.com/efroemling/ballistica) source tree (for headless binary / assets tooling)
- Prefab or cmake **headless** binary: `ballisticakit_headless`
- This repo on `PYTHONPATH`

## Quick start

```bash
# 1) Engine (sibling folder)
git clone --depth 1 https://github.com/efroemling/ballistica.git
cd ballistica && make prefab-server-debug-build   # needs uv + clang-format; see docs/NEXT_STEPS.md

# 2) Trainer
cd ../ballistica-walktrain
export BALLISTICA_ROOT=../ballistica
./walktrain/run.sh
```

Checkpoint default: `output/spaz_walk.brain`. Rerun the same command to **resume**.

| Env var | Default | Meaning |
|---------|---------|---------|
| `WALKTRAIN_BRAIN` | `output/spaz_walk.brain` | Checkpoint path |
| `WALKTRAIN_SPEED` | `3` | Wall-clock `2**n` via debug speed exponent |
| `WALKTRAIN_SAVE_EVERY` | `5` | Episodes between saves |
| `BALLISTICA_ROOT` | `../ballistica` | Engine checkout for `run.sh` |

## Package layout

```
walktrain/brain.py      # MLP + Adam + .brain I/O (pure Python)
walktrain/activity.py   # Session/Activity, Spaz, rewards
walktrain/boot.py       # --exec entrypoint
walktrain/run.sh        # Launch helper
```

## License

Trainer code in this repository: MIT (same spirit as Ballistica’s MIT sources).  
Ballistica engine/assets: follow [efroemling/ballistica](https://github.com/efroemling/ballistica) licensing — do not redistribute proprietary assets.
