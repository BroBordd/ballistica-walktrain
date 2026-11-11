# WalkTrain — headless Spaz locomotion trainer

Trains a Spaz to walk around **The Pad** without falling, using **real
Ballistica physics** (same 8 ms ODE step as the game). No graphics path is
used; the headless binary just fires the identical sim math as fast as you
ask (`set_debug_speed_exponent`).

Progress is continuously written to a single **`.brain` checkpoint** (gzipped
JSON weights + Adam state + episode counters). Kill the process and relaunch
with the same path to resume — same idea as continuing an LLM run from a
checkpoint.

## Layout

| File | Role |
|------|------|
| `brain.py` | Pure-Python MLP policy + value baseline, save/load + `meta` |
| `activity.py` | `WalkTrainSession` / `WalkTrainActivity` (Spaz + The Pad) |
| `boot.py` | `--exec` entry: hijacks host session after boot |
| `walktrain_brain.py` | Standalone BombSquad 1.8 / API 9 plugin (drop-in) |
| `install_plugin.sh` | Copy that one plugin file + brain into mods/ |
| `run.sh` | Finds/builds headless binary and starts training |
| `output/spaz_walk.brain` | Default checkpoint (created on first save) |

## Quick start

From the **standalone** `ballistica-walktrain` repo (engine as sibling):

```bash
export BALLISTICA_ROOT=../ballistica
./walktrain/run.sh   # creates/resumes output/spaz_walk.brain
```

Custom checkpoint / faster wall-clock:

```bash
WALKTRAIN_BRAIN=/tmp/pad.brain WALKTRAIN_SPEED=5 ./walktrain/run.sh
```

## What the brain learns

- **Obs (12):** relative position/velocity, height, near-edge flag, center
  distance, speed, episode-time sin/cos.
- **Actions (18):** 8 directions + idle, each with/without run.
- **Reward:** survive + move distance − rim loitering − large fall penalty.
- **Update:** REINFORCE with learned baseline (Adam), once per episode.

## Notes

- Physics identity comes from the engine, not a reimplementation. This package
  only supplies the policy loop and checkpointing.
- `WALKTRAIN_SPEED` is a *wall-clock* multiplier (`2**n`); sim timestep stays
  8 ms.
- Saves: every `WALKTRAIN_SAVE_EVERY` episodes (default 5) and every 30 s of
  wall time.
