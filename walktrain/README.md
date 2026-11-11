# WalkTrain — headless Spaz duel trainer

Trains two Spazzes to **fight** on a **flat Football Stadium** (no powerups,
no flags) using **real Ballistica physics**. Each agent observes the other's
**node xyz**; punch damage scales with **momentum** via the stock Spaz
`HitMessage` → node `impulse` path (`punch_momentum_angular` /
`punch_momentum_linear`).

Progress is continuously written to a single **`.brain` checkpoint** (gzipped
JSON weights + Adam state + episode counters + flat-map meta).

## Layout

| File | Role |
|------|------|
| `brain.py` | Pure-Python MLP policy + value baseline, save/load + flat `meta` |
| `activity.py` | `WalkTrainSession` / duel activity (2 Spazzes + Football Stadium) |
| `boot.py` | `--exec` entry: hijacks host session after boot |
| `walktrain_brain.py` | Standalone BombSquad 1.8 / API 9 plugin (drop-in) |
| `install_plugin.sh` | Copy that one plugin file + brain into mods/ |
| `run.sh` | Finds/builds headless binary and starts training |
| `output/spaz_fight.brain` | Default checkpoint (created on first save) |

## Quick start

```bash
export BALLISTICA_ROOT=../ballistica
./walktrain/run.sh   # creates/resumes output/spaz_fight.brain
```

## What the brain learns

- **Obs (12):** self pos/vel, opponent node xyz relative to self, speeds, HP frac.
- **Actions (36):** 8 directions + idle, each with run × punch bits.
- **Reward:** damage dealt − damage taken + close-range + kill/death terminals.
- **Update:** REINFORCE with learned baseline (Adam), both fighters each duel.

Brain `meta` stores flat rules (`flat_map`, `no_powerups`, `no_flags`) so the
plugin can force a bare Football Stadium Death Match and spawn the duel.
