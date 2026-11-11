# HANDOFF — continue WalkTrain after context loss

**Read this first** if you are a new agent/LLM picking up this project.

## Goal (user intent)

Build a **headless minimal program** that:

1. Uses **Ballistica / BombSquad game physics** (identical math to the real game).
2. Spawns **two Spazzes** on a **flat Football Stadium** (no powerups, no flags).
3. Each Spaz observes the **3D node position** of the other.
4. They **fight and learn** (self-play). More **momentum** → more punch damage
   (stock Spaz `HitMessage` → node `impulse` using `punch_momentum_*`).
5. Continuously saves progress to a **`.brain` checkpoint**.

Repo owner: **BroBordd** on GitHub. Canonical tree: **`ballistica-walktrain/`**.

## Current status

| Item | Status |
|------|--------|
| Pure-Python MLP brain + gzip JSON checkpoints | **Done** (`walktrain/brain.py`, **v2**) |
| Flat Football Stadium duel activity (2 Spazzes) | **Done** (`walktrain/activity.py`) |
| Brain `meta` flat rules (`flat_map`, `no_powerups`, `no_flags`) | **Done** |
| BombSquad plugin reads flat rules + spawns duel | **Done** (`walktrain_brain.py`) |
| Old The Pad walk brain (`spaz_walk.brain`) | **Deleted** — use `spaz_fight.brain` |

**Next agent priority:** run/train `output/spaz_fight.brain` (single or
`run_parallel.sh`); tune damage/close/kill rewards if agents stall; keep
plugin flat powerup patch working on API 9.

## How to run

```bash
cd /path/to/ballistica-walktrain
export BALLISTICA_ROOT=../ballistica
./walktrain/run.sh
```

| Env | Default | Meaning |
|-----|---------|---------|
| `WALKTRAIN_BRAIN` | `output/spaz_fight.brain` | Checkpoint path |
| `WALKTRAIN_SPEED` | `3` | Wall-clock `2**n` |
| `WALKTRAIN_SAVE_EVERY` | `5` | Duels between saves |
| `WALKTRAIN_MAX_EP` | `40` | Duel timeout (sim seconds) |

## Momentum → damage (game API)

Punches send `bs.HitMessage` with magnitudes from
`node.punch_momentum_angular` / `punch_momentum_linear` and `punch_power`.
Targets apply node message `'impulse'`; resulting `node.damage` becomes HP
loss (`damage_scale * node.damage`). Running into punches deals more damage.
Flat damage (`flat_damage`) bypasses that — we don't use it for punches.

## Critical pitfalls

1. Use `on_move_left_right` / `on_move_up_down` (not `on_move`).
2. Brain **v2** obs/actions differ from walk v1 — old brains will not load.
3. Plugin suppresses DeathMatch powerups when `meta.flat_map` / `no_powerups`.
4. `print(..., flush=True)` for headless logs (stdlib logging often silent).
5. `bs.new_host_session` wants a class; settings via `SESSION_SETTINGS`.
