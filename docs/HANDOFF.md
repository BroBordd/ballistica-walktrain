# HANDOFF — continue WalkTrain after context loss

**Read this first** if you are a new agent/LLM picking up this project.

## Goal (user intent)

Build a **headless minimal program** that:

1. Uses **Ballistica / BombSquad game physics** (identical math to the real game — not a toy sim).
2. Spawns a **Spaz** on a map (**The Pad**).
3. **Trains** an AI to move around **without falling**.
4. No graphics required — headless binary fires the same 8 ms ODE sim.
5. Continuously saves progress to a **`.brain` checkpoint file** that can resume training (LLM-checkpoint style).

Repo owner: **BroBordd** on GitHub. Project: this repository.

## Current status (as of this handoff)

| Item | Status |
|------|--------|
| Pure-Python MLP brain + gzip JSON checkpoints | **Done** (`walktrain/brain.py`) |
| Session/Activity: Spaz on The Pad + RL tick loop | **Done** (`walktrain/activity.py`) |
| `--exec` boot + `run.sh` (standalone + `BALLISTICA_ROOT`) | **Done** |
| Prefab headless binary | **Done** — `ballistica/build/prefab/full/linux_x86_64_server/debug/dist/ballisticakit_headless` |
| libpython / PYTHONHOME bootstrap in `run.sh` | **Done** (conda shared `libpython3.14`) |
| Movement API | **Done** — use `on_move_left_right` / `on_move_up_down` (not `on_move`) |
| Fall physics actually reachable | **Done** — delete The Pad bumper railing on map load |
| Near-edge obs / edge penalty | **Done** — ThePad has no `is_point_near_edge`; use pad-radius heuristic |
| Center spawn curriculum | **Done** — FFA rim spawns were too harsh for REINFORCE |
| End-to-end episodes + `.brain` save/resume | **Verified** |
| Policy reliably avoids falls | **In progress** — at ~150 eps, overall fall ~20–40%; **rim** starts still ~40–70% fall. REINFORCE is slow; keep training / tune |

**Next agent priority:** continue training on the same brain; if rim fall rate stays flat past ~500–1000 eps, try lower LR, longer episodes inland, or a stronger center-seeking reward; optional eval mode / jump.

## How to run

```bash
cd /path/to/ballistica-walktrain
export BALLISTICA_ROOT=../ballistica   # sibling engine checkout
./walktrain/run.sh
```

Useful env:

| Env | Default | Meaning |
|-----|---------|---------|
| `WALKTRAIN_BRAIN` | `output/spaz_walk.brain` | Checkpoint path (resume = same path) |
| `WALKTRAIN_SPEED` | `3` | Wall-clock `2**n` via debug speed exponent |
| `WALKTRAIN_SAVE_EVERY` | `5` | Episodes between saves |
| `WALKTRAIN_MAX_EP` | `30` | Episode timeout (sim seconds) |
| `LIBPYTHON_DIR` | auto | Dir containing `libpython3.14.so.1.0` |
| `PYTHONHOME` | auto with conda | Must match the shared libpython prefix |

Codespaces note: MS CPython is often static-only. Prefab headless needs shared `libpython3.14.so.1.0` — `run.sh` picks conda’s (`/opt/conda/lib`) and sets matching `PYTHONHOME`.

## Sample log (working)

```text
WalkTrain boot → {..., 'speed_exponent': 5, 'max_episode_seconds': 25.0}
WalkTrain: launching WalkTrainSession
WalkTrain: brain=.../output/spaz_walk.brain episodes=0 steps=0 speed=2^5
ep=1 ret=… survive=25.0s fell=False dist=… drift=… best=25.0s …
ep=2 ret=… survive=4.5s fell=True dist=… drift=… best=25.0s …
Saved brain → .../output/spaz_walk.brain (episodes=…)
```

## Critical pitfalls (already solved — do not rediscover)

1. **`Spaz.on_move(x,y)` is broken** on current engine: sends node message `'move'` → `Invalid node-message type`. Use `on_move_left_right` / `on_move_up_down` like SpazBot.
2. **The Pad bumper railing** prevents walking off; trainer deletes `map.railing` after construct so falls are learnable.
3. **`Map.is_point_near_edge` is a no-op** on ThePad (base returns False). Trainer uses `_NEAR_EDGE_RADIUS` around `_PAD_CENTER`.
4. **FFA spawn points are on the rim** — random policy dies instantly. Trainer spawns near center with expanding jitter.
5. **stdlib `logging` often invisible** on headless stdout — use `print(..., flush=True)` for episode lines.
6. **`bs.new_host_session` wants a class**, not an instance — settings via `SESSION_SETTINGS` before launch.

## Layout

```
ballistica-walktrain/
  README.md
  docs/HANDOFF.md
  docs/ARCHITECTURE.md
  docs/NEXT_STEPS.md
  walktrain/
    brain.py activity.py boot.py run.sh
  output/                   # *.brain checkpoints (gitignored)
```

## Engine facts

- Sim step: **8 ms** — same in headless.
- Fall: `OutOfBoundsMessage` → `DieMessage(FALL)`; also Y-drop backup in trainer.
- Upstream: `https://github.com/efroemling/ballistica`
- Prefab: `make prefab-server-debug-build` (needs `uv`, `clang-format`).

## Success criteria

1. Headless process logs `ep=N ret=… survive=… fell=…` indefinitely.
2. `output/spaz_walk.brain` updates periodically.
3. Kill + relaunch continues `episodes` from checkpoint.
4. Over time, fall rate decreases / survival increases on The Pad (railing removed).
