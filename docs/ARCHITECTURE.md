# Architecture

## High-level

```
ballisticakit_headless  (--exec walktrain.boot.main)
        │
        ▼
  WalkTrainSession  (bascenev1.Session, rejects players)
        │
        ▼
  WalkTrainActivity
        ├─ ThePad map (real collision meshes; bumper railing deleted)
        ├─ Spaz trainee (real node + ODE bodies)
        ├─ bs.Timer(tick) → observe → Brain.act → move L/R + U/D + run
        ├─ episode end (fall or timeout) → Brain.learn_episode → maybe save
        └─ BaseTimer(30s wall) → periodic checkpoint
```

Physics identity: **engine ODE**, not approximated in Python. WalkTrain only supplies policy + rewards + I/O.

### The Pad specifics

- Stock map installs a **bumper railing** that bounce-walks characters back onto the floor. WalkTrain deletes `map.railing` so walking off edges yields real `OutOfBoundsMessage` / `DieMessage(FALL)`.
- Base `Map.is_point_near_edge` returns False for ThePad; obs uses horizontal distance from `_PAD_CENTER` vs `_NEAR_EDGE_RADIUS`.
- Prefer center + expanding jitter spawns over FFA rim points (rim starts make early REINFORCE all-fall).
- Controls: `on_move_left_right` / `on_move_up_down` — **not** `Spaz.on_move()` (invalid node message on current engine).

## Brain (`walktrain/brain.py`)

- Pure Python (no numpy/torch) so it runs in Ballistica’s embedded interpreter.
- MLP: obs(12) → hidden(64) → policy(18) + value(1).
- Softmax policy; REINFORCE with learned baseline; Adam.
- Checkpoint: **gzipped JSON** (`.brain`), atomic write via `*.tmp.PID` then `os.replace`.
- Contents: weights, Adam moments, hyperparams, **`meta`** (map/task/obs scales), episodes/steps, best survival, RNG state, timestamps.
- `BRAIN_VERSION = 1` — bump if obs/action layout changes; loaders reject mismatches.
- `meta` is optional on load (defaults to The Pad walk-train); always written on save.

### Playback plugin (`walktrain/walktrain_brain.py`)

Standalone BombSquad **1.8 / API 9** plugin (`# ba_meta require api 9`) — one file, no package deps. Polls `bs.get_chat_messages` for `/brainbot`, then starts a teams Death Match forced to `meta.map_name`, posts a command guide via `bs.chatmessage`, and ticks an inlined `BrainPlayer` (greedy by default). Commands: `/brainbot`, `/brainbot stop`, `/brainbot info`. Install via `install_plugin.sh` → `mods/walktrain_brain.py`.

### Observation (12)

| idx | meaning |
|-----|---------|
| 0–2 | position relative to pad center, scaled |
| 3–5 | velocity, scaled |
| 6 | height above spawn y |
| 7 | near_edge (0/1) via pad-radius heuristic (`_NEAR_EDGE_RADIUS`) |
| 8 | horizontal dist from center |
| 9 | horizontal speed |
| 10–11 | sin/cos episode time |

### Actions (18)

8 compass dirs + idle, each × run bit → `on_move_left_right` / `on_move_up_down` + `on_run`.

### Rewards (defaults in `HyperParams`)

- `+survive_reward` per tick
- `+move_reward * delta_xz`
- `-edge_penalty` when near edge
- `-fall_penalty` on last step if fell
- `+0.5 * fall_penalty` bonus if survived full episode window

### Spawning

Center of The Pad + polar jitter. Radius grows with `episodes` (curriculum) so early runs are safer and later runs start closer to the rim.

## Activity (`walktrain/activity.py`)

- `TraineeSpaz` wraps `Spaz.handlemessage` to detect `OutOfBoundsMessage` / fall deaths.
- Respawns via new `Spaz` + `StandMessage` at rotating `ffa_spawn_points`.
- `SESSION_SETTINGS` module global: required because `new_host_session` takes a class type only.
- Speed: `bs.set_debug_speed_exponent` — multiplies wall clock, **does not** change 8 ms step math.

## Boot (`walktrain/boot.py`)

1. Ensure repo root on `sys.path`.
2. Read env: `WALKTRAIN_BRAIN`, `WALKTRAIN_SPEED`, `WALKTRAIN_SAVE_EVERY`, `WALKTRAIN_MAX_EP`.
3. `babase.apptimer(2.5, …)` then set `SESSION_SETTINGS` and `bs.new_host_session(WalkTrainSession)`.

## Run script (`walktrain/run.sh`)

- Assumes Ballistica project root is parent of `walktrain/` **or** adjust paths.
- In **this GitHub repo**, layout is `ballistica-walktrain/walktrain/…`, so either:
  - symlink/copy `walktrain` into a ballistica checkout, or
  - set `PYTHONPATH` to this repo and invoke the binary with `--exec` manually (preferred for clarity).
- Searches common prefab/cmake staged binary paths; may run `make prefab-server-debug-build`.

## Upstream Ballistica touchpoints (absolute paths in original workspace)

| Topic | Path under `efroemling/ballistica` |
|-------|-----------------------------------|
| Spaz actor | `src/assets/ba_data/python/bascenev1lib/actor/spaz.py` |
| SpazBot (rule AI) | `…/actor/spazbot.py` |
| The Pad | `…/maps.py` + `mapdata/the_pad.py` |
| Benchmark session pattern | `…/baclassic/_benchmark.py` |
| Headless adapter | `src/ballistica/base/app_adapter/app_adapter_headless.*` |
| Prefab/server make targets | top-level `Makefile` (`prefab-server-debug-build`, `cmake-server-build`) |
| App run helper | `tools/batools/apprun.py` |

## Non-goals (for now)

- Multi-agent / punching / bombs
- Training outside the real engine
- GUI visualization of policy
- Torch/CUDA (keep embedded-Python friendly unless user asks)
