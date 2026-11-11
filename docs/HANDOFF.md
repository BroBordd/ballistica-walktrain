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

## Current status (as of handoff)

| Item | Status |
|------|--------|
| Pure-Python MLP brain + gzip JSON checkpoints | **Done** (`walktrain/brain.py`) |
| Session/Activity: Spaz on The Pad + RL tick loop | **Done** (`walktrain/activity.py`) |
| `--exec` boot + `run.sh` | **Done** (`walktrain/boot.py`, `walktrain/run.sh`) |
| Prefab headless binary downloaded in original workspace | **Partial** — was at `ballistica/build/prefab/full/linux_x86_64_server/debug/dist/ballisticakit_headless` after `make prefab-server-debug-build` |
| End-to-end training run verified (episodes logging + brain save in-engine) | **NOT DONE** — paused before first live training run |
| Docs / GitHub repo | **This handoff** |

**Next agent priority:** get a headless binary, run `./walktrain/run.sh` (or equivalent), fix any boot/import/map-asset errors until episodes log and `output/spaz_walk.brain` appears, then iterate reward/obs if the agent never learns.

## Layout of this repo

```
ballistica-walktrain/
  README.md                 # User-facing quick start
  docs/HANDOFF.md           # This file — agent continuation
  docs/ARCHITECTURE.md      # Design, APIs, pitfalls
  docs/NEXT_STEPS.md        # Ordered TODO for the next agent
  walktrain/
    __init__.py
    brain.py                # Policy + value net, save/load
    activity.py             # WalkTrainSession / WalkTrainActivity
    boot.py                 # --exec entry
    run.sh                  # Locate/build binary and launch
  output/                   # Default checkpoint dir (gitignored *.brain)
```

## How it is supposed to run

WalkTrain is **not** a reimplementation of physics. It is a Python package that loads **inside** `ballisticakit_headless` and drives a real `Spaz` via:

- `bascenev1lib.actor.spaz.Spaz`
- `bascenev1lib.maps.ThePad`
- `bs.Timer` at `tick_interval` (default 0.05 s sim time)
- `bs.set_debug_speed_exponent(n)` for wall-clock speedup (`2**n`)

Expected launch (from a machine that has this repo **and** a Ballistica checkout/binary):

```bash
# Option A: helper script (expects to live inside or beside a ballistica tree — see NEXT_STEPS)
./walktrain/run.sh

# Option B: manual
export PYTHONPATH=/path/to/ballistica-walktrain:$PYTHONPATH
export WALKTRAIN_BRAIN=/path/to/ballistica-walktrain/output/spaz_walk.brain
/path/to/staged/dist/ballisticakit_headless \
  --exec 'import walktrain.boot; walktrain.boot.main()'
```

Resume training = same `WALKTRAIN_BRAIN` path; `Brain.load_or_create` restores weights, Adam state, episode counters, RNG.

## Engine facts (do not rediscover from scratch)

- Sim step: **8 ms** (`kGameStepMilliseconds`) — same in headless.
- Fall detection: `OutOfBoundsMessage` → `DieMessage(FALL)`; map bounds + optional death materials.
- Spaz controls: `on_move(x,y)`, `on_run(v)`, jump press/release.
- `bs.new_host_session(SessionClass)` needs a **class**, not a factory — settings go through `walktrain.activity.SESSION_SETTINGS` set in `boot.py` before launch.
- Headless build/fetch in Ballistica tree: `make prefab-server-debug-build` (needs `uv`, `clang-format`, network). Binary name: `ballisticakit_headless`.
- Env bootstrap for Ballistica make: install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`), `clang-format` via apt.
- No existing neural bot in upstream Ballistica — stock AI is rule-based `SpazBot`.

Upstream engine clone used during development:

```text
https://github.com/efroemling/ballistica
```

Shallow clone is fine: `git clone --depth 1 https://github.com/efroemling/ballistica.git`

## Security / secrets

- **Never** commit GitHub PATs, tokens, or `.brain` files with secrets.
- A PAT was pasted in chat for the initial push; owner should **revoke/rotate** it after use.
- This repo must not contain Ballistica proprietary asset redistributions beyond what upstream tooling downloads locally.

## Success criteria

1. Headless process runs indefinitely, logging lines like `ep=N ret=… survive=… fell=…`.
2. `output/spaz_walk.brain` updates on disk periodically.
3. Killing and relaunching continues `episodes` from the saved value.
4. Over time, fall rate decreases / survival time increases on The Pad.
