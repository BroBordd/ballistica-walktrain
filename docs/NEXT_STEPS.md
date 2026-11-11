# Next steps (ordered)

Work top-down. Check off as you go.

## Done

- [x] Clone Ballistica + fetch `prefab-server-debug-build` headless binary
- [x] Adapt `run.sh` for standalone repo + `BALLISTICA_ROOT` + libpython/PYTHONHOME
- [x] Fix movement (`on_move_left_right` / `on_move_up_down`)
- [x] Remove The Pad bumper railing so falls are possible
- [x] Pad-radius near-edge obs (ThePad has no `is_point_near_edge`)
- [x] Center spawn + expanding jitter (FFA rim spawns too harsh)
- [x] Verify episode logs + `.brain` save/resume
- [x] Capture sample log lines in `docs/HANDOFF.md`

## 1. Keep training / measure learning

```bash
cd ballistica-walktrain
export BALLISTICA_ROOT=../ballistica
export WALKTRAIN_SPEED=5
export WALKTRAIN_MAX_EP=25
export WALKTRAIN_SAVE_EVERY=5
./walktrain/run.sh
```

After ≥100–200 episodes, plot or summarize:

- fall rate in rolling windows of 20
- mean survival time
- `best_survival` in the brain file

Expect early fall rate high but not 100%; survival should trend up if REINFORCE is learning.

## 2. Likely tuning knobs (`walktrain/brain.py` `HyperParams`)

| Symptom | Try |
|---------|-----|
| Always charges off edge | ↑ `edge_penalty`, ↓ `move_reward`, ↑ `survive_reward` |
| Never explores / stands still | ↑ `move_reward`, ↑ `entropy_bonus` |
| Unstable late collapse | ↓ `learning_rate`, ↑ episode length slowly |
| Falls only after long walks | already good — lengthen `max_episode_seconds` |

**Important:** hyperparams are stored inside the `.brain`. Changing defaults in code only affects *new* brains unless you override after `load_or_create` (speed / max_ep already overridden from env).

## 3. Optional improvements

- [x] Eval / playback: greedy argmax via `Brain.act_greedy` + BombSquad plugin
- [ ] Add jump bit to action space (bump `BRAIN_VERSION`) — railings-off makes jump optional
- [x] `ba_meta` Plugin export (`walktrain.plugin.WalkTrainBrainPlugin`, API 9)
- [x] Brain `meta` block (map name, obs scales) for map-checked playback
- [ ] Unit tests for near-edge heuristic / action decode (brain save/load already smoke-tested)
- [x] Shared-file locking + multi-worker Hogwild (`run_parallel.sh`, `save_shared`/`pull_shared`)
- [ ] Sync `ballistica/walktrain/` copy or delete it — canonical tree is `ballistica-walktrain/`

## 4. Out of scope unless user asks

- Redistributing Ballistica proprietary assets into this GitHub repo
- Torch/GPU trainers
- Non–The-Pad maps (swap map class in `on_transition_in`; re-tune center/radius)
