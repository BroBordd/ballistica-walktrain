# Next steps (ordered)

Work top-down. Check off as you go.

## 1. Workspace setup

1. Clone this repo.
2. Clone Ballistica engine (sibling directory recommended):

   ```bash
   git clone --depth 1 https://github.com/efroemling/ballistica.git
   ```

3. In the ballistica tree, bootstrap and fetch headless prefab:

   ```bash
   # needs: curl, make, cmake-ish toolchain optional for prefab
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # apt: clang-format (checkenv requires it)
   sudo apt-get install -y clang-format
   export PATH="$HOME/.local/bin:$PATH"
   cd ballistica && make prefab-server-debug-build
   ```

4. Confirm binary exists, e.g.:

   `ballistica/build/prefab/full/linux_x86_64_server/debug/dist/ballisticakit_headless`

## 2. First training smoke test

From **this** repo root:

```bash
mkdir -p output
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WALKTRAIN_BRAIN="$PWD/output/spaz_walk.brain"
export WALKTRAIN_SPEED=3
export WALKTRAIN_SAVE_EVERY=2

BIN=../ballistica/build/prefab/full/linux_x86_64_server/debug/dist/ballisticakit_headless
# adjust BIN if layout differs
STAGED="$(cd "$(dirname "$BIN")/.." && pwd)"  # often …/debug or …/staged
cd "$(dirname "$BIN")/.."   # directory that contains dist/ + ba_data if staged that way
# Prefer running with cwd = staged root that contains ba_data:
# Inspect: ls "$(dirname "$BIN")/../ba_data" or "$(dirname "$BIN")/ba_data"

"$BIN" --exec 'import walktrain.boot; walktrain.boot.main()'
```

**Also fix `walktrain/run.sh`** so it works when the package lives in this standalone repo (today it assumes `walktrain/` is inside the ballistica tree with `ROOT=parent`). Update path discovery:

- `REPO_ROOT` = parent of `walktrain/`
- `BALLISTICA_ROOT` = env `BALLISTICA_ROOT` or `../ballistica`
- Put `REPO_ROOT` on `PYTHONPATH`
- Find binary under `$BALLISTICA_ROOT/build/prefab/...`

## 3. Likely failures and fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: walktrain` | PYTHONPATH | Export repo root on PYTHONPATH before `--exec` |
| Missing mesh / asset / construct errors loading The Pad | headless-minimal assets incomplete for maps | May need fuller server asset stage (`assets-server` / non-minimal profile) or GUI-less but full scene assets — inspect Ballistica staging docs |
| Session never starts | `apptimer(2.5)` too early/late; classic mode not up | Increase delay; ensure classic app mode active on headless server |
| `new_host_session` type error | Passed instance/lambda | Must pass `WalkTrainSession` class; settings via `SESSION_SETTINGS` |
| Spaz falls through floor | Collision mesh missing in headless bundle | Same as asset issue — verify collision meshes resolve |
| No brain file | Crash before save; wrong cwd permissions | Check logs; ensure `output/` writable; `save_every` |

## 4. After smoke test works

1. Run ≥50 episodes; confirm `episodes` in checkpoint increments across restarts.
2. Tune `HyperParams` if agent always charges off the edge (increase `edge_penalty`, reduce `move_reward`, add jump later).
3. Optional: multiple parallel headless processes with different brain paths or shared file locking (not implemented).
4. Optional: eval mode (greedy argmax, no learn) for measuring survival.

## 5. Code polish left undone

- [ ] Adapt `run.sh` for standalone repo + `BALLISTICA_ROOT`
- [ ] Verify The Pad collision under current `headless-minimal` prefab
- [ ] Add `docs` note once first successful log line is captured (paste sample)
- [ ] Consider `ba_meta` Plugin export for enabling without `--exec`
- [ ] Unit tests beyond brain save/load (brain already smoke-tested outside engine)

## 6. Out of scope unless user asks

- Pushing proprietary Ballistica assets into this GitHub repo
- Torch/GPU trainers
- Non–The-Pad maps (easy later: swap map class in `on_transition_in`)
