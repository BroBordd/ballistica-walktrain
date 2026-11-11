# Released under the MIT License. See LICENSE for details.
"""Headless Spaz walk-training session using real Ballistica physics."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any, Sequence, override

import babase
import bascenev1 as bs

from walktrain.brain import OBS_SIZE, Brain, HyperParams

if TYPE_CHECKING:
    from bascenev1lib.actor.spaz import Spaz

# Map center-ish for The Pad (from mapdata flag_default / bounds).
_PAD_CENTER = (0.46, 4.38, -3.0)
# Horizontal radius past which we treat the Spaz as near the rim.
# Spawns sit ~7.3 from center; railings sit around that ring.
_NEAR_EDGE_RADIUS = 5.8
# If Y drops this far below spawn, treat as a fall even before OOB.
_FALL_Y_DROP = 3.5
_POS_SCALE = 0.12
_VEL_SCALE = 0.08

_DEFAULT_BRAIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output',
    'spaz_walk.brain',
)

# new_host_session() requires a Session *type*, so boot.py stuffs
# knobs here before launching.
SESSION_SETTINGS: dict[str, Any] = {}


class TraineeSpaz:
    """Thin wrapper: real Spaz + fall callback for the trainer."""

    def __init__(self, spaz: Spaz, on_fell: Any) -> None:
        self.spaz = spaz
        self._on_fell = on_fell
        self._fell = False
        # Wrap handlemessage to catch falls without forking Spaz.
        orig = spaz.handlemessage

        def _wrapped(msg: Any) -> Any:
            if isinstance(msg, bs.OutOfBoundsMessage) and not self._fell:
                self._fell = True
                self._on_fell()
            elif isinstance(msg, bs.DieMessage) and not self._fell:
                how = getattr(msg, 'how', None)
                if how in (bs.DeathType.FALL, bs.DeathType.OUT_OF_BOUNDS):
                    self._fell = True
                    self._on_fell()
            return orig(msg)

        spaz.handlemessage = _wrapped  # type: ignore[method-assign]

    @property
    def fell(self) -> bool:
        return self._fell

    def is_alive(self) -> bool:
        try:
            return bool(self.spaz.is_alive()) and not self._fell
        except Exception:
            return False


class WalkTrainActivity(bs.Activity[bs.Player, bs.Team]):
    """Spawn a Spaz on The Pad; train a brain to walk without falling."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.map: bs.Map | None = None
        self.brain_path: str = settings.get('brain_path', _DEFAULT_BRAIN)
        self.save_every: int = int(settings.get('save_every', 5))
        self.log_every: int = int(settings.get('log_every', 1))
        self.worker_id: int = int(settings.get('worker_id', 0))
        self.shared_brain: bool = bool(settings.get('shared_brain', True))
        self.brain = Brain.load_or_create(
            self.brain_path,
            hyper=HyperParams(
                speed_exponent=int(settings.get('speed_exponent', 3)),
                max_episode_seconds=float(
                    settings.get('max_episode_seconds', 30.0)
                ),
            ),
        )
        # Env/boot settings win over checkpoint hyper for wall-clock knobs.
        self._speed_override = (
            int(settings['speed_exponent'])
            if 'speed_exponent' in settings
            else None
        )
        self._max_ep_override = (
            float(settings['max_episode_seconds'])
            if 'max_episode_seconds' in settings
            else None
        )
        self._apply_runtime_overrides()
        self._trainee: TraineeSpaz | None = None
        self._timer: bs.Timer | None = None
        self._save_timer: bs.Timer | None = None
        self._episode_t0 = 0.0
        self._last_pos: tuple[float, float, float] | None = None
        self._spawn_pos: tuple[float, float, float] | None = None
        self._dist_moved = 0.0
        self._obs: list[list[float]] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._values: list[float] = []
        self._log_probs: list[float] = []
        self._episode_fell = False
        self._spawn_y = _PAD_CENTER[1]
        self._episodes_since_save = 0
        self._started_wall = time.time()
        self._local_episodes = 0

    @override
    def on_transition_in(self) -> None:
        super().on_transition_in()
        from bascenev1lib.maps import ThePad

        ThePad.preload()
        self.map = ThePad()
        # Stock The Pad has bumper railings that bounce a walking Spaz
        # back onto the floor — without removing them, "don't fall" is
        # nearly free and the policy never sees OutOfBoundsMessage.
        railing = getattr(self.map, 'railing', None)
        if railing is not None:
            try:
                railing.delete()
            except Exception:
                pass
            self.map.railing = None  # type: ignore[attr-defined]
        # Quiet camera / lighting defaults; headless ignores draw.
        g = self.globalsnode
        g.tint = (1.1, 1.05, 1.0)
        g.ambient_color = (1.0, 1.0, 1.0)

    @override
    def on_begin(self) -> None:
        super().on_begin()
        hyp = self.brain.hyper
        try:
            bs.set_debug_speed_exponent(hyp.speed_exponent)
        except Exception as exc:
            logging.warning('Could not set debug speed: %s', exc)

        # print() reaches headless stdout; stdlib logging often does not.
        print(
            f'WalkTrain w{self.worker_id}: brain={self.brain_path}'
            f' episodes={self.brain.episodes} steps={self.brain.steps}'
            f' speed=2^{hyp.speed_exponent} shared={self.shared_brain}',
            flush=True,
        )
        self._start_episode()
        self._timer = bs.Timer(hyp.tick_interval, self._tick, repeat=True)
        # Wall-clock periodic save so a kill -9 still keeps recent progress.
        self._save_timer = bs.BaseTimer(
            30.0, bs.WeakCallStrict(self._periodic_save), repeat=True
        )

    def _spawn_point(self) -> tuple[float, float, float]:
        assert self.map is not None
        # Mix safe interior starts with rim-adjacent starts so the
        # policy both explores and must learn not to walk off.
        rng = self.brain.rng
        pts = self.map.ffa_spawn_points
        y = float(pts[0][1]) if pts else _PAD_CENTER[1]

        # Every 3rd episode: real FFA rim spawn (hard).
        if self.brain.episodes % 3 == 2 and pts:
            p = pts[self.brain.episodes % len(pts)]
            return (float(p[0]), float(p[1]), float(p[2]))

        # Otherwise: annulus whose outer radius grows with training.
        # Early: mostly inland. Later: spawn already near the lip.
        outer = min(2.5 + 0.06 * self.brain.episodes, 6.8)
        inner = max(0.0, outer - 2.5)
        ang = rng.random() * 6.283185307179586
        # Uniform in annulus area.
        rad = math.sqrt(
            rng.random() * (outer * outer - inner * inner) + inner * inner
        )
        x = _PAD_CENTER[0] + rad * math.cos(ang)
        z = _PAD_CENTER[2] + rad * math.sin(ang)
        return (x, y, z)

    def _make_spaz(self, pos: Sequence[float]) -> TraineeSpaz:
        from bascenev1lib.actor.spaz import Spaz

        spaz = Spaz(
            color=(0.2, 0.8, 1.0),
            highlight=(1.0, 1.0, 1.0),
            character='Spaz',
            start_invincible=False,
            can_accept_powerups=False,
        )
        spaz.autoretain()
        spaz.handlemessage(bs.StandMessage(position=pos, angle=0.0))

        def _on_fell() -> None:
            self._episode_fell = True

        return TraineeSpaz(spaz, _on_fell)

    def _apply_runtime_overrides(self) -> None:
        if self._speed_override is not None:
            self.brain.hyper.speed_exponent = self._speed_override
        if self._max_ep_override is not None:
            self.brain.hyper.max_episode_seconds = self._max_ep_override

    def _start_episode(self) -> None:
        # Flush local updates before adopting peer weights, otherwise a
        # pull would discard uncommitted REINFORCE steps.
        if self.shared_brain and self.brain._pending_episodes > 0:
            self._save_brain()
            self._episodes_since_save = 0
        if self.shared_brain:
            try:
                if self.brain.pull_shared(self.brain_path):
                    self._apply_runtime_overrides()
            except Exception:
                logging.exception('shared brain pull failed')

        # Drop previous actor.
        if self._trainee is not None:
            try:
                self._trainee.spaz.handlemessage(
                    bs.DieMessage(immediate=True)
                )
            except Exception:
                pass
            self._trainee = None

        pos = self._spawn_point()
        self._spawn_y = pos[1]
        self._spawn_pos = (pos[0], pos[1], pos[2])
        self._trainee = self._make_spaz(pos)
        self._episode_t0 = bs.time()
        self._last_pos = self._spawn_pos
        self._dist_moved = 0.0
        self._obs = []
        self._actions = []
        self._rewards = []
        self._values = []
        self._log_probs = []
        self._episode_fell = False

    def _observe(self) -> list[float] | None:
        assert self._trainee is not None and self.map is not None
        node = self._trainee.spaz.node
        if not node:
            return None
        pos = node.position
        vel = node.velocity
        cx, cy, cz = _PAD_CENTER
        dx = (pos[0] - cx) * _POS_SCALE
        dy = (pos[1] - cy) * _POS_SCALE
        dz = (pos[2] - cz) * _POS_SCALE
        vx, vy, vz = (
            vel[0] * _VEL_SCALE,
            vel[1] * _VEL_SCALE,
            vel[2] * _VEL_SCALE,
        )
        height = (pos[1] - self._spawn_y) * _POS_SCALE
        # ThePad does not override Map.is_point_near_edge (always False).
        # Use a pad-radius heuristic so edge_penalty / obs are meaningful.
        dist = math.sqrt(dx * dx + dz * dz)
        near = 1.0 if (dist / _POS_SCALE) > _NEAR_EDGE_RADIUS else 0.0
        speed = math.sqrt(vx * vx + vz * vz)
        et = bs.time() - self._episode_t0
        obs = [
            dx,
            dy,
            dz,
            vx,
            vy,
            vz,
            height,
            near,
            dist,
            speed,
            math.sin(et),
            math.cos(et),
        ]
        assert len(obs) == OBS_SIZE
        return obs

    def _apply_action(self, action: int) -> None:
        assert self._trainee is not None
        mx, my, run = Brain.decode_action(action)
        spaz = self._trainee.spaz
        # Prefer attribute setters: Spaz.on_move() sends node message
        # 'move', which current SpazNode C++ does not handle (Invalid
        # node-message type). Stock SpazBot still uses these APIs.
        spaz.on_move_left_right(mx)
        spaz.on_move_up_down(my)
        spaz.on_run(run)

    def _tick(self) -> None:
        if self._trainee is None:
            self._start_episode()
            return

        hyp = self.brain.hyper
        elapsed = bs.time() - self._episode_t0
        alive = self._trainee.is_alive() and not self._episode_fell

        # Episode timeout → soft end (survived).
        timed_out = elapsed >= hyp.max_episode_seconds
        if not alive or timed_out:
            self._end_episode(fell=not alive and not timed_out)
            return

        obs = self._observe()
        if obs is None:
            self._end_episode(fell=True)
            return

        action, logp, value = self.brain.act(obs)
        self._apply_action(action)

        node = self._trainee.spaz.node
        pos = node.position if node else self._last_pos
        assert pos is not None
        # Backup fall detection: long drop before map-bounds OOB fires.
        if pos[1] < self._spawn_y - _FALL_Y_DROP:
            self._episode_fell = True
        moved = 0.0
        if self._last_pos is not None:
            moved = math.sqrt(
                (pos[0] - self._last_pos[0]) ** 2
                + (pos[2] - self._last_pos[2]) ** 2
            )
        self._dist_moved += moved
        self._last_pos = (pos[0], pos[1], pos[2])

        near = obs[7]
        reward = hyp.survive_reward + hyp.move_reward * moved
        if near > 0.5:
            # Penalize loitering at the rim (falling risk).
            reward -= hyp.edge_penalty

        self._obs.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._log_probs.append(logp)

        # Fall may arrive between ticks via OutOfBoundsMessage.
        if self._episode_fell or not self._trainee.is_alive():
            self._end_episode(fell=True)

    def _end_episode(self, *, fell: bool) -> None:
        hyp = self.brain.hyper
        survival = bs.time() - self._episode_t0
        if self._rewards:
            if fell:
                self._rewards[-1] -= hyp.fall_penalty
            else:
                # Bonus for lasting the full window without falling.
                self._rewards[-1] += hyp.fall_penalty * 0.5

        ep_return = self.brain.learn_episode(
            self._obs,
            self._actions,
            self._rewards,
            self._values,
            self._log_probs,
        )
        if survival > self.brain.best_survival:
            self.brain.best_survival = survival

        self._episodes_since_save += 1
        self._local_episodes += 1
        if self.brain.episodes % self.log_every == 0:
            wall = time.time() - self._started_wall
            end = self._last_pos or self._spawn_pos or (0.0, 0.0, 0.0)
            spawn = self._spawn_pos or end
            drift = math.sqrt(
                (end[0] - spawn[0]) ** 2 + (end[2] - spawn[2]) ** 2
            )
            print(
                f'w{self.worker_id} ep={self.brain.episodes}'
                f' local={self._local_episodes} ret={ep_return:.2f}'
                f' survive={survival:.1f}s fell={fell}'
                f' dist={self._dist_moved:.1f} drift={drift:.1f}'
                f' best={self.brain.best_survival:.1f}s'
                f' steps={self.brain.steps} wall={wall:.0f}s',
                flush=True,
            )

        if self._episodes_since_save >= self.save_every:
            self._save_brain()
            self._episodes_since_save = 0

        self._start_episode()

    def _save_brain(self) -> None:
        try:
            if self.shared_brain:
                self.brain.save_shared(self.brain_path)
            else:
                self.brain.save(self.brain_path)
            print(
                f'w{self.worker_id} Saved brain → {self.brain_path}'
                f' (episodes={self.brain.episodes})',
                flush=True,
            )
        except Exception:
            logging.exception('Failed to save brain')

    def _periodic_save(self) -> None:
        if self._episodes_since_save > 0:
            self._save_brain()
            self._episodes_since_save = 0


class WalkTrainSession(bs.Session):
    """Host session that only runs WalkTrainActivity."""

    def __init__(self) -> None:
        super().__init__()
        settings = dict(SESSION_SETTINGS)
        self.setactivity(bs.newactivity(WalkTrainActivity, settings))

    @override
    def on_player_request(self, player: bs.SessionPlayer) -> bool:
        return False
