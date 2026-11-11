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
_POS_SCALE = 0.12
_VEL_SCALE = 0.08

_DEFAULT_BRAIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'output', 'spaz_walk.brain'
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


class WalkTrainActivity(bs.Activity['bs.Player', 'bs.Team']):
    """Spawn a Spaz on The Pad; train a brain to walk without falling."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.map: bs.Map | None = None
        self.brain_path: str = settings.get('brain_path', _DEFAULT_BRAIN)
        self.save_every: int = int(settings.get('save_every', 5))
        self.log_every: int = int(settings.get('log_every', 1))
        self.brain = Brain.load_or_create(
            self.brain_path,
            hyper=HyperParams(
                speed_exponent=int(settings.get('speed_exponent', 3)),
                max_episode_seconds=float(
                    settings.get('max_episode_seconds', 30.0)
                ),
            ),
        )
        self._trainee: TraineeSpaz | None = None
        self._timer: bs.Timer | None = None
        self._save_timer: bs.Timer | None = None
        self._episode_t0 = 0.0
        self._last_pos: tuple[float, float, float] | None = None
        self._obs: list[list[float]] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._values: list[float] = []
        self._log_probs: list[float] = []
        self._episode_fell = False
        self._spawn_y = _PAD_CENTER[1]
        self._episodes_since_save = 0
        self._started_wall = time.time()

    @override
    def on_transition_in(self) -> None:
        super().on_transition_in()
        from bascenev1lib.maps import ThePad

        ThePad.preload()
        self.map = ThePad()
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

        logging.info(
            'WalkTrain: brain=%s episodes=%d steps=%d speed=2^%d',
            self.brain_path,
            self.brain.episodes,
            self.brain.steps,
            hyp.speed_exponent,
        )
        self._start_episode()
        self._timer = bs.Timer(hyp.tick_interval, self._tick, repeat=True)
        # Wall-clock periodic save so a kill -9 still keeps recent progress.
        self._save_timer = bs.BaseTimer(
            30.0, bs.WeakCallStrict(self._periodic_save), repeat=True
        )

    def _spawn_point(self) -> tuple[float, float, float]:
        assert self.map is not None
        pts = self.map.ffa_spawn_points
        # Cycle spawns so the policy sees different edges.
        idx = self.brain.episodes % max(len(pts), 1)
        p = pts[idx]
        return (float(p[0]), float(p[1]), float(p[2]))

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

    def _start_episode(self) -> None:
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
        self._trainee = self._make_spaz(pos)
        self._episode_t0 = bs.time()
        self._last_pos = (pos[0], pos[1], pos[2])
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
        near = (
            1.0
            if self.map.is_point_near_edge(bs.Vec3(pos[0], pos[1], pos[2]), True)
            else 0.0
        )
        dist = math.sqrt(dx * dx + dz * dz)
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
        spaz.on_move(mx, my)
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
        moved = 0.0
        if self._last_pos is not None:
            moved = math.sqrt(
                (pos[0] - self._last_pos[0]) ** 2
                + (pos[2] - self._last_pos[2]) ** 2
            )
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
        if self.brain.episodes % self.log_every == 0:
            wall = time.time() - self._started_wall
            logging.info(
                'ep=%d ret=%.2f survive=%.1fs fell=%s best=%.1fs'
                ' steps=%d wall=%.0fs',
                self.brain.episodes,
                ep_return,
                survival,
                fell,
                self.brain.best_survival,
                self.brain.steps,
                wall,
            )

        if self._episodes_since_save >= self.save_every:
            self._save_brain()
            self._episodes_since_save = 0

        self._start_episode()

    def _save_brain(self) -> None:
        try:
            self.brain.save(self.brain_path)
            logging.info(
                'Saved brain → %s (episodes=%d)',
                self.brain_path,
                self.brain.episodes,
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
