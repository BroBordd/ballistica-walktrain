# Released under the MIT License. See LICENSE for details.
"""Headless Spaz duel trainer on flat Football Stadium (real physics)."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any, Sequence, override

import babase
import bascenev1 as bs

from walktrain.brain import OBS_SIZE, Brain, BrainMeta, HyperParams

if TYPE_CHECKING:
    from bascenev1lib.actor.spaz import Spaz

# Football Stadium flag_default / midfield — flat ground, no railings.
_STADIUM_CENTER = (-0.1, 0.04, 0.11)
_POS_SCALE = 0.08
_VEL_SCALE = 0.08
_FALL_Y_DROP = 5.0
_DEFAULT_HITPOINTS = 1000

_DEFAULT_BRAIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output',
    'spaz_fight.brain',
)

# new_host_session() requires a Session *type*, so boot.py stuffs
# knobs here before launching.
SESSION_SETTINGS: dict[str, Any] = {}


def _default_meta() -> BrainMeta:
    return BrainMeta(
        map_name='Football Stadium',
        map_type='FootballStadium',
        task='duel_fight',
        trainer='walktrain',
        pad_center=_STADIUM_CENTER,
        near_edge_radius=10.0,
        pos_scale=_POS_SCALE,
        vel_scale=_VEL_SCALE,
        fall_y_drop=_FALL_Y_DROP,
        railing_removed_in_training=False,
        flat_map=True,
        no_powerups=True,
        no_flags=True,
        notes=(
            'Self-play duel on flat Football Stadium (no powerups/flags). '
            'Punch damage scales with momentum via Spaz HitMessage impulse.'
        ),
    )


class TraineeSpaz:
    """Thin wrapper: real Spaz + death callback for the trainer."""

    def __init__(self, spaz: Spaz, on_died: Any) -> None:
        self.spaz = spaz
        self._on_died = on_died
        self._dead = False
        orig = spaz.handlemessage

        def _wrapped(msg: Any) -> Any:
            if self._dead:
                return orig(msg)
            if isinstance(msg, bs.OutOfBoundsMessage):
                self._dead = True
                self._on_died()
            elif isinstance(msg, bs.DieMessage):
                self._dead = True
                self._on_died()
            return orig(msg)

        spaz.handlemessage = _wrapped  # type: ignore[method-assign]

    @property
    def dead(self) -> bool:
        return self._dead

    def is_alive(self) -> bool:
        try:
            if self._dead:
                return False
            if not self.spaz.is_alive():
                return False
            return int(getattr(self.spaz, 'hitpoints', 0) or 0) > 0
        except Exception:
            return False

    def hitpoints(self) -> int:
        try:
            return int(getattr(self.spaz, 'hitpoints', 0) or 0)
        except Exception:
            return 0


class FighterRoll:
    """Per-agent episode buffers for one side of a duel."""

    def __init__(self) -> None:
        self.trainee: TraineeSpaz | None = None
        self.spawn_pos: tuple[float, float, float] | None = None
        self.spawn_y = _STADIUM_CENTER[1]
        self.last_hp = float(_DEFAULT_HITPOINTS)
        self.damage_dealt = 0.0
        self.obs: list[list[float]] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.died = False


class WalkTrainActivity(bs.Activity[bs.Player, bs.Team]):
    """Spawn two Spazzes on Football Stadium; self-play fight + learn."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.map: bs.Map | None = None
        self.brain_path: str = settings.get('brain_path', _DEFAULT_BRAIN)
        self.save_every: int = int(settings.get('save_every', 5))
        self.log_every: int = int(settings.get('log_every', 1))
        self.worker_id: int = int(settings.get('worker_id', 0))
        self.shared_brain: bool = bool(settings.get('shared_brain', True))
        meta = _default_meta()
        self.brain = Brain.load_or_create(
            self.brain_path,
            hyper=HyperParams(
                speed_exponent=int(settings.get('speed_exponent', 3)),
                max_episode_seconds=float(
                    settings.get('max_episode_seconds', 40.0)
                ),
            ),
            meta=meta,
        )
        # Always stamp flat-stadium duel rules into the checkpoint.
        self.brain.meta = _default_meta()
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
        self._fighters: list[FighterRoll] = [FighterRoll(), FighterRoll()]
        self._timer: bs.Timer | None = None
        self._save_timer: bs.Timer | None = None
        self._episode_t0 = 0.0
        self._episodes_since_save = 0
        self._started_wall = time.time()
        self._local_episodes = 0

    @override
    def on_transition_in(self) -> None:
        super().on_transition_in()
        from bascenev1lib.maps import FootballStadium

        FootballStadium.preload()
        self.map = FootballStadium()
        # Flat stadium: no powerups, no flags, no TNT — just the floor.
        # (We never call setup_standard_powerup_drops; map props stay unused.)
        g = self.globalsnode
        g.tint = (1.3, 1.2, 1.0)
        g.ambient_color = (1.3, 1.2, 1.0)

    @override
    def on_begin(self) -> None:
        super().on_begin()
        hyp = self.brain.hyper
        try:
            bs.set_debug_speed_exponent(hyp.speed_exponent)
        except Exception as exc:
            logging.warning('Could not set debug speed: %s', exc)

        print(
            f'WalkTrain duel w{self.worker_id}: brain={self.brain_path}'
            f' map={self.brain.meta.map_name}'
            f' flat={self.brain.meta.flat_map}'
            f' no_powerups={self.brain.meta.no_powerups}'
            f' episodes={self.brain.episodes} steps={self.brain.steps}'
            f' speed=2^{hyp.speed_exponent} shared={self.shared_brain}',
            flush=True,
        )
        self._start_episode()
        self._timer = bs.Timer(hyp.tick_interval, self._tick, repeat=True)
        self._save_timer = bs.BaseTimer(
            30.0, bs.WeakCallStrict(self._periodic_save), repeat=True
        )

    def _apply_runtime_overrides(self) -> None:
        if self._speed_override is not None:
            self.brain.hyper.speed_exponent = self._speed_override
        if self._max_ep_override is not None:
            self.brain.hyper.max_episode_seconds = self._max_ep_override

    def _side_spawn(self, side: int) -> tuple[float, float, float]:
        assert self.map is not None
        # Opposite halves of the field (still "one side" each), but pulled
        # inward from stock team spawns so early policies can meet & punch.
        # Stock spawn1/spawn2 sit ~±10 from mid; 0.55 keeps clear sides.
        try:
            p = self.map.get_start_position(side)
            cx, cy, cz = _STADIUM_CENTER
            x = cx + 0.55 * (float(p[0]) - cx)
            z = cz + 0.55 * (float(p[2]) - cz)
            return (x, float(p[1]), z)
        except Exception:
            y = _STADIUM_CENTER[1]
            x = -5.5 if side == 0 else 5.5
            return (x, y, 0.0)

    def _make_spaz(
        self,
        pos: Sequence[float],
        *,
        color: tuple[float, float, float],
        on_died: Any,
    ) -> TraineeSpaz:
        from bascenev1lib.actor.spaz import Spaz

        spaz = Spaz(
            color=color,
            highlight=(1.0, 1.0, 1.0),
            character='Spaz',
            start_invincible=False,
            can_accept_powerups=False,
        )
        spaz.bomb_count = 0
        spaz.autoretain()
        angle = 0.0 if pos[0] < 0 else 180.0
        spaz.handlemessage(bs.StandMessage(position=pos, angle=angle))
        return TraineeSpaz(spaz, on_died)

    def _drop_fighter(self, roll: FighterRoll) -> None:
        if roll.trainee is not None:
            try:
                roll.trainee.spaz.handlemessage(bs.DieMessage(immediate=True))
            except Exception:
                pass
            roll.trainee = None

    def _start_episode(self) -> None:
        if self.shared_brain and self.brain._pending_episodes > 0:
            self._save_brain()
            self._episodes_since_save = 0
        if self.shared_brain:
            try:
                if self.brain.pull_shared(self.brain_path):
                    self._apply_runtime_overrides()
                    self.brain.meta = _default_meta()
            except Exception:
                logging.exception('shared brain pull failed')

        colors = ((0.2, 0.7, 1.0), (1.0, 0.35, 0.2))
        for side, roll in enumerate(self._fighters):
            self._drop_fighter(roll)
            pos = self._side_spawn(side)
            roll.spawn_pos = pos
            roll.spawn_y = pos[1]
            roll.last_hp = float(_DEFAULT_HITPOINTS)
            roll.damage_dealt = 0.0
            roll.obs = []
            roll.actions = []
            roll.rewards = []
            roll.values = []
            roll.log_probs = []
            roll.died = False

            def _on_died(r: FighterRoll = roll) -> None:
                r.died = True

            roll.trainee = self._make_spaz(
                pos, color=colors[side], on_died=_on_died
            )
            roll.last_hp = float(roll.trainee.hitpoints() or _DEFAULT_HITPOINTS)

        self._episode_t0 = bs.time()

    def _observe(
        self, me: FighterRoll, foe: FighterRoll
    ) -> list[float] | None:
        assert me.trainee is not None
        node = me.trainee.spaz.node
        if not node:
            return None
        foe_node = None
        if foe.trainee is not None:
            foe_node = foe.trainee.spaz.node
        if not foe_node:
            return None

        meta = self.brain.meta
        cx, cy, cz = meta.pad_center
        pos = node.position
        vel = node.velocity
        fpos = foe_node.position
        fvel = foe_node.velocity

        dx = (pos[0] - cx) * meta.pos_scale
        dy = (pos[1] - cy) * meta.pos_scale
        dz = (pos[2] - cz) * meta.pos_scale
        vx = vel[0] * meta.vel_scale
        vy = vel[1] * meta.vel_scale
        vz = vel[2] * meta.vel_scale
        ox = (fpos[0] - pos[0]) * meta.pos_scale
        oy = (fpos[1] - pos[1]) * meta.pos_scale
        oz = (fpos[2] - pos[2]) * meta.pos_scale
        speed = math.sqrt(vx * vx + vz * vz)
        foe_speed = math.sqrt(
            (fvel[0] * meta.vel_scale) ** 2 + (fvel[2] * meta.vel_scale) ** 2
        )
        hp_frac = max(0.0, min(1.0, me.trainee.hitpoints() / float(_DEFAULT_HITPOINTS)))
        obs = [dx, dy, dz, vx, vy, vz, ox, oy, oz, speed, foe_speed, hp_frac]
        assert len(obs) == OBS_SIZE
        return obs

    def _apply_action(self, roll: FighterRoll, action: int) -> None:
        assert roll.trainee is not None
        mx, my, run, punch = Brain.decode_action(action)
        spaz = roll.trainee.spaz
        spaz.on_move_left_right(mx)
        spaz.on_move_up_down(my)
        spaz.on_run(run)
        if punch:
            spaz.on_punch_press()
            spaz.on_punch_release()

    def _tick(self) -> None:
        if any(f.trainee is None for f in self._fighters):
            self._start_episode()
            return

        hyp = self.brain.hyper
        elapsed = bs.time() - self._episode_t0
        timed_out = elapsed >= hyp.max_episode_seconds

        a, b = self._fighters
        alive_a = a.trainee is not None and a.trainee.is_alive() and not a.died
        alive_b = b.trainee is not None and b.trainee.is_alive() and not b.died

        # Backup fall / below-floor death on the flat pitch.
        for roll in self._fighters:
            if roll.trainee is None:
                continue
            node = roll.trainee.spaz.node
            if node and node.position[1] < roll.spawn_y - _FALL_Y_DROP:
                roll.died = True

        alive_a = a.trainee is not None and a.trainee.is_alive() and not a.died
        alive_b = b.trainee is not None and b.trainee.is_alive() and not b.died

        if timed_out or not alive_a or not alive_b:
            self._end_episode(alive_a=alive_a, alive_b=alive_b, timed_out=timed_out)
            return

        # Snapshot HP before actions so damage this tick is attributable.
        hp_before = [float(f.trainee.hitpoints()) for f in self._fighters]

        for i, roll in enumerate(self._fighters):
            foe = self._fighters[1 - i]
            obs = self._observe(roll, foe)
            if obs is None:
                roll.died = True
                continue
            action, logp, value = self.brain.act(obs)
            self._apply_action(roll, action)
            roll.obs.append(obs)
            roll.actions.append(action)
            roll.values.append(value)
            roll.log_probs.append(logp)
            # Placeholder; filled after both act and HP deltas resolve.
            roll.rewards.append(0.0)

        if any(f.died or (f.trainee is not None and not f.trainee.is_alive()) for f in self._fighters):
            # Finalize rewards then end.
            self._fill_tick_rewards(hp_before)
            alive_a = a.trainee is not None and a.trainee.is_alive() and not a.died
            alive_b = b.trainee is not None and b.trainee.is_alive() and not b.died
            self._end_episode(alive_a=alive_a, alive_b=alive_b, timed_out=False)
            return

        self._fill_tick_rewards(hp_before)

    def _fill_tick_rewards(self, hp_before: list[float]) -> None:
        hyp = self.brain.hyper
        hp_now = []
        for roll in self._fighters:
            if roll.trainee is None:
                hp_now.append(0.0)
            else:
                hp_now.append(float(roll.trainee.hitpoints()))

        for i, roll in enumerate(self._fighters):
            if not roll.rewards:
                continue
            foe_i = 1 - i
            lost = max(0.0, hp_before[i] - hp_now[i])
            dealt = max(0.0, hp_before[foe_i] - hp_now[foe_i])
            roll.damage_dealt += dealt
            roll.last_hp = hp_now[i]

            reward = hyp.alive_reward + hyp.damage_reward * dealt
            reward -= hyp.hit_penalty * lost

            # Encourage closing to punch range (momentum punches need contact).
            if roll.obs:
                ox, oy, oz = roll.obs[-1][6], roll.obs[-1][7], roll.obs[-1][8]
                dist = math.sqrt(ox * ox + oy * oy + oz * oz) / max(
                    self.brain.meta.pos_scale, 1e-6
                )
                if dist <= hyp.close_range:
                    reward += hyp.close_reward
                else:
                    # Mild pull toward the opponent.
                    reward += hyp.close_reward * 0.15 * max(0.0, 1.0 - dist / 20.0)

            roll.rewards[-1] = reward

    def _end_episode(
        self, *, alive_a: bool, alive_b: bool, timed_out: bool
    ) -> None:
        hyp = self.brain.hyper
        duration = bs.time() - self._episode_t0

        # Terminal bonuses: winner / loser (draws on timeout get nothing extra).
        if not timed_out:
            if alive_a and not alive_b:
                if self._fighters[0].rewards:
                    self._fighters[0].rewards[-1] += hyp.kill_reward
                if self._fighters[1].rewards:
                    self._fighters[1].rewards[-1] -= hyp.death_penalty
            elif alive_b and not alive_a:
                if self._fighters[1].rewards:
                    self._fighters[1].rewards[-1] += hyp.kill_reward
                if self._fighters[0].rewards:
                    self._fighters[0].rewards[-1] -= hyp.death_penalty

        returns = []
        for roll in self._fighters:
            ep_return = self.brain.learn_episode(
                roll.obs,
                roll.actions,
                roll.rewards,
                roll.values,
                roll.log_probs,
            )
            returns.append(ep_return)

        best = max(returns) if returns else 0.0
        if best > self.brain.best_survival:
            self.brain.best_survival = best

        self._episodes_since_save += 1
        self._local_episodes += 1
        # learn_episode increments episodes per fighter; treat a duel as one
        # logical episode in logs (brain.episodes counts fighter-updates).
        if self._local_episodes % self.log_every == 0:
            wall = time.time() - self._started_wall
            result = (
                'timeout'
                if timed_out
                else (
                    'A'
                    if alive_a and not alive_b
                    else 'B' if alive_b and not alive_a else 'draw'
                )
            )
            print(
                f'w{self.worker_id} duel={self._local_episodes}'
                f' brain_eps={self.brain.episodes}'
                f' retA={returns[0]:.2f} retB={returns[1]:.2f}'
                f' dmgA={self._fighters[0].damage_dealt:.0f}'
                f' dmgB={self._fighters[1].damage_dealt:.0f}'
                f' t={duration:.1f}s winner={result}'
                f' best={self.brain.best_survival:.1f}'
                f' steps={self.brain.steps} wall={wall:.0f}s',
                flush=True,
            )

        if self._episodes_since_save >= self.save_every:
            self._save_brain()
            self._episodes_since_save = 0

        self._start_episode()

    def _save_brain(self) -> None:
        try:
            self.brain.meta = _default_meta()
            if self.shared_brain:
                self.brain.save_shared(self.brain_path)
            else:
                self.brain.save(self.brain_path)
            print(
                f'w{self.worker_id} Saved brain → {self.brain_path}'
                f' (episodes={self.brain.episodes}'
                f' map={self.brain.meta.map_name}'
                f' flat={self.brain.meta.flat_map})',
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
