# Released under the MIT License. See LICENSE for details.
"""Drive a Spaz with a loaded walktrain brain (inference only)."""

from __future__ import annotations

import math
from typing import Any

import babase
import bascenev1 as bs

from walktrain.brain import OBS_SIZE, Brain


class BrainPlayer:
    """Spawns a Spaz and ticks a trained brain against the live map."""

    def __init__(
        self,
        activity: bs.Activity,
        brain: Brain,
        *,
        greedy: bool = True,
        character: str = 'Spaz',
        color: tuple[float, float, float] = (0.2, 0.9, 0.4),
    ) -> None:
        self.activity = activity
        self.brain = brain
        self.greedy = greedy
        self.character = character
        self.color = color
        self._spaz: Any | None = None
        self._timer: bs.Timer | None = None
        self._episode_t0 = 0.0
        self._spawn_y = float(brain.meta.pad_center[1])
        self._alive = False

    @property
    def alive(self) -> bool:
        if not self._alive or self._spaz is None:
            return False
        try:
            return bool(self._spaz.is_alive())
        except Exception:
            return False

    def start(self) -> None:
        """Spawn the bot at an FFA point (or map center) and begin ticking."""
        self.stop()
        with self.activity.context:
            pos = self._pick_spawn()
            self._spawn_y = float(pos[1])
            self._spaz = self._make_spaz(pos)
            self._episode_t0 = bs.time()
            self._alive = True
            interval = float(self.brain.hyper.tick_interval)
            self._timer = bs.Timer(interval, self._tick, repeat=True)
            babase.screenmessage(
                f'Brain bot spawned ({self.brain.meta.map_name})',
                color=(0.2, 1.0, 0.4),
            )

    def stop(self) -> None:
        """Remove the bot and cancel its timer."""
        self._timer = None
        self._alive = False
        if self._spaz is not None:
            try:
                with self.activity.context:
                    self._spaz.handlemessage(bs.DieMessage(immediate=True))
            except Exception:
                pass
            self._spaz = None

    def _pick_spawn(self) -> tuple[float, float, float]:
        mmap = getattr(self.activity, 'map', None)
        if mmap is not None:
            pts = getattr(mmap, 'ffa_spawn_points', None) or []
            if pts:
                # Prefer an inland-ish point: average of FFA points, or first.
                if len(pts) >= 3:
                    sx = sum(float(p[0]) for p in pts) / len(pts)
                    sy = sum(float(p[1]) for p in pts) / len(pts)
                    sz = sum(float(p[2]) for p in pts) / len(pts)
                    return (sx, sy, sz)
                p = pts[0]
                return (float(p[0]), float(p[1]), float(p[2]))
        c = self.brain.meta.pad_center
        return (float(c[0]), float(c[1]), float(c[2]))

    def _make_spaz(self, pos: tuple[float, float, float]) -> Any:
        from bascenev1lib.actor.spaz import Spaz

        spaz = Spaz(
            color=self.color,
            highlight=(1.0, 1.0, 1.0),
            character=self.character,
            start_invincible=True,
            can_accept_powerups=False,
        )
        spaz.autoretain()
        spaz.handlemessage(bs.StandMessage(position=pos, angle=0.0))
        return spaz

    def _observe(self) -> list[float] | None:
        assert self._spaz is not None
        node = self._spaz.node
        if not node:
            return None
        meta = self.brain.meta
        cx, cy, cz = meta.pad_center
        pos = node.position
        vel = node.velocity
        dx = (pos[0] - cx) * meta.pos_scale
        dy = (pos[1] - cy) * meta.pos_scale
        dz = (pos[2] - cz) * meta.pos_scale
        vx = vel[0] * meta.vel_scale
        vy = vel[1] * meta.vel_scale
        vz = vel[2] * meta.vel_scale
        height = (pos[1] - self._spawn_y) * meta.pos_scale
        dist = math.sqrt(dx * dx + dz * dz)
        near = 1.0 if (dist / meta.pos_scale) > meta.near_edge_radius else 0.0
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
        assert self._spaz is not None
        mx, my, run = Brain.decode_action(action)
        self._spaz.on_move_left_right(mx)
        self._spaz.on_move_up_down(my)
        self._spaz.on_run(run)

    def _tick(self) -> None:
        if not self.alive:
            self.stop()
            return
        assert self._spaz is not None
        node = self._spaz.node
        if node:
            if node.position[1] < self._spawn_y - self.brain.meta.fall_y_drop:
                self.stop()
                babase.screenmessage('Brain bot fell.', color=(1.0, 0.5, 0.2))
                return
        obs = self._observe()
        if obs is None:
            self.stop()
            return
        if self.greedy:
            action, _, _ = self.brain.act_greedy(obs)
        else:
            action, _, _ = self.brain.act(obs)
        self._apply_action(action)
