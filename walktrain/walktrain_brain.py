# ba_meta require api 9
# Released under the MIT License. See LICENSE for details.
"""Standalone BombSquad 1.8 / API 9 plugin: play a walktrain ``.brain``.

Drop this single file into your mods folder (plus a brain under
``mods/brains/spaz_walk.brain``). Enable under Settings → Plugins, host a
match on the map named in the brain metadata, then chat ``/brainbot``.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence, override

import babase
import bascenev1 as bs

# ---------------------------------------------------------------------------
# Brain inference (load + forward only; no training / shared-lock I/O)
# ---------------------------------------------------------------------------

BRAIN_VERSION = 1
OBS_SIZE = 12

DIRS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (0.0, 1.0),
    (0.707, 0.707),
    (1.0, 0.0),
    (0.707, -0.707),
    (0.0, -1.0),
    (-0.707, -0.707),
    (-1.0, 0.0),
    (-0.707, 0.707),
]
N_DIRS = len(DIRS)
N_ACTIONS = N_DIRS * 2

_CONFIG_BRAIN_PATH = 'WalkTrain Brain Path'
_CONFIG_GREEDY = 'WalkTrain Brain Greedy'


@dataclass
class BrainMeta:
    map_name: str = 'The Pad'
    map_type: str = 'ThePad'
    task: str = 'walk_no_fall'
    trainer: str = 'walktrain'
    pad_center: tuple[float, float, float] = (0.46, 4.38, -3.0)
    near_edge_radius: float = 5.8
    pos_scale: float = 0.12
    vel_scale: float = 0.08
    fall_y_drop: float = 3.5
    railing_removed_in_training: bool = True
    notes: str = (
        'Trained headless on The Pad with bumper railing removed. '
        'Obs/actions are walktrain v1.'
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BrainMeta:
        if not data:
            return cls()
        center = data.get('pad_center', (0.46, 4.38, -3.0))
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            center = (0.46, 4.38, -3.0)
        return cls(
            map_name=str(data.get('map_name', 'The Pad')),
            map_type=str(data.get('map_type', 'ThePad')),
            task=str(data.get('task', 'walk_no_fall')),
            trainer=str(data.get('trainer', 'walktrain')),
            pad_center=(float(center[0]), float(center[1]), float(center[2])),
            near_edge_radius=float(data.get('near_edge_radius', 5.8)),
            pos_scale=float(data.get('pos_scale', 0.12)),
            vel_scale=float(data.get('vel_scale', 0.08)),
            fall_y_drop=float(data.get('fall_y_drop', 3.5)),
            railing_removed_in_training=bool(
                data.get('railing_removed_in_training', True)
            ),
            notes=str(data.get('notes', cls.notes)),
        )

    def matches_map(self, map_name: str) -> bool:
        return self.map_name.strip().casefold() == map_name.strip().casefold()


@dataclass
class HyperParams:
    hidden: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    entropy_bonus: float = 0.02
    fall_penalty: float = 40.0
    survive_reward: float = 0.12
    move_reward: float = 0.04
    edge_penalty: float = 0.25
    max_episode_seconds: float = 30.0
    tick_interval: float = 0.05
    speed_exponent: int = 3


def _matmul_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


def _relu(v: list[float]) -> list[float]:
    return [x if x > 0.0 else 0.0 for x in v]


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


class Brain:
    """MLP policy for playback: obs → action probabilities + value."""

    def __init__(
        self,
        *,
        hyper: HyperParams | None = None,
        meta: BrainMeta | None = None,
        seed: int | None = None,
    ) -> None:
        self.hyper = hyper or HyperParams()
        self.meta = meta or BrainMeta()
        self.episodes = 0
        self.steps = 0
        self.best_survival = 0.0
        self.total_reward = 0.0
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.seed = seed if seed is not None else random.randrange(1 << 31)
        self.rng = random.Random(self.seed)
        self.W1: list[list[float]] = []
        self.b1: list[float] = []
        self.W2: list[list[float]] = []
        self.b2: list[float] = []
        self.Wv: list[list[float]] = []
        self.bv: list[float] = []

    def forward(
        self, obs: Sequence[float]
    ) -> tuple[list[float], float, list[float]]:
        if len(obs) != OBS_SIZE:
            raise ValueError(f'expected obs size {OBS_SIZE}, got {len(obs)}')
        o = list(obs)
        z1 = _add(_matmul_vec(self.W1, o), self.b1)
        h = _relu(z1)
        logits = _add(_matmul_vec(self.W2, h), self.b2)
        probs = _softmax(logits)
        value = _add(_matmul_vec(self.Wv, h), self.bv)[0]
        return probs, value, z1

    def act(self, obs: Sequence[float]) -> tuple[int, float, float]:
        probs, value, _ = self.forward(obs)
        r = self.rng.random()
        cum = 0.0
        action = N_ACTIONS - 1
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                action = i
                break
        p = max(probs[action], 1e-8)
        return action, math.log(p), value

    def act_greedy(self, obs: Sequence[float]) -> tuple[int, float, float]:
        probs, value, _ = self.forward(obs)
        action = max(range(len(probs)), key=lambda i: probs[i])
        p = max(probs[action], 1e-8)
        return action, math.log(p), value

    @staticmethod
    def decode_action(action: int) -> tuple[float, float, float]:
        d = action // 2
        run = float(action % 2)
        mx, my = DIRS[d]
        return mx, my, run

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Brain:
        if data.get('brain_version') != BRAIN_VERSION:
            raise ValueError(
                f"unsupported brain_version {data.get('brain_version')}"
                f' (want {BRAIN_VERSION})'
            )
        if data.get('obs_size') != OBS_SIZE or data.get('n_actions') != N_ACTIONS:
            raise ValueError('brain obs/action size mismatch')
        hyper_raw = dict(data.get('hyper') or {})
        known = {f.name for f in HyperParams.__dataclass_fields__.values()}
        hyper = HyperParams(**{k: v for k, v in hyper_raw.items() if k in known})
        brain = cls(
            hyper=hyper,
            meta=BrainMeta.from_dict(data.get('meta')),
            seed=int(data.get('seed', 0)),
        )
        brain.episodes = int(data['episodes'])
        brain.steps = int(data['steps'])
        brain.best_survival = float(data['best_survival'])
        brain.total_reward = float(data.get('total_reward', 0.0))
        brain.created_at = float(data.get('created_at', time.time()))
        brain.updated_at = float(data.get('updated_at', brain.created_at))
        w = data['weights']
        brain.W1, brain.b1 = w['W1'], w['b1']
        brain.W2, brain.b2 = w['W2'], w['b2']
        brain.Wv, brain.bv = w['Wv'], w['bv']
        state_mt = data.get('rng_state')
        if state_mt is not None:
            brain.rng.setstate((3, tuple(state_mt), None))
        return brain

    @classmethod
    def load(cls, path: str) -> Brain:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# In-game Spaz driven by a loaded brain
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


def _default_brain_path() -> str:
    env = babase.app.env
    user = env.python_directory_user
    if user:
        return os.path.join(user, 'brains', 'spaz_walk.brain')
    here = os.path.dirname(os.path.abspath(__file__))
    # Repo layout: walktrain/walktrain_brain.py → ../output/
    repo = os.path.dirname(here)
    return os.path.join(repo, 'output', 'spaz_walk.brain')


def _resolve_brain_path() -> str:
    cfg = babase.app.config
    path = cfg.get(_CONFIG_BRAIN_PATH)
    if isinstance(path, str) and path.strip():
        return os.path.expanduser(path.strip())
    return _default_brain_path()


def _set_brain_path(path: str) -> None:
    cfg = babase.app.config
    cfg[_CONFIG_BRAIN_PATH] = path
    cfg.commit()


def _greedy_enabled() -> bool:
    return bool(babase.app.config.get(_CONFIG_GREEDY, True))


def _current_map_name() -> str | None:
    activity = bs.get_foreground_host_activity()
    if activity is None:
        return None
    mmap = getattr(activity, 'map', None)
    if mmap is None:
        return None
    try:
        return str(mmap.getname())
    except Exception:
        name = getattr(type(mmap), 'name', None)
        return str(name) if name else None


def _load_brain(path: str) -> Brain:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f'brain file not found: {path}')
    return Brain.load(path)


# ba_meta export babase.Plugin
class WalkTrainBrainPlugin(babase.Plugin):
    """Load a walktrain brain and spawn a playback bot on a matching map."""

    def __init__(self) -> None:
        self._player: BrainPlayer | None = None
        self._orig_filter = None

    @override
    def on_app_running(self) -> None:
        self._install_chat_hook()
        path = _resolve_brain_path()
        babase.screenmessage(
            f'WalkTrain Brain plugin ready. Brain: {os.path.basename(path)}',
            color=(0.6, 0.9, 1.0),
        )
        print(
            'WalkTrain Brain plugin: chat /brainbot to spawn, '
            f'/brainbot stop to remove, brain={path}',
            flush=True,
        )

    @override
    def on_app_shutdown(self) -> None:
        self._despawn()
        self._uninstall_chat_hook()

    @override
    def has_settings_ui(self) -> bool:
        return True

    @override
    def show_settings_ui(self, source_widget: Any | None) -> None:
        _SettingsWindow(source_widget)

    def _install_chat_hook(self) -> None:
        import bascenev1._hooks as hooks

        if self._orig_filter is not None:
            return
        self._orig_filter = hooks.filter_chat_message
        plugin = self

        def _filter(msg: str, client_id: int) -> str | None:
            handled = plugin._handle_chat(msg, client_id)
            if handled:
                return None
            assert plugin._orig_filter is not None
            return plugin._orig_filter(msg, client_id)

        hooks.filter_chat_message = _filter  # type: ignore[assignment]

    def _uninstall_chat_hook(self) -> None:
        if self._orig_filter is None:
            return
        import bascenev1._hooks as hooks

        hooks.filter_chat_message = self._orig_filter  # type: ignore[assignment]
        self._orig_filter = None

    def _handle_chat(self, msg: str, client_id: int) -> bool:
        text = msg.strip()
        if not text.lower().startswith('/brainbot'):
            return False
        if client_id not in (-1,):
            babase.screenmessage(
                'Only the host can spawn the brain bot.', color=(1, 0.5, 0.2)
            )
            return True
        parts = text.split()
        cmd = parts[1].lower() if len(parts) > 1 else 'spawn'
        if cmd in ('stop', 'kill', 'remove'):
            self._despawn()
            babase.screenmessage('Brain bot removed.', color=(1.0, 0.8, 0.3))
            return True
        if cmd in ('info', 'status'):
            self._show_info()
            return True
        if cmd in ('spawn', 'start', 'play'):
            self._spawn()
            return True
        babase.screenmessage(
            'Usage: /brainbot [spawn|stop|info]', color=(1, 1, 0.3)
        )
        return True

    def _show_info(self) -> None:
        path = _resolve_brain_path()
        try:
            brain = _load_brain(path)
        except Exception as exc:
            babase.screenmessage(
                f'Brain load failed: {exc}', color=(1, 0.3, 0.3)
            )
            return
        meta = brain.meta
        cur = _current_map_name() or '(no active map)'
        babase.screenmessage(
            f'Brain map={meta.map_name} task={meta.task} '
            f'eps={brain.episodes} current={cur}',
            color=(0.7, 0.9, 1.0),
        )

    def _despawn(self) -> None:
        if self._player is not None:
            try:
                self._player.stop()
            except Exception:
                pass
            self._player = None

    def _spawn(self) -> None:
        activity = bs.get_foreground_host_activity()
        if activity is None:
            babase.screenmessage(
                'No active host game. Start a match first.',
                color=(1, 0.4, 0.3),
            )
            return

        map_name = _current_map_name()
        if not map_name:
            babase.screenmessage(
                'Current activity has no map yet.', color=(1, 0.4, 0.3)
            )
            return

        path = _resolve_brain_path()
        try:
            brain = _load_brain(path)
        except Exception as exc:
            babase.screenmessage(
                f'Could not load brain: {exc}', color=(1, 0.3, 0.3)
            )
            return

        if not brain.meta.matches_map(map_name):
            babase.screenmessage(
                f'Map mismatch: brain trained on "{brain.meta.map_name}", '
                f'current is "{map_name}".',
                color=(1.0, 0.35, 0.25),
            )
            return

        self._despawn()
        try:
            player = BrainPlayer(activity, brain, greedy=_greedy_enabled())
            player.start()
            self._player = player
        except Exception as exc:
            babase.screenmessage(
                f'Failed to spawn brain bot: {exc}', color=(1, 0.3, 0.3)
            )
            print(f'WalkTrain Brain spawn error: {exc}', flush=True)


class _SettingsWindow:
    """Minimal settings: brain path + status."""

    def __init__(self, origin_widget: Any | None) -> None:
        import bauiv1 as bui

        uiscale = bui.app.ui_v1.uiscale
        width = 520.0
        height = 340.0
        self.root_widget = bui.containerwidget(
            size=(width, height),
            transition='in_scale',
            scale=(
                1.6
                if uiscale is bui.UIScale.SMALL
                else 1.25 if uiscale is bui.UIScale.MEDIUM else 1.0
            ),
        )
        bui.textwidget(
            parent=self.root_widget,
            position=(width * 0.5, height - 40),
            size=(0, 0),
            text='WalkTrain Brain',
            scale=1.1,
            h_align='center',
            v_align='center',
            color=(0.8, 1.0, 0.9),
        )

        path = _resolve_brain_path()
        meta_line = '(brain not loaded)'
        try:
            brain = _load_brain(path)
            meta_line = (
                f'map={brain.meta.map_name}  task={brain.meta.task}  '
                f'episodes={brain.episodes}'
            )
        except Exception as exc:
            meta_line = f'load error: {exc}'

        bui.textwidget(
            parent=self.root_widget,
            position=(30, height - 90),
            size=(width - 60, 40),
            text=f'Path:\n{path}',
            maxwidth=width - 70,
            scale=0.7,
            h_align='left',
            v_align='top',
            color=(0.8, 0.8, 0.8),
        )
        bui.textwidget(
            parent=self.root_widget,
            position=(30, height - 160),
            size=(width - 60, 40),
            text=meta_line,
            maxwidth=width - 70,
            scale=0.75,
            h_align='left',
            v_align='top',
            color=(0.7, 0.95, 0.8),
        )
        bui.textwidget(
            parent=self.root_widget,
            position=(30, height - 210),
            size=(width - 60, 40),
            text='Chat /brainbot while hosting on the matching map.',
            maxwidth=width - 70,
            scale=0.7,
            h_align='left',
            v_align='top',
            color=(0.75, 0.75, 0.6),
        )

        self._path_field = bui.textwidget(
            parent=self.root_widget,
            position=(30, 70),
            size=(width - 200, 40),
            text=path,
            editable=True,
            max_chars=400,
            autoselect=True,
            v_align='center',
            description='Brain file path',
        )

        def _save_path() -> None:
            new_path = bui.textwidget(query=self._path_field).strip()
            if new_path:
                _set_brain_path(new_path)
                bui.screenmessage('Saved brain path', color=(0.4, 1.0, 0.5))

        def _close() -> None:
            bui.containerwidget(edit=self.root_widget, transition='out_scale')

        bui.buttonwidget(
            parent=self.root_widget,
            position=(width - 160, 70),
            size=(130, 40),
            label='Save Path',
            autoselect=True,
            on_activate_call=_save_path,
        )
        close = bui.buttonwidget(
            parent=self.root_widget,
            position=(width * 0.5 - 50, 20),
            size=(100, 35),
            label='Close',
            autoselect=True,
            on_activate_call=_close,
        )
        bui.containerwidget(
            edit=self.root_widget, cancel_button=close, start_button=close
        )
        del origin_widget
