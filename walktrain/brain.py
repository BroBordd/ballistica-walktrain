# Released under the MIT License. See LICENSE for details.
"""Pure-Python MLP policy brain with resumable checkpoints.

No numpy/torch — runs inside Ballistica's embedded interpreter.
Checkpoint format is deliberately LLM-like: one file you can stop,
copy, and resume training from.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence


BRAIN_VERSION = 1

# Observation layout (fixed; bump BRAIN_VERSION if changed):
# 0-2: pos xyz relative to map center, scaled
# 3-5: velocity xyz, scaled
# 6:   height above spawn y, scaled
# 7:   near_edge (0/1)
# 8:   dist_from_center, scaled
# 9:   speed, scaled
# 10:  sin(episode_time)
# 11:  cos(episode_time)
OBS_SIZE = 12

# Actions: 8 compass dirs + idle, each with/without run → 18
# Index: dir*2 + run   where dir in 0..8 (0=idle, 1=N ... 8=NW)
DIRS: list[tuple[float, float]] = [
    (0.0, 0.0),  # idle
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
N_ACTIONS = N_DIRS * 2  # * run bit


@dataclass
class BrainMeta:
    """Where/how this brain was trained — used by playback plugins.

    Old checkpoints without a ``meta`` block default to The Pad walk-train
    values so existing ``.brain`` files keep working.
    """

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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['pad_center'] = [float(x) for x in self.pad_center]
        return data

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
        """True if ``map_name`` is the map this brain was trained for."""
        want = self.map_name.strip().casefold()
        got = map_name.strip().casefold()
        return want == got


@dataclass
class HyperParams:
    """Trainable / architectural knobs stored in the brain file."""

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
    speed_exponent: int = 3  # 2**n realtime (sim still identical)


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _zeros1(n: int) -> list[float]:
    return [0.0] * n


def _xavier(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    scale = math.sqrt(2.0 / (rows + cols))
    return [
        [rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)
    ]


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


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class Brain:
    """Small MLP policy: obs → action probabilities + value baseline."""

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
        self._loaded_mtime = 0.0
        # Episodes/steps learned since last successful shared save (for merge).
        self._pending_episodes = 0
        self._pending_steps = 0
        self.updated_at = self.created_at
        self.seed = seed if seed is not None else random.randrange(1 << 31)
        self.rng = random.Random(self.seed)

        h = self.hyper.hidden
        self.W1 = _xavier(h, OBS_SIZE, self.rng)
        self.b1 = _zeros1(h)
        self.W2 = _xavier(N_ACTIONS, h, self.rng)
        self.b2 = _zeros1(N_ACTIONS)
        self.Wv = _xavier(1, h, self.rng)
        self.bv = _zeros1(1)

        # Adam moments
        self._m: dict[str, Any] = {}
        self._v: dict[str, Any] = {}
        self._t = 0

    # --- forward / sample -------------------------------------------------

    def forward(
        self, obs: Sequence[float]
    ) -> tuple[list[float], float, list[float]]:
        """Return (action_probs, value, hidden_pre_relu)."""
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
        """Sample an action. Returns (action, log_prob, value)."""
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
        """Pick the highest-probability action (playback / eval)."""
        probs, value, _ = self.forward(obs)
        action = max(range(len(probs)), key=lambda i: probs[i])
        p = max(probs[action], 1e-8)
        return action, math.log(p), value

    @staticmethod
    def decode_action(action: int) -> tuple[float, float, float]:
        """Map action index → (move_x, move_y, run)."""
        d = action // 2
        run = float(action % 2)
        mx, my = DIRS[d]
        return mx, my, run

    # --- learning (REINFORCE + baseline) ----------------------------------

    def learn_episode(
        self,
        obs_list: list[list[float]],
        actions: list[int],
        rewards: list[float],
        values: list[float],
        log_probs: list[float],
    ) -> float:
        """One on-policy update from a finished episode. Returns return."""
        del log_probs  # recomputed with current policy for simplicity
        n = len(rewards)
        if n == 0:
            return 0.0

        gamma = self.hyper.gamma
        returns = _zeros1(n)
        g = 0.0
        for t in range(n - 1, -1, -1):
            g = rewards[t] + gamma * g
            returns[t] = g

        # Normalize advantages for stability
        mean_r = sum(returns) / n
        var = sum((r - mean_r) ** 2 for r in returns) / n
        std = math.sqrt(var) + 1e-6

        lr = self.hyper.learning_rate
        ent_c = self.hyper.entropy_bonus

        # Accumulate gradients then apply Adam once
        gW1 = _zeros(len(self.W1), OBS_SIZE)
        gb1 = _zeros1(len(self.b1))
        gW2 = _zeros(N_ACTIONS, len(self.W1))
        gb2 = _zeros1(N_ACTIONS)
        gWv = _zeros(1, len(self.W1))
        gbv = _zeros1(1)

        for t in range(n):
            obs = obs_list[t]
            action = actions[t]
            ret = returns[t]
            adv = (ret - values[t]) / std

            probs, value, z1 = self.forward(obs)
            h = _relu(z1)

            # Policy gradient: ∇ log π(a) * adv  + entropy
            dlogits = [-p * ent_c for p in probs]
            dlogits[action] += 1.0 * adv + ent_c * (
                -math.log(max(probs[action], 1e-8))
            )
            # Actually entropy grad for softmax is more subtle; use
            # simple: encourage flatness via -ent_c * (log p + 1) on chosen
            # Keep a light entropy push: add -ent_c * log(p_i) into all.
            for i, p in enumerate(probs):
                dlogits[i] += -ent_c * (math.log(max(p, 1e-8)) + 1.0) * p

            # value loss: (V - R)^2 → dV = 2(V-R)
            dv = 2.0 * (value - ret)

            # backprop value head
            for j, hj in enumerate(h):
                gWv[0][j] += dv * hj
            gbv[0] += dv

            # backprop policy head into h
            dh = _zeros1(len(h))
            for i in range(N_ACTIONS):
                gb2[i] += dlogits[i]
                for j, hj in enumerate(h):
                    gW2[i][j] += dlogits[i] * hj
                    dh[j] += dlogits[i] * self.W2[i][j]
            # value into h
            for j in range(len(h)):
                dh[j] += dv * self.Wv[0][j]

            # relu
            for j, zj in enumerate(z1):
                if zj <= 0.0:
                    dh[j] = 0.0

            for j in range(len(self.b1)):
                gb1[j] += dh[j]
                for k, ok in enumerate(obs):
                    gW1[j][k] += dh[j] * ok

        # Average over steps
        inv = 1.0 / n
        self._t += 1
        self._adam_update('W1', self.W1, gW1, lr, inv, is_mat=True)
        self._adam_update('b1', self.b1, gb1, lr, inv, is_mat=False)
        self._adam_update('W2', self.W2, gW2, lr, inv, is_mat=True)
        self._adam_update('b2', self.b2, gb2, lr, inv, is_mat=False)
        self._adam_update('Wv', self.Wv, gWv, lr, inv, is_mat=True)
        self._adam_update('bv', self.bv, gbv, lr, inv, is_mat=False)

        self.episodes += 1
        self.steps += n
        self._pending_episodes += 1
        self._pending_steps += n
        self.total_reward += returns[0]
        self.updated_at = time.time()
        return returns[0]

    def _adam_update(
        self,
        name: str,
        param: Any,
        grad: Any,
        lr: float,
        scale: float,
        *,
        is_mat: bool,
    ) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = self._t
        if name not in self._m:
            if is_mat:
                self._m[name] = _zeros(len(param), len(param[0]))
                self._v[name] = _zeros(len(param), len(param[0]))
            else:
                self._m[name] = _zeros1(len(param))
                self._v[name] = _zeros1(len(param))

        m = self._m[name]
        v = self._v[name]
        if is_mat:
            for i in range(len(param)):
                for j in range(len(param[0])):
                    g = grad[i][j] * scale
                    m[i][j] = beta1 * m[i][j] + (1 - beta1) * g
                    v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g
                    mhat = m[i][j] / (1 - beta1**t)
                    vhat = v[i][j] / (1 - beta2**t)
                    param[i][j] -= lr * mhat / (math.sqrt(vhat) + eps)
                    param[i][j] = _clip(param[i][j], -10.0, 10.0)
        else:
            for i in range(len(param)):
                g = grad[i] * scale
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                mhat = m[i] / (1 - beta1**t)
                vhat = v[i] / (1 - beta2**t)
                param[i] -= lr * mhat / (math.sqrt(vhat) + eps)
                param[i] = _clip(param[i], -10.0, 10.0)

    # --- checkpoint I/O ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            'brain_version': BRAIN_VERSION,
            'obs_size': OBS_SIZE,
            'n_actions': N_ACTIONS,
            'meta': self.meta.to_dict(),
            'hyper': asdict(self.hyper),
            'episodes': self.episodes,
            'steps': self.steps,
            'best_survival': self.best_survival,
            'total_reward': self.total_reward,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'seed': self.seed,
            'rng_state': list(self.rng.getstate()[1]),
            'adam_t': self._t,
            'weights': {
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
                'Wv': self.Wv,
                'bv': self.bv,
            },
            'adam_m': self._m,
            'adam_v': self._v,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Brain:
        if data.get('brain_version') != BRAIN_VERSION:
            raise ValueError(
                f"unsupported brain_version {data.get('brain_version')}"
                f' (want {BRAIN_VERSION})'
            )
        if data.get('obs_size') != OBS_SIZE or data.get('n_actions') != N_ACTIONS:
            raise ValueError('brain obs/action size mismatch')
        brain = cls(
            hyper=HyperParams(**data['hyper']),
            meta=BrainMeta.from_dict(data.get('meta')),
            seed=data['seed'],
        )
        brain.episodes = int(data['episodes'])
        brain.steps = int(data['steps'])
        brain.best_survival = float(data['best_survival'])
        brain.total_reward = float(data['total_reward'])
        brain.created_at = float(data['created_at'])
        brain.updated_at = float(data['updated_at'])
        w = data['weights']
        brain.W1, brain.b1 = w['W1'], w['b1']
        brain.W2, brain.b2 = w['W2'], w['b2']
        brain.Wv, brain.bv = w['Wv'], w['bv']
        brain._t = int(data.get('adam_t', 0))
        brain._m = data.get('adam_m') or {}
        brain._v = data.get('adam_v') or {}
        # Restore RNG far enough for continuous sampling continuity.
        state_mt = data.get('rng_state')
        if state_mt is not None:
            brain.rng.setstate((3, tuple(state_mt), None))
        return brain

    def save(self, path: str) -> None:
        """Atomically write a gzipped JSON brain checkpoint."""
        self.updated_at = time.time()
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = f'{path}.tmp.{os.getpid()}'
        payload = json.dumps(self.to_dict(), separators=(',', ':'))
        with gzip.open(tmp, 'wt', encoding='utf-8') as f:
            f.write(payload)
        os.replace(tmp, path)
        try:
            self._loaded_mtime = os.path.getmtime(path)
        except OSError:
            pass

    def save_shared(self, path: str) -> None:
        """Exclusive-lock then atomic save (multi-worker safe).

        Counters are merged as disk_total + this worker's pending updates so
        parallel Hogwild writers don't under-count experience.
        """
        with _brain_lock(path, exclusive=True):
            pending_ep = self._pending_episodes
            pending_st = self._pending_steps
            if path and os.path.isfile(path):
                try:
                    with gzip.open(path, 'rt', encoding='utf-8') as f:
                        disk = json.load(f)
                    self.episodes = int(disk.get('episodes', 0)) + pending_ep
                    self.steps = int(disk.get('steps', 0)) + pending_st
                    self.best_survival = max(
                        self.best_survival,
                        float(disk.get('best_survival', 0.0)),
                    )
                    self.total_reward = max(
                        self.total_reward,
                        float(disk.get('total_reward', 0.0)),
                    )
                    disk_t = int(disk.get('adam_t', 0))
                    if disk_t > self._t:
                        self._t = disk_t
                except Exception:
                    self.episodes = max(self.episodes, pending_ep)
            self.save(path)
            self._pending_episodes = 0
            self._pending_steps = 0

    def pull_shared(self, path: str) -> bool:
        """If a peer wrote a newer checkpoint, adopt its weights.

        Hogwild-style: take the latest file under a shared lock. Pending
        local episode/step deltas are preserved and applied on next save.
        """
        if not path or not os.path.isfile(path):
            return False
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False
        if mtime <= self._loaded_mtime:
            return False
        with _brain_lock(path, exclusive=False):
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                return False
            if mtime <= self._loaded_mtime:
                return False
            other = Brain.load(path)
        self._adopt_peer(other, mtime)
        return True

    def _adopt_peer(self, other: Brain, mtime: float) -> None:
        # Keep pending_* — those updates are not on disk yet.
        pending_ep = self._pending_episodes
        pending_st = self._pending_steps
        self.W1, self.b1 = other.W1, other.b1
        self.W2, self.b2 = other.W2, other.b2
        self.Wv, self.bv = other.Wv, other.bv
        self._m = other._m
        self._v = other._v
        self._t = max(self._t, other._t)
        self.episodes = other.episodes + pending_ep
        self.steps = other.steps + pending_st
        self.best_survival = max(self.best_survival, other.best_survival)
        self.total_reward = max(self.total_reward, other.total_reward)
        self.hyper = other.hyper
        self.meta = other.meta
        self._loaded_mtime = mtime
        self._pending_episodes = pending_ep
        self._pending_steps = pending_st

    @classmethod
    def load(cls, path: str) -> Brain:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        brain = cls.from_dict(data)
        try:
            brain._loaded_mtime = os.path.getmtime(path)
        except OSError:
            brain._loaded_mtime = time.time()
        return brain

    @classmethod
    def load_or_create(cls, path: str, **kwargs: Any) -> Brain:
        if path and os.path.isfile(path):
            return cls.load(path)
        return cls(**kwargs)


@contextmanager
def _brain_lock(path: str, *, exclusive: bool) -> Iterator[None]:
    """Advisory lock beside the brain file (fcntl; works across processes)."""
    lock_path = f'{path}.lock'
    directory = os.path.dirname(os.path.abspath(lock_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
