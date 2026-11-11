# ba_meta require api 9
# Released under the MIT License. See LICENSE for details.
"""BombSquad 1.8 / API 9 plugin: play a walktrain ``.brain`` in-game.

Enable under Settings → Plugins, put a brain file in your mods ``brains/``
folder (or set the path in the plugin settings), join a match on the map
named in the brain metadata, then chat ``/brainbot`` (host) to spawn it.
"""

from __future__ import annotations

import os
from typing import Any, override

import babase
import bascenev1 as bs

_CONFIG_BRAIN_PATH = 'WalkTrain Brain Path'
_CONFIG_GREEDY = 'WalkTrain Brain Greedy'


def _default_brain_path() -> str:
    env = babase.app.env
    user = env.python_directory_user
    if user:
        return os.path.join(user, 'brains', 'spaz_walk.brain')
    # Repo / developer fallback next to the walktrain package.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, 'output', 'spaz_walk.brain')


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
    val = babase.app.config.get(_CONFIG_GREEDY, True)
    return bool(val)


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


def _load_brain(path: str) -> Any:
    from walktrain.brain import Brain

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f'brain file not found: {path}')
    return Brain.load(path)


# ba_meta export babase.Plugin
class WalkTrainBrainPlugin(babase.Plugin):
    """Load a walktrain brain and spawn a playback bot on a matching map."""

    def __init__(self) -> None:
        self._player: Any | None = None
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
        # Host only (client_id -1). Keep the command local.
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
        from walktrain.player import BrainPlayer

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
    """Minimal settings: brain path + spawn/despawn helpers."""

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
