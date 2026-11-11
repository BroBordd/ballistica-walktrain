# Released under the MIT License. See LICENSE for details.
"""Boot WalkTrain inside a running Ballistica process (headless --exec)."""

from __future__ import annotations

import logging
import os
import sys


def _ensure_path() -> None:
    # Allow `import walktrain` whether we live next to the binary or in-repo.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (root, os.path.dirname(here)):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    # Parent of the walktrain package directory.
    pkg_parent = os.path.dirname(here)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)


def start(
    *,
    brain_path: str | None = None,
    speed_exponent: int = 3,
    save_every: int = 5,
    max_episode_seconds: float = 30.0,
) -> None:
    """Replace the host session with the walk trainer."""
    import babase
    import bascenev1 as bs

    _ensure_path()
    from walktrain.activity import WalkTrainSession, _DEFAULT_BRAIN

    path = brain_path or os.environ.get('WALKTRAIN_BRAIN', _DEFAULT_BRAIN)
    shared_env = os.environ.get('WALKTRAIN_SHARED', '1').strip().lower()
    settings = {
        'brain_path': path,
        'speed_exponent': int(
            os.environ.get('WALKTRAIN_SPEED', speed_exponent)
        ),
        'save_every': int(os.environ.get('WALKTRAIN_SAVE_EVERY', save_every)),
        'max_episode_seconds': float(
            os.environ.get('WALKTRAIN_MAX_EP', max_episode_seconds)
        ),
        'worker_id': int(os.environ.get('WALKTRAIN_WORKER_ID', '0')),
        'shared_brain': shared_env not in ('0', 'false', 'no', 'off'),
    }
    print(f'WalkTrain boot → {settings}', flush=True)

    def _go() -> None:
        # new_host_session wants a Session subclass, not an instance.
        from walktrain import activity as actmod

        actmod.SESSION_SETTINGS = settings
        print('WalkTrain: launching WalkTrainSession', flush=True)
        bs.new_host_session(WalkTrainSession)

    # Wait for classic mode / main menu to come up, then hijack the session.
    babase.apptimer(2.5, _go)


def main() -> None:
    """Entry used by --exec 'import walktrain.boot; walktrain.boot.main()'."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    _ensure_path()
    start()


# Running as --exec module body:
if __name__ != 'walktrain.boot':
    # When exec'd as a script snippet, still start.
    pass
