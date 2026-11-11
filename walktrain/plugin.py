# Released under the MIT License. See LICENSE for details.
"""Compatibility shim — the playback plugin is now one standalone file.

Install / enable ``walktrain_brain.WalkTrainBrainPlugin`` (see
``walktrain_brain.py`` and ``install_plugin.sh``). This module only re-exports
the class for older imports; it is **not** a ``ba_meta`` plugin entry.
"""

from __future__ import annotations

from walktrain.walktrain_brain import WalkTrainBrainPlugin

__all__ = ['WalkTrainBrainPlugin']
