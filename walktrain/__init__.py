# Released under the MIT License. See LICENSE for details.
"""Headless Spaz duel trainer (real Ballistica physics + resumable brain).

In-game playback is the standalone file ``walktrain_brain.py``
(``walktrain_brain.WalkTrainBrainPlugin``) — not this package.
"""

from walktrain.brain import Brain, BrainMeta, HyperParams

__all__ = ['Brain', 'BrainMeta', 'HyperParams']
