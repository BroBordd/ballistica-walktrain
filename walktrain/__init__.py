# ba_meta require api 9
# Released under the MIT License. See LICENSE for details.
"""Headless Spaz walk trainer (real Ballistica physics + resumable brain).

Also ships ``walktrain.plugin.WalkTrainBrainPlugin`` for BombSquad 1.8
(API 9) playback of ``.brain`` checkpoints.
"""

from walktrain.brain import Brain, BrainMeta, HyperParams

__all__ = ['Brain', 'BrainMeta', 'HyperParams']
