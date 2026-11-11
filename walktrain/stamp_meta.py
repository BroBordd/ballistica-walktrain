# Released under the MIT License. See LICENSE for details.
"""Stamp / refresh ``meta`` on an existing ``.brain`` without retraining.

Usage:
  python -m walktrain.stamp_meta output/spaz_walk.brain
  python -m walktrain.stamp_meta output/spaz_walk.brain --map "The Pad"
"""

from __future__ import annotations

import argparse
import sys

from walktrain.brain import Brain, BrainMeta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Stamp brain file metadata')
    parser.add_argument('brain', help='Path to .brain checkpoint')
    parser.add_argument('--map', default='The Pad', help='Map display name')
    parser.add_argument('--map-type', default='ThePad', help='Map class name')
    parser.add_argument('--task', default='walk_no_fall')
    args = parser.parse_args(argv)

    brain = Brain.load(args.brain)
    brain.meta = BrainMeta(
        map_name=args.map,
        map_type=args.map_type,
        task=args.task,
        trainer='walktrain',
        pad_center=brain.meta.pad_center,
        near_edge_radius=brain.meta.near_edge_radius,
        pos_scale=brain.meta.pos_scale,
        vel_scale=brain.meta.vel_scale,
        fall_y_drop=brain.meta.fall_y_drop,
        railing_removed_in_training=brain.meta.railing_removed_in_training,
        notes=brain.meta.notes,
    )
    brain.save(args.brain)
    print(
        f'Stamped {args.brain}: map={brain.meta.map_name} '
        f'task={brain.meta.task} episodes={brain.episodes}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
