#!/usr/bin/env python3
"""Paired first-action divergence, post-processing only. Python 3.8+, NumPy.

The observation-age protocol names "first action divergence relative to the paired condition"
without defining it. Two readings exist and they are not interchangeable, so both are reported:

- first_action: distance between the two tick-0 actions. Under paired noise and identical first
  inputs this measures numerics, not observation age, because staleness cannot act at tick 0.
- first_divergent_tick: the earliest tick whose actions differ, and the distance there.

No threshold, and no change to any protocol or schedule. Which reading counts is Alba's (audit R16).
"""
import argparse
import json
from pathlib import Path


def distance(a, b):
    import numpy as np
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"l2": float(np.linalg.norm(difference)), "max_abs": float(np.abs(difference).max())}


def first_action_divergence(actions_a, actions_b):
    """Both arrays are ticks x action_dim. Compares only the common prefix of the two episodes."""
    import numpy as np
    a, b = np.asarray(actions_a), np.asarray(actions_b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected ticks x dim with equal dim, got {} and {}".format(a.shape, b.shape))
    if not len(a) or not len(b):
        raise ValueError("empty action array")
    common = min(len(a), len(b))
    differs = np.any(a[:common] != b[:common], axis=1)
    tick = int(np.argmax(differs)) if differs.any() else None
    return {"first_action": distance(a[0], b[0]), "first_divergent_tick": tick,
            "at_first_divergent_tick": None if tick is None else distance(a[tick], b[tick]),
            "common_ticks": common, "lengths": [len(a), len(b)]}


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("action_array_a", type=Path)
    parser.add_argument("action_array_b", type=Path)
    args = parser.parse_args()
    print(json.dumps(first_action_divergence(np.load(args.action_array_a), np.load(args.action_array_b)), indent=2))


if __name__ == "__main__":
    main()
