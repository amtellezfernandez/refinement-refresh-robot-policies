#!/usr/bin/env python3
"""Record synthetic fixture bundles on THIS machine.  No paper evidence is read.

Tasks used are outside every paper task set (development Spatial 5-8, confirmation
Object 5-8 and Goal 5-8):  libero_object task 2 (salad dressing) and libero_goal
task 1 (bowl on stove).

Sources of action streams, none produced by a policy:
  demo      LIBERO human teleoperation demos (yifengzhu-hf/LIBERO-datasets @ f13aa24),
            played open-loop from the official init state the task file holds;
            stop at first success, otherwise padded with no-op to the horizon.
  zero      all-zero translation/rotation, gripper open: a known failure.
  random    seeded uniform actions: a known failure with a non-trivial trajectory.

The recorder uses the SAME reset/step code as the replayer, so fixtures test the
comparison, validation and determinism logic.  They cannot test whether that reset
path matches the original LeRobot wrapper; that is the protocol's Stage 0.

Usage: make_fixtures.py DEMO_DIR OUT.jsonl [--demos-per-task N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import libero_replay as lr  # noqa: E402
from replay import canonical_str, keyed_seed  # noqa: E402
import hashlib  # noqa: E402

TASKS = [("libero_object", 2, "libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5"),
         ("libero_goal", 1, "libero_goal/put_the_bowl_on_the_stove_demo.hdf5")]
PAPER_TASKS = {("libero_spatial", t) for t in (5, 6, 7, 8)} | {("libero_object", t) for t in (5, 6, 7, 8)} | \
              {("libero_goal", t) for t in (5, 6, 7, 8)}


def rec(a, dtype):
    a = np.ascontiguousarray(np.asarray(a, dtype=dtype))
    return {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(), "sha256": lr.array_sha(a)}


def record(rep, suite, task_id, state_id, actions, key, source, vector=None):
    if vector is None:
        states, _ = lr.official_init_states(suite, task_id)
        pre = np.asarray(states[state_id], dtype=np.float64)
        reset_source = {"kind": "official", "initial_state_id": state_id}
        seed = keyed_seed([suite, task_id, state_id], domain="environment")
    else:
        pre = np.asarray(vector, dtype=np.float64)
        reset_source = {"kind": "vector", "state": rec(pre, np.float64)}
        seed = keyed_seed([suite, task_id, key], domain="environment")
    init = rep.reset(pre, seed)
    flags, used = [], []
    for t in range(rep.horizon):
        a = np.asarray(actions[t] if t < len(actions) else [0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        s, _, _ = rep.step(a)
        used.append(rec(a, np.float32))
        flags.append(s)
        if s:
            break
    ok = any(flags)
    return {"schema": "replay_bundle.v1", "provenance": "synthetic_fixture", "episode_key": key,
            "fixture_source": source, "suite": suite, "task_id": task_id,
            "reset_source": reset_source,
            "environment_seed": seed, "recorded_initial_state": rec(init, np.float64),
            "actions": used, "action_ticks": list(range(len(used))), "recorded_success_flags": flags,
            "label": {"success": ok, "steps": len(used), "termination_reason": "success" if ok else "task_horizon"},
            "horizon": rep.horizon, "terminal_state_sha256": rep.state_sha(),
            "recorded_action_stream_sha256": hashlib.sha256(canonical_str([u["sha256"] for u in used])).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo_dir")
    ap.add_argument("out")
    ap.add_argument("--demos-per-task", type=int, default=6)
    a = ap.parse_args()
    import h5py
    out = Path(a.out)
    with out.open("x") as f:
        for suite, task_id, rel in TASKS:
            assert (suite, task_id) not in PAPER_TASKS
            rep = lr.Replayer(suite, task_id, render=True)
            h = h5py.File(Path(a.demo_dir) / rel, "r")
            demos = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))[: a.demos_per_task]
            for i, dk in enumerate(demos):
                acts = np.asarray(h["data"][dk]["actions"])
                # the demo's own recorded start (states[0]); open-loop success is not guaranteed after
                # the 10 settle steps and a MuJoCo version change, and both outcomes are useful fixtures
                b = record(rep, suite, task_id, None, acts, f"fixture.{suite}.t{task_id}.demo_{dk}",
                           {"kind": "demo", "demo": dk, "file": rel}, vector=np.asarray(h["data"][dk]["states"][0]))
                f.write(json.dumps(b) + "\n"); f.flush()
                print(b["episode_key"], b["label"], flush=True)
            b = record(rep, suite, task_id, 10, np.zeros((0, 7)), f"fixture.{suite}.t{task_id}.s10.zero",
                       {"kind": "zero"})
            f.write(json.dumps(b) + "\n"); f.flush()
            print(b["episode_key"], b["label"], flush=True)
            rng = np.random.default_rng(20261104 + task_id)
            b = record(rep, suite, task_id, 11, rng.uniform(-1, 1, size=(400, 7)),
                       f"fixture.{suite}.t{task_id}.s11.random", {"kind": "random", "seed": 20261104 + task_id})
            f.write(json.dumps(b) + "\n"); f.flush()
            print(b["episode_key"], b["label"], flush=True)
            rep.close()


if __name__ == "__main__":
    main()
