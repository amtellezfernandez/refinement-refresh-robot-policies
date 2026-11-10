#!/usr/bin/env python3
"""Generate and serialize fresh LIBERO initial states. CPU only: no renderer, no policy.

Run in the SmolVLA environment:
    generate_states.py --cohort cross_policy_confirmation_v1 --output DIR

A state is sampled by the benchmark's own BDDL placement sampler through env.reset()
under a seed keyed by cohort/suite/task/slot/draw. A draw is rejected only on grounds
visible before any policy runs: task already satisfied, or a non-finite state. Every
rejected draw is recorded. No policy outcome can reach this code.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STATE_SLOTS, SUITES, TASK_IDS, atomic_write_json, file_sha, keyed_int, sha  # noqa: E402

MAX_DRAWS = 20


def make_env(task):
    import os
    from libero.libero import get_libero_path
    from libero.libero.envs import TASK_MAPPING
    from libero.libero.envs.bddl_utils import get_problem_info
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    info = get_problem_info(bddl)
    # Same controller and horizon defaults as ControlEnv, without any renderer.
    env = TASK_MAPPING[info["problem_name"].lower()](
        bddl_file_name=bddl, robots=["Panda"], controller_configs=controller_config(),
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        ignore_done=True, reward_shaping=True, control_freq=20, horizon=1000)
    return env, bddl


def controller_config():
    from robosuite import load_controller_config
    return load_controller_config(default_controller="OSC_POSE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from libero.libero import benchmark
    suites = benchmark.get_benchmark_dict()
    states, rows, rejected = {}, [], []
    for suite_name in SUITES:
        suite = suites[suite_name]()
        for task_id in TASK_IDS:
            task = suite.get_task(task_id)
            env, bddl = make_env(task)
            for slot in STATE_SLOTS:
                for draw in range(MAX_DRAWS):
                    seed = keyed_int("state", args.cohort, suite_name, task_id, slot, draw)
                    np.random.seed(seed)
                    env.reset()
                    state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
                    # _check_success is the same predicate the evaluators read as success.
                    reasons = []
                    if not np.isfinite(state).all():
                        reasons.append("non_finite_state")
                    if env._check_success():
                        reasons.append("task_already_satisfied")
                    if not reasons:
                        break
                    rejected.append({"suite": suite_name, "task_id": task_id, "state_slot": slot,
                                     "draw": draw, "seed": seed, "reasons": reasons})
                else:
                    raise RuntimeError("no valid draw for {} {} {}".format(suite_name, task_id, slot))
                # Exact reset-state reuse: restoring the state must reproduce it bit for bit.
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                restored = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
                if restored.tobytes() != state.tobytes():
                    raise RuntimeError("state did not round trip: {} {} {}".format(suite_name, task_id, slot))
                key = "{}.task{:03d}.slot{}".format(suite_name, task_id, slot)
                states[key] = state
                rows.append({"key": key, "suite": suite_name, "task_id": task_id, "state_slot": slot,
                             "draw": draw, "seed": seed, "state_length": int(state.size),
                             "state_sha256": __import__("hashlib").sha256(state.tobytes()).hexdigest(),
                             "task_bddl_sha256": file_sha(bddl), "task_language": task.language})
            env.close()
            print(suite_name, task_id, "done", flush=True)
    np.savez(args.output / "initial_states.npz", **states)
    import libero as libero_package
    import mujoco
    import robosuite
    manifest = {"schema_version": 1, "cohort": args.cohort, "state_count": len(rows), "states": rows,
                "rejected_draws": rejected, "rejected_draw_count": len(rejected),
                "states_sha256": sha(rows),
                "initial_states_npz_sha256": file_sha(args.output / "initial_states.npz"),
                "generator_sha256": file_sha(__file__),
                "versions": {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
                             "libero_path": str(libero_package.__file__)},
                "generation_rule": "BDDL placement sampler via env.reset(); keyed numpy seed; "
                                   "rejection only on pre-policy grounds, all rejections recorded."}
    atomic_write_json(args.output / "state_manifest.json", manifest)
    print(json.dumps({"states": len(rows), "rejected": len(rejected),
                      "npz": manifest["initial_states_npz_sha256"]}))


if __name__ == "__main__":
    main()
