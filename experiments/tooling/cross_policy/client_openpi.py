#!/usr/bin/env python3
"""OpenPI LIBERO client attempts and parity. Python 3.8. NOT YET EXECUTED AGAINST A MODEL.

Environment, settling, image handling and state vector follow examples/libero/main.py at the pinned
revision. Differences from the example: attempts come from a schedule, state and noise are keyed,
and h and d vary by condition. The client itself uses no GPU and connects to server_openpi.py.

    client_openpi.py attempts --phase confirmation --schedule S --states N --output O
    client_openpi.py parity --output O
"""
import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_guard  # noqa: E402
from common import SUITES, TASK_IDS, atomic_write_json, sha  # noqa: E402
from journal import AttemptJournal  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION, RESIZE, WAIT_STEPS = 256, 224, 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
PARITY_SEED, PARITY_EPISODES, NATIVE_H, NATIVE_D = 790002, 10, 5, 10
EXAMPLE_ENV_SEED = 7


def quat2axisangle(quat):
    import numpy as np
    w = max(min(quat[3], 1.0), -1.0)
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def digest(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    return hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode() + array.tobytes()).hexdigest()


def make_env(task):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RESOLUTION, camera_widths=ENV_RESOLUTION)
    env.seed(EXAMPLE_ENV_SEED)  # examples/libero/main.py default; set once per environment, as there.
    return env, str(task.language)


def policy_input(obs, prompt):
    import numpy as np
    from openpi_client import image_tools
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {"observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, RESIZE, RESIZE)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE)),
            "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
            "prompt": prompt}


def input_hash(element):
    return sha({k: digest(v) if k != "prompt" else v for k, v in element.items()})


def rollout(row, env, prompt, client, state, record_video=False, expected_reset=None):
    import numpy as np
    ident, h, d = row["episode_identity"], row["h"], row["d"]
    refresh = row.get("refresh_every_queries", 1)
    env.reset()
    obs = env.set_init_state(state)
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    settled = digest(np.asarray(env.sim.get_state().flatten()))
    first_observation = input_hash(policy_input(obs, prompt))
    if expected_reset is not None and (settled, first_observation) != expected_reset:
        raise ValueError("reset_state_mismatch: state/observation pair differs before policy execution")
    plan = collections.deque()
    actions, queries, frames, server_timing, memory, done = [], [], [], [], None, False
    supplied, supplied_tick, chunk_record = None, 0, None
    started = time.perf_counter()
    for tick in range(MAX_STEPS[ident["suite"]]):
        current = policy_input(obs, prompt)
        refreshed = tick % refresh == 0
        if refreshed:
            supplied, supplied_tick = current, tick
        generates = not plan
        chunk_record = None
        if generates:
            request = dict(supplied, __d__=d, __noise_key__=json.dumps([ident, tick]))
            begin = time.perf_counter()
            response = client.infer(request)
            if response.get("server_configured_num_steps") != d:
                raise ValueError("unsupported_refinement_setting: server did not attest configured d={}".format(d))
            chunk = np.asarray(response["actions"])
            if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all() or len(chunk) < h:
                raise ValueError("unsupported_refinement_setting: chunk {} finite={}".format(chunk.shape, bool(np.isfinite(chunk).all())))
            chunk_record = {"shape": list(chunk.shape), "sha256": digest(chunk)}
            server_timing.append({"tick": tick, "client_round_trip_s": time.perf_counter() - begin,
                                  "host_generation_s": response["host_generation_s"],
                                  "server_configured_num_steps": response["server_configured_num_steps"]})
            memory = response["memory"]
            plan.extend(chunk[:h])
        action = plan.popleft()
        queries.append({"tick": tick, "refresh": refreshed, "new_generation": generates,
                        "current_observation_sha256": input_hash(current), "supplied_input_sha256": input_hash(supplied),
                        "supplied_observation_age": tick - supplied_tick, "chunk": chunk_record,
                        "executed_action_sha256": digest(np.asarray(action))})
        actions.append(np.asarray(action))
        if record_video:
            frames.append(current["observation/image"])
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
    return {"outcome": {"success": bool(done), "episode_length": len(actions),
                        "termination_reason": "success" if done else "task_horizon"},
            "settled_state_sha256": settled, "first_observation_sha256": first_observation,
            "queries": queries, "actions": np.asarray(actions), "frames": frames,
            "timing": {"rollout_wall_s": time.perf_counter() - started, "generation": server_timing},
            "memory": memory}


def provenance(args):
    import provenance as pv
    return pv.capture("pi05_client", args, {})


def connect(port):
    from openpi_client import websocket_client_policy
    return websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)


def attempts(args):
    import numpy as np
    from libero.libero import benchmark
    rows = json.loads(args.schedule.read_text("utf-8"))["schedule"]
    if not all(row["state_sha256"] for row in rows):
        raise SystemExit("schedule has no bound state hashes")
    states = np.load(str(args.states))
    journal = AttemptJournal(args.output / "attempts")
    print("interrupted:", journal.reconcile_interrupted("process restart"), flush=True)
    run_provenance = provenance(args)
    atomic_write_json(args.output / "provenance.{}.json".format(run_provenance["sha256"][:12]), run_provenance)
    client, suites, env, key, settled = connect(args.port), benchmark.get_benchmark_dict(), None, None, {}
    for row in rows:
        if journal.attempts_for(row["planned_run_id"]):
            continue
        path = journal.begin(row, run_provenance["sha256"], sys.argv)
        journal.update(path, "running", started_unix=time.time())
        try:
            ident = row["episode_identity"]
            if (ident["suite"], ident["task_id"]) != key:
                if env is not None:
                    env.close()
                env, prompt = make_env(suites[ident["suite"]]().get_task(ident["task_id"]))
                key = (ident["suite"], ident["task_id"])
            state = states[row["state_key"]]
            if hashlib.sha256(state.tobytes()).hexdigest() != row["state_sha256"]:
                raise ValueError("state hash mismatch")
            result = rollout(row, env, prompt, client, state,
                             expected_reset=settled.get(row["block_id"]))
            reset_pair = (result["settled_state_sha256"], result["first_observation_sha256"])
            expected = settled.setdefault(row["block_id"], reset_pair)
            arrays = args.output / "arrays"
            arrays.mkdir(exist_ok=True)
            array_path = arrays / (path.stem + ".npy")
            np.save(str(array_path), result["actions"])
            journal.update(path, "completed", outcome=result["outcome"], queries=result["queries"],
                           settled_state_sha256=result["settled_state_sha256"],
                           settled_state_matches_block=expected == reset_pair,
                           first_observation_sha256=result["first_observation_sha256"],
                           action_array=array_path.name, timing=result["timing"], memory=result["memory"], finished_unix=time.time())
        except Exception as error:  # Retained as a record; never replaced.
            journal.update(path, "failed", error=repr(error), finished_unix=time.time())


def parity(args):
    import imageio
    import numpy as np
    from libero.libero import benchmark
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output / "provenance.json", provenance(args))
    journal = AttemptJournal(args.output / "parity_attempts")
    videos = args.output / "videos"
    videos.mkdir()
    client, suites = connect(args.port), benchmark.get_benchmark_dict()
    for suite_name in SUITES:
        suite = suites[suite_name]()
        for task_id in TASK_IDS:
            task = suite.get_task(task_id)
            env, prompt = make_env(task)
            official = suite.get_task_init_states(task_id)
            for episode in range(PARITY_EPISODES):
                ident = {"suite": suite_name, "task_id": task_id, "state_slot": "official{}".format(episode), "policy_seed": PARITY_SEED}
                row = {"planned_run_id": "parity.pi05.{}.task{:03d}.ep{}".format(suite_name, task_id, episode), "h": NATIVE_H, "d": NATIVE_D,
                       "episode_identity": ident, "condition_id": "NATIVE"}
                path = journal.begin(row, sha({"preflight": str(args.output)}), sys.argv)
                journal.update(path, "running")
                try:
                    result = rollout(row, env, prompt, client, official[episode], record_video=True)
                    video = videos / (path.stem + ".mp4")
                    imageio.mimwrite(str(video), result["frames"], fps=10)
                    journal.update(path, "completed", outcome=result["outcome"], first_action_sha256=result["queries"][0]["executed_action_sha256"],
                                   action_shape=list(result["actions"].shape), action_abs_max=float(np.abs(result["actions"]).max()),
                                   queries=result["queries"], video=video.name, memory=result["memory"])
                except Exception as error:
                    journal.update(path, "failed", error=repr(error))
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("attempts", "parity"))
    parser.add_argument("--phase", default="preflight", choices=("preflight", "confirmation", "observation_age"))
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    # The server owns the GPU. Its process is expected, so the guard cannot require an empty GPU here.
    launch_guard.check(args.phase, processes=lambda: [],
                       schedule_path=args.schedule if args.mode == "attempts" else None,
                       model="pi05" if args.mode == "attempts" else None)
    (attempts if args.mode == "attempts" else parity)(args)


if __name__ == "__main__":
    main()
