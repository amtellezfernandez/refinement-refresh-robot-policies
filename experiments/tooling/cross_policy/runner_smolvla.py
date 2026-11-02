#!/usr/bin/env python3
"""SmolVLA attempts and preflight on pinned LeRobot. Python 3.12. NOT YET EXECUTED ON A GPU.

The guard runs before torch is imported. Reuses run_paired.py helpers without changing that file.

    ALBA_GPU_RESERVATION=<rid> runner_smolvla.py preflight --checkpoint DIR --output DIR
    ALBA_GPU_RESERVATION=<rid> runner_smolvla.py attempts --phase confirmation \
        --schedule experiments/runs/<cohort>/smolvla/schedule.json \
        --states experiments/runs/<cohort>/states/initial_states.npz --checkpoint DIR --output DIR
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import launch_guard  # noqa: E402
from common import SUITES, TASK_IDS, atomic_write_json, sha  # noqa: E402
from journal import AttemptJournal  # noqa: E402

PROFILE_D = (2, 5, 10)
PARITY_EPISODES_PER_TASK = 10
PARITY_SEED = 790001  # Disjoint from every recorded and planned policy seed.


def gpu_sampler(stop, samples, device_index=0, period_s=0.5):
    import subprocess
    while not stop.wait(period_s):
        output = subprocess.check_output(["nvidia-smi", "-i", str(device_index), "--query-gpu=memory.used,utilization.gpu",
                                          "--format=csv,noheader,nounits"], universal_newlines=True)
        samples.append([time.time()] + [float(x) for x in output.strip().split(",")])


class ChunkTap:
    """Records each generated chunk's hash and shape at the policy boundary."""
    def __init__(self, policy):
        self.policy, self.chunks = policy, []
        self.original = policy._get_action_chunk
        def chunk(*args, **kwargs):
            result = self.original(*args, **kwargs)
            array = result.detach().to("cpu").numpy()
            self.chunks.append({"shape": list(array.shape), "finite": bool(__import__("numpy").isfinite(array).all()),
                                "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
            return result
        policy._get_action_chunk = chunk

    def restore(self):
        self.policy._get_action_chunk = self.original


def rollout(row, env, policy, pre, post, env_pre, env_post, state, record_video=False, expected_reset=None):
    """One episode. Writes nothing itself; returns arrays and per-query records."""
    import numpy as np
    import run_paired as rp
    import torch
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.utils.constants import ACTION
    ident, h, d = row["episode_identity"], row["h"], row["d"]
    refresh = row.get("refresh_every_queries", 1)
    base = env.envs[0].unwrapped
    # Upstream reads _init_states[init_state_id], so substituting the array needs no source change.
    base._init_states = np.asarray(state)[None, ...]
    policy.config.n_action_steps, policy.config.num_steps = h, d
    policy.reset()
    for processor in (pre, post, env_pre, env_post):
        if hasattr(processor, "reset"):
            processor.reset()
    base.init_state_id = 0
    environment_seed = rp.keyed_seed([ident["suite"], ident["task_id"], ident["state_slot"]], domain="environment")
    rp.seed_environment(environment_seed)
    observation, _ = env.reset(seed=[environment_seed], options={NEW_ROLLOUT_OPTION: True})
    settled = rp.array_record(base._env.sim.get_state().flatten())["sha256"]
    first_observation = rp.observation_sha(observation)
    if expected_reset is not None and (settled, first_observation) != expected_reset:
        raise ValueError("reset_state_mismatch: state/observation pair differs before policy execution")
    tap = rp_tap = ChunkTap(policy)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    actions, queries, frames = [], [], []
    host_s, success, terminated, truncated = [], False, False, False
    supplied, supplied_tick = None, 0
    started = time.perf_counter()
    try:
        with torch.inference_mode(), (torch.autocast(device_type="cuda") if policy.config.use_amp else nullcontext()):
            for tick in range(int(base._max_episode_steps)):
                refreshed = tick % refresh == 0
                if refreshed:
                    supplied, supplied_tick = rp.copy.deepcopy(observation), tick
                batch = rp.prepare_observation(supplied, env, env_pre, pre)
                generates = len(policy._queues[ACTION]) == 0
                noise = rp.make_noise(policy, ident, tick) if generates else None
                chunks_before = len(tap.chunks)
                query_start = time.perf_counter()
                action = env_post({ACTION: post(policy.select_action(batch, noise=noise))})[ACTION]
                host_s.append(time.perf_counter() - query_start)
                array = action.detach().to("cpu").numpy()
                rp.require(array.shape == (1, 7) and np.isfinite(array).all(), "invalid action")
                queries.append({"tick": tick, "refresh": refreshed, "new_generation": generates,
                                "current_observation_sha256": rp.observation_sha(observation),
                                "supplied_input_sha256": rp.observation_sha(supplied),
                                "supplied_observation_age": tick - supplied_tick,
                                "chunk": tap.chunks[-1] if len(tap.chunks) > chunks_before else None,
                                "executed_action_sha256": rp.array_record(array[0])["sha256"]})
                actions.append(array[0])
                if record_video:
                    frames.append(observation["pixels"]["image"][0].copy())
                observation, _, terminated_array, truncated_array, info = env.step(array)
                terminated, truncated = bool(terminated_array[0]), bool(truncated_array[0])
                success = success or rp.batch_one_success(info)
                if terminated or truncated:
                    break
    finally:
        rp_tap.restore()
    return {"outcome": {"success": success, "episode_length": len(actions),
                        "termination_reason": "success" if success else "terminated" if terminated else
                        "truncated" if truncated else "task_horizon"},
            "settled_state_sha256": settled, "first_observation_sha256": first_observation,
            "queries": queries, "actions": np.asarray(actions), "frames": frames,
            "timing": {"rollout_wall_s": time.perf_counter() - started, "query_host_s": host_s,
                       "cuda_event": "see profile; query-level CUDA events not separated from env post-processing"},
            "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()}}


def make_task_env(suite, task_id, policy, image_size=360):
    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    config = LiberoEnv(task=suite, task_ids=[task_id], init_states=True, hard_reset=True, control_mode="relative",
                       observation_height=image_size, observation_width=image_size)
    env = make_env(config, n_envs=1, use_async_envs=False)[suite][task_id]
    env_pre, env_post = make_env_pre_post_processors(config, policy.config)
    return env, env_pre, env_post


def provenance(args):
    import provenance as pv
    return pv.capture("smolvla", args, {"checkpoint_dir": str(args.checkpoint)})


def attempts(args):
    import numpy as np
    import run_paired as rp
    document = json.loads(args.schedule.read_text("utf-8"))
    rows = document["schedule"]
    states = np.load(args.states)
    rp.require(all(row["state_sha256"] for row in rows), "schedule has no bound state hashes")
    journal = AttemptJournal(args.output / "attempts")
    print("interrupted:", journal.reconcile_interrupted("process restart"), flush=True)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rp.seed_environment(0)
    policy, pre, post, _ = rp.load_policy_and_processors(args.checkpoint)
    run_provenance = provenance(args)
    atomic_write_json(args.output / "provenance.{}.json".format(run_provenance["sha256"][:12]), run_provenance)
    env, key, env_pre, env_post, settled = None, None, None, None, {}
    for row in rows:
        if journal.attempts_for(row["planned_run_id"]):
            continue
        path = journal.begin(row, run_provenance["sha256"], sys.argv)
        journal.update(path, "running", started_unix=time.time())
        stop, samples = threading.Event(), []
        sampler = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
        sampler.start()
        try:
            ident = row["episode_identity"]
            task_key = (ident["suite"], ident["task_id"])
            if task_key != key:
                if env is not None:
                    env.close()
                env, env_pre, env_post = make_task_env(ident["suite"], ident["task_id"], policy)
                key = task_key
            state = states[row["state_key"]]
            rp.require(hashlib.sha256(state.tobytes()).hexdigest() == row["state_sha256"], "state hash mismatch")
            result = rollout(row, env, policy, pre, post, env_pre, env_post, state,
                             expected_reset=settled.get(row["block_id"]))
            reset_pair = (result["settled_state_sha256"], result["first_observation_sha256"])
            expected = settled.setdefault(row["block_id"], reset_pair)
            arrays = args.output / "arrays"
            arrays.mkdir(exist_ok=True)
            array_path = arrays / (path.stem + ".npy")
            np.save(array_path, result["actions"])
            journal.update(path, "completed", outcome=result["outcome"], queries=result["queries"],
                           settled_state_sha256=result["settled_state_sha256"],
                           settled_state_matches_block=expected == reset_pair,
                           first_observation_sha256=result["first_observation_sha256"],
                           action_array=array_path.name, timing=result["timing"], memory=result["memory"],
                           gpu_samples=samples, finished_unix=time.time())
        except Exception as error:  # Retained as a record; never replaced.
            journal.update(path, "failed", error=repr(error), gpu_samples=samples, finished_unix=time.time())
        finally:
            stop.set()
            sampler.join()


def preflight(args):
    import imageio.v2 as imageio
    import numpy as np
    import run_paired as rp
    import torch
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rp.seed_environment(0)
    args.output.mkdir(parents=True, exist_ok=False)
    policy, pre, post, native = rp.load_policy_and_processors(args.checkpoint)
    atomic_write_json(args.output / "provenance.json", provenance(args))
    # 1. Profile on a fixed input: 100 warm-up then 100 measured calls per d.
    env, env_pre, env_post = make_task_env("libero_spatial", 0, policy)
    base = env.envs[0].unwrapped
    base.init_state_id = 0
    observation, _ = env.reset(seed=[rp.keyed_seed(["profile"], domain="environment")])
    batch = rp.prepare_observation(observation, env, env_pre, pre)
    profile = {}
    for d in PROFILE_D:
        policy.config.num_steps = d
        counters, host, cuda = rp.Counters(policy), [], []
        with torch.inference_mode():
            for index in range(100):
                policy.reset()
                policy._get_action_chunk(rp.copy.deepcopy(batch), noise=rp.make_noise(policy, {"profile": d}, index, "warmup"))
            counters.reset()
            torch.cuda.reset_peak_memory_stats()
            for index in range(100):
                policy.reset()
                chunk = policy._get_action_chunk(rp.copy.deepcopy(batch), noise=rp.make_noise(policy, {"profile": d}, index, "measure"))
        rp.require(bool(torch.isfinite(chunk).all()), "non-finite chunk at d={}".format(d))
        profile[str(d)] = {"host_s": counters.host_s, "cuda_ms": counters.cuda_ms, "counts": counters.counts,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "chunk_shape": list(chunk.shape)}
        counters.restore()
    atomic_write_json(args.output / "profile.json", profile)
    env.close()
    # 2. Native parity, 4 suites x 10 tasks x 10 episodes, on official init states 0-9.
    # Fresh campaign states are not touched, so parity cannot expose a confirmation identity.
    journal = AttemptJournal(args.output / "parity_attempts")
    run_provenance = sha({"preflight": str(args.output)})
    videos = args.output / "videos"
    videos.mkdir()
    for suite in SUITES:
        for task_id in TASK_IDS:
            env, env_pre, env_post = make_task_env(suite, task_id, policy)
            official = np.array(env.envs[0].unwrapped._init_states)
            for episode in range(PARITY_EPISODES_PER_TASK):
                ident = {"suite": suite, "task_id": task_id, "state_slot": "official{}".format(episode), "policy_seed": PARITY_SEED}
                row = {"planned_run_id": "parity.smolvla.{}.task{:03d}.ep{}".format(suite, task_id, episode), "h": native["h"],
                       "d": native["d"], "episode_identity": ident, "condition_id": "NATIVE"}
                path = journal.begin(row, run_provenance, sys.argv)
                journal.update(path, "running")
                try:
                    result = rollout(row, env, policy, pre, post, env_pre, env_post, official[episode], record_video=True)
                    video = videos / (path.stem + ".mp4")
                    imageio.mimwrite(str(video), result["frames"], fps=20)
                    journal.update(path, "completed", outcome=result["outcome"], first_action_sha256=result["queries"][0]["executed_action_sha256"],
                                   action_shape=list(result["actions"].shape), action_abs_max=float(np.abs(result["actions"]).max()),
                                   queries=result["queries"], video=video.name, memory=result["memory"])
                except Exception as error:
                    journal.update(path, "failed", error=repr(error))
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "attempts"))
    parser.add_argument("--phase", default="preflight", choices=("preflight", "confirmation", "observation_age"))
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "attempts" and args.phase == "preflight":
        raise SystemExit("attempts need --phase confirmation or observation_age")
    launch_guard.check(args.phase, schedule_path=args.schedule if args.mode == "attempts" else None,
                       model="smolvla" if args.mode == "attempts" else None)
    (preflight if args.mode == "preflight" else attempts)(args)


if __name__ == "__main__":
    main()
