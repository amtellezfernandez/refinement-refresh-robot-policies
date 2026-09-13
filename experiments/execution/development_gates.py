#!/usr/bin/env python3
"""Eight real chunk calls and a zero-action LIBERO development control.

These are pipeline checks on Spatial task 0, initial state 0, not a policy-quality
estimate. No reserved evaluation task/state is touched. cProfile counts Python
method calls in pinned original source; they are not GPU FLOPs or kernel counts.
"""
from __future__ import annotations

import argparse
import copy
import cProfile
import inspect
import json
import math
import os
from pathlib import Path
import pstats
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paired import (
    CHECKPOINT_REVISION, EXPECTED_REVISION, Counters, array_record,
    batch_one_success, file_sha, json_line, keyed_seed, load_policy_and_processors,
    make_noise, observation_sha, prepare_observation, require, seed_environment,
    sha, verify_source_tree,
)

RTOL, ATOL = 1e-5, 1e-6
IDENTITY = {"suite": "libero_spatial", "task_id": 0,
            "initial_state_id": 0, "policy_seed": 1000}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n",
                          encoding="utf-8")


def compare_arrays(reference, candidate, rtol=RTOL, atol=ATOL):
    import numpy as np
    left, right = np.asarray(reference), np.asarray(candidate)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(),
            "comparison requires finite arrays with identical shapes")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"exact_equal": bool(np.array_equal(left, right)),
            "allclose": bool(np.allclose(left, right, rtol=rtol, atol=atol)),
            "max_abs_diff": float(delta.max()), "mean_abs_diff": float(delta.mean()),
            "rtol": rtol, "atol": atol}


def profiler_counts(profiler, methods):
    """Match original method code identity, not ambiguous function-name strings."""
    stats = pstats.Stats(profiler).stats
    result = {}
    for label, method in methods.items():
        code = inspect.unwrap(method).__code__
        key = (code.co_filename, code.co_firstlineno, code.co_name)
        primitive, total, own_s, cumulative_s, _ = stats.get(key, (0, 0, 0.0, 0.0, {}))
        result[label] = {"primitive_calls": primitive, "total_calls": total,
                         "source_file": code.co_filename, "first_line": code.co_firstlineno,
                         "function": code.co_name}
    return result


def image_variants(observation):
    import numpy as np
    require(isinstance(observation.get("pixels"), dict), "raw pixels dictionary required")
    keys = sorted(observation["pixels"])
    require(len(keys) == 2, "exactly two real camera inputs required")
    a, b = (np.asarray(observation["pixels"][key]) for key in keys)
    require(a.shape == b.shape and a.dtype == b.dtype, "camera swap requires matching arrays")
    require(not np.array_equal(a, b), "camera images unexpectedly identical")
    swap, zero = copy.deepcopy(observation), copy.deepcopy(observation)
    swap["pixels"][keys[0]], swap["pixels"][keys[1]] = b.copy(), a.copy()
    for key in keys:
        zero["pixels"][key] = np.zeros_like(observation["pixels"][key])
    return {"native": copy.deepcopy(observation), "camera_swap": swap, "zero_images": zero}, keys


def run(args):
    started = time.perf_counter()
    state = {"status": "running", "issued_chunk_calls": 0, "completed_chunk_calls": 0,
             "active_call": None, "zero_action_steps": 0}
    env, counters = None, None
    write_json(args.output / "protocol.json", {
        "schema_version": 1, "kind": "development_pipeline_gates", "identity": IDENTITY,
        "planned_chunk_calls": 8, "max_seconds": args.max_seconds,
        "image_size": args.image_size, "rtol": RTOL, "atol": ATOL,
        "checkpoint_revision": CHECKPOINT_REVISION, "lerobot_revision": EXPECTED_REVISION,
        "call_order": ["native", "native_repeat", "camera_swap", "zero_images",
                       "profile_d5", "counter_d5", "profile_d10", "counter_d10"],
        "decision_rule": "Native repeat and profiler/counter outputs must be allclose; at least one image perturbation must change the full output beyond tolerance. Both independent source profiles must count one sample_actions and d denoise_step calls. Zero-action control must not succeed and must reach the task horizon or an actual terminal condition.",
        "timing_scope": "Budget includes local model load and environment setup, checked between operations; an in-progress CUDA or simulator operation is not forcibly killed.",
        "limitations": "One development input. Perturbation sensitivity does not establish useful visual grounding or policy quality. No accuracy, power, FLOP or safety claim."})

    def budget():
        require(time.perf_counter() - started <= args.max_seconds,
                f"development wall budget {args.max_seconds}s exceeded between operations")

    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy as np
        import torch
        from lerobot.envs import make_env, make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.utils import NEW_ROLLOUT_OPTION
        require(torch.cuda.is_available(), "CUDA GPU required")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        seed_environment(0)
        policy, pre, post, native = load_policy_and_processors(args.checkpoint)
        budget()
        cfg = LiberoEnv(task=IDENTITY["suite"], task_ids=[IDENTITY["task_id"]],
                        init_states=True, hard_reset=True, control_mode="relative",
                        observation_height=args.image_size, observation_width=args.image_size)
        env = make_env(cfg, n_envs=1, use_async_envs=False)[IDENTITY["suite"]][IDENTITY["task_id"]]
        env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
        base = env.envs[0].unwrapped
        require(base.hard_reset and base.init_states, "explicit hard reset required")
        require(0 <= IDENTITY["initial_state_id"] < len(base._init_states), "state out of range")
        environment_seed = keyed_seed([IDENTITY["suite"], IDENTITY["task_id"], 0], domain="environment")

        def reset():
            budget()
            base.init_state_id = IDENTITY["initial_state_id"]
            seed_environment(environment_seed)
            for processor in (pre, post, env_pre, env_post):
                if hasattr(processor, "reset"):
                    processor.reset()
            result, _ = env.reset(seed=[environment_seed], options={NEW_ROLLOUT_OPTION: True})
            return result, array_record(base._env.sim.get_state().flatten())

        observation, initial = reset()
        obs_hash = observation_sha(observation)
        variants, camera_keys = image_variants(observation)
        batches = {name: prepare_observation(value, env, env_pre, pre)
                   for name, value in variants.items()}
        noise = make_noise(policy, IDENTITY, 0)
        # Preserve actual arrays, including the original pixels, before any calls.
        torch.save({"raw_observations": variants,
                    "prepared_observations": {name: {key: value.detach().cpu() if isinstance(value, torch.Tensor)
                                                     else copy.deepcopy(value) for key, value in batch.items()}
                                               for name, batch in batches.items()},
                    "noise": noise.detach().cpu()}, args.output / "inputs.pt")
        manifest = {"schema_version": 1, "native_checkpoint_config": native,
                    "checkpoint_revision": CHECKPOINT_REVISION,
                    "checkpoint_files_sha256": {str(path.relative_to(args.checkpoint)): file_sha(path)
                                                 for path in args.checkpoint.rglob("*")
                                                 if path.is_file() and ".cache" not in path.parts},
                    "verified_pinned_source_sha256": verify_source_tree(),
                    "runner_sha256": file_sha(__file__), "run_paired_sha256": file_sha(Path(__file__).with_name("run_paired.py")),
                    "inputs_sha256": file_sha(args.output / "inputs.pt"),
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "native_config": native,
                    "precision": {"matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                                  "deterministic_algorithms": True, "autocast_enabled": policy.config.use_amp},
                    "initial_state": initial, "initial_observation_sha256": obs_hash,
                    "environment_seed": environment_seed, "camera_keys": camera_keys,
                    "variant_observation_sha256": {name: observation_sha(value) for name, value in variants.items()},
                    "noise": array_record(noise.detach().cpu().numpy()),
                    "task_horizon": int(base._max_episode_steps), "MUJOCO_GL": os.environ["MUJOCO_GL"]}
        write_json(args.output / "manifest.json", manifest)
        outputs, profiles, counted, calls = {}, {}, {}, []
        originals = {"sample_actions": policy.model.sample_actions,
                     "denoise_step": policy.model.denoise_step}

        with (args.output / "calls.jsonl").open("x", encoding="utf-8") as stream:
            for label, variant, d, mode in [
                ("native", "native", 10, "plain"),
                ("native_repeat", "native", 10, "plain"),
                ("camera_swap", "camera_swap", 10, "plain"),
                ("zero_images", "zero_images", 10, "plain"),
                ("profile_d5", "native", 5, "profile"),
                ("counter_d5", "native", 5, "counter"),
                ("profile_d10", "native", 10, "profile"),
                ("counter_d10", "native", 10, "counter"),
            ]:
                budget()
                state["active_call"] = label
                policy.config.num_steps = d
                policy.reset()
                profiler = cProfile.Profile() if mode == "profile" else None
                counters = Counters(policy) if mode == "counter" else None
                if profiler:
                    require(all(getattr(policy.model, name) == method for name, method in originals.items()),
                            "profiler requires original unwrapped methods")
                torch.cuda.synchronize()
                call_started = time.perf_counter()
                state["issued_chunk_calls"] += 1
                try:
                    if profiler:
                        profiler.enable()
                    from contextlib import nullcontext
                    with torch.inference_mode(), (torch.autocast(device_type="cuda") if policy.config.use_amp else nullcontext()):
                        output = policy.predict_action_chunk(copy.deepcopy(batches[variant]), noise=noise.clone())
                    torch.cuda.synchronize()
                finally:
                    if profiler:
                        profiler.disable()
                    if counters:
                        counters.restore()
                output_numpy = output.detach().cpu().numpy()
                require(list(output_numpy.shape) == [1, 50, 7], "expected full unpadded action chunk")
                outputs[label] = output_numpy.copy()
                record = {"label": label, "variant": variant, "denoising_steps": d,
                          "instrumentation": mode, "host_wall_s": time.perf_counter() - call_started,
                          "noise_sha256": manifest["noise"]["sha256"],
                          "observation_sha256": manifest["variant_observation_sha256"][variant],
                          "action": array_record(output_numpy)}
                if profiler:
                    profiles[d] = profiler_counts(profiler, originals)
                    record["original_source_profile"] = profiles[d]
                    profiler.dump_stats(str(args.output / f"profile_d{d}.pstats"))
                if counters:
                    counted[d] = counters.counts.copy()
                    record["counter_counts"] = counted[d]
                    counters = None
                calls.append(record)
                json_line(stream, record)
                state["completed_chunk_calls"] += 1
                state["active_call"] = None
                print(f"CALL {len(calls)}/8 {label} complete", flush=True)

        comparisons = {label: compare_arrays(outputs["native"], outputs[label])
                       for label in ("native_repeat", "camera_swap", "zero_images")}
        profile_parity = {str(d): compare_arrays(outputs[f"profile_d{d}"], outputs[f"counter_d{d}"])
                          for d in (5, 10)}
        counts_match = all(profiles[d]["sample_actions"]["total_calls"] == counted[d]["generations"] == 1
                           and profiles[d]["denoise_step"]["total_calls"] == counted[d]["denoise_steps"] == d
                           for d in (5, 10))

        zero_obs, zero_initial = reset()
        require(zero_initial["sha256"] == initial["sha256"] and observation_sha(zero_obs) == obs_hash,
                "zero-action control reset does not reproduce the model input")
        horizon, zero_success, terminated, truncated = int(base._max_episode_steps), False, False, False
        original_step, driver_records = base.step, []

        def driver_step(action):
            result = original_step(action)
            _, reward, term, trunc, info = result
            independent_success = bool(base._env.check_success())
            require(independent_success == bool(info["is_success"]), "base success disagrees with simulator query")
            driver_records.append({"action": array_record(action), "reward": float(reward),
                                   "terminated": bool(term), "truncated": bool(trunc),
                                   "success": independent_success})
            return result

        base.step = driver_step
        try:
            with (args.output / "zero_action_steps.jsonl").open("x", encoding="utf-8") as stream:
                for tick in range(horizon):
                    budget()
                    action = np.zeros((1, 7), dtype=np.float32)
                    _, reward, term, trunc, info = env.step(action)
                    require(len(driver_records) == tick + 1, "zero-action driver step count disagreement")
                    driver = driver_records[-1]
                    success = batch_one_success(info)
                    terminated, truncated = bool(term[0]), bool(trunc[0])
                    require(success == driver["success"] and terminated == driver["terminated"]
                            and float(reward[0]) == driver["reward"], "zero-action independent outcome disagreement")
                    require(array_record(action[0])["sha256"] == driver["action"]["sha256"],
                            "zero-action producer/consumer disagreement")
                    zero_success = zero_success or success
                    state["zero_action_steps"] = tick + 1
                    json_line(stream, {"tick": tick, "vector_success": success,
                                       "vector_terminated": terminated, "vector_truncated": truncated,
                                       "driver": driver})
                    if terminated or truncated:
                        break
        finally:
            base.step = original_step

        checks = {"native_repeat_allclose": comparisons["native_repeat"]["allclose"],
                  "at_least_one_image_perturbation_changes_output": any(not comparisons[key]["allclose"]
                                                                       for key in ("camera_swap", "zero_images")),
                  "independent_profile_counts_match": counts_match,
                  "profile_counter_outputs_allclose": all(value["allclose"] for value in profile_parity.values()),
                  "zero_action_not_successful": not zero_success,
                  "zero_action_horizon_or_terminal": state["zero_action_steps"] == horizon or terminated or truncated,
                  "exactly_eight_chunk_calls": state["completed_chunk_calls"] == state["issued_chunk_calls"] == 8}
        summary = {"schema_version": 1, "status": "passed" if all(checks.values()) else "failed",
                   "checks": checks, "comparisons": comparisons, "profile_counter_parity": profile_parity,
                   "source_profile_counts": profiles, "counter_counts": counted,
                   "zero_action": {"success": zero_success, "steps": state["zero_action_steps"],
                                   "horizon": horizon, "terminated": terminated, "truncated": truncated,
                                   "initial_state_sha256": zero_initial["sha256"]},
                   "issued_chunk_calls": state["issued_chunk_calls"], "completed_chunk_calls": state["completed_chunk_calls"],
                   "wall_s": time.perf_counter() - started, "manifest_sha256": sha(manifest),
                   "claim_scope": "Development pipeline checks; no held-out quality claim, no FLOP estimate."}
        write_json(args.output / "summary.json", summary)
        require(summary["status"] == "passed", f"development gate failure: {checks}")
        print(json.dumps({"status": "passed", "checks": checks, "wall_s": summary["wall_s"]}), flush=True)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(state, status="failed_or_interrupted",
                    error=repr(error), traceback=traceback.format_exc(), wall_s=time.perf_counter() - started))
        raise
    finally:
        if counters is not None:
            counters.restore()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--image-size", type=int, default=360)
    args = parser.parse_args()
    require(math.isfinite(args.max_seconds) and args.max_seconds > 0, "positive finite budget required")
    require(args.image_size > 0, "positive image size required")
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args)


if __name__ == "__main__":
    main()
