#!/usr/bin/env python3
"""Smallest useful SmolVLA GPU preflight. Frozen in review/GPU_PREFLIGHT_PREREG.md. Python 3.12."""
import argparse
import hashlib
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import launch_guard  # noqa: E402
from common import atomic_write_json, file_sha  # noqa: E402

SMOKE_SEED = 790011
CONDITIONS = (("NATIVE", 1, 10), ("R22", 2, 2), ("R25", 2, 5))
ALTERNATE_NOISES = 8


def summary(values):
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered), "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "min": ordered[0], "max": ordered[-1]}


def distances(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return {"l2": float(np.linalg.norm(a - b)), "max_abs": float(np.abs(a - b).max())}


def load_reference(path):
    """Allow-listed load of Alba's recorded inputs. Same list as the earlier pixel comparison."""
    import numpy as np
    import numpy._core.multiarray as multiarray
    import numpy.dtypes as dtypes
    import torch
    allowed = [multiarray._reconstruct, np.ndarray, np.dtype] + [
        getattr(dtypes, name) for name in ("UInt8DType", "Float64DType", "Float32DType", "Int64DType", "BoolDType")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, weights_only=True, map_location="cpu")


def action_comparison(args, policy, pre, post, report):
    import copy
    import numpy as np
    import run_paired as rp
    import torch
    from development_gates import IDENTITY
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from runner_smolvla import make_task_env
    import json
    manifest = json.loads(args.reference_manifest.read_text("utf-8"))
    actual = file_sha(args.reference_inputs)
    if actual != manifest["inputs_sha256"]:
        raise RuntimeError("reference inputs hash mismatch: {}".format(actual))
    reference = load_reference(args.reference_inputs)
    env, env_pre, env_post = make_task_env(IDENTITY["suite"], IDENTITY["task_id"], policy)
    base = env.envs[0].unwrapped
    base.init_state_id = IDENTITY["initial_state_id"]
    seed = rp.keyed_seed([IDENTITY["suite"], IDENTITY["task_id"], 0], domain="environment")
    rp.seed_environment(seed)
    for processor in (pre, post, env_pre, env_post):
        if hasattr(processor, "reset"):
            processor.reset()
    rendered, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    state = rp.array_record(base._env.sim.get_state().flatten())
    batches = {"egl": rp.prepare_observation(reference["raw_observations"]["native"], env, env_pre, pre),
               "osmesa": rp.prepare_observation(rendered, env, env_pre, pre),
               "camera_swap": rp.prepare_observation(reference["raw_observations"]["camera_swap"], env, env_pre, pre)}
    noise = reference["noise"].to(policy.config.device)

    def chunk(name, noise_tensor, d):
        policy.reset()
        policy.config.num_steps = d
        with torch.inference_mode():
            output = policy._get_action_chunk(copy.deepcopy(batches[name]), noise=noise_tensor)
            environment_units = post(output[:, 0, :].clone())
        return output.detach().cpu().numpy()[0], environment_units.detach().cpu().numpy()[0]

    def compare(left, right):
        return {"first_action_model_units": distances(left[0][0], right[0][0]),
                "first_action_environment_units": distances(left[1], right[1]),
                "chunk_mean_l2_model_units": float(np.linalg.norm(left[0] - right[0], axis=-1).mean())}

    result = {"scope": "ONE state, SmolVLA only; descriptive; no parity threshold exists (R9)",
              "state_sha256": state["sha256"], "reference_state_sha256": manifest["initial_state"]["sha256"],
              "state_identical_to_reference": state["sha256"] == manifest["initial_state"]["sha256"],
              "reference_gl": manifest.get("MUJOCO_GL"), "rendered_gl": os.environ.get("MUJOCO_GL"),
              "reference_gpu": manifest.get("gpu"), "reference_torch": manifest.get("torch_version"), "by_d": {}}
    for d in (2, 5, 10):
        egl = chunk("egl", noise, d)
        others = [compare(egl, chunk("egl", rp.make_noise(policy, {"alternate": d}, index, "preflight"), d))
                  for index in range(ALTERNATE_NOISES)]
        osmesa = compare(egl, chunk("osmesa", noise, d))
        natural = [x["first_action_environment_units"]["l2"] for x in others]
        value = osmesa["first_action_environment_units"]["l2"]
        result["by_d"][str(d)] = {
            "egl_chunk_sha256": hashlib.sha256(egl[0].tobytes()).hexdigest(),
            "numerical_floor_egl_repeat": compare(egl, chunk("egl", noise, d)),
            "osmesa_vs_egl_same_noise": osmesa, "egl_other_noises": others,
            "positive_control_camera_swap": compare(egl, chunk("camera_swap", noise, d)),
            "reading": "BELOW_NOISE_RANGE" if value < min(natural) else "ABOVE_NOISE_RANGE" if value > max(natural) else "WITHIN_NOISE_RANGE"}
    env.close()
    report["action_comparison"] = result


def run(args, report):
    import torch
    import run_paired as rp
    from runner_smolvla import gpu_sampler, make_task_env, rollout
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable to torch")
    report["device"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "name": torch.cuda.get_device_name(0)}
    stop, samples = threading.Event(), []
    sampler = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    try:
        begin = time.perf_counter()
        policy, pre, post, native = rp.load_policy_and_processors(args.checkpoint)
        report["load"] = {"seconds": time.perf_counter() - begin, "native": native}
        env, env_pre, env_post = make_task_env("libero_spatial", 0, policy)
        base = env.envs[0].unwrapped
        base.init_state_id = 0
        observation, _ = env.reset(seed=[rp.keyed_seed(["small_preflight"], domain="environment")])
        batch = rp.prepare_observation(observation, env, env_pre, pre)
        report["generation"] = {}
        for d in (2, 5, 10):
            policy.config.num_steps = d
            with torch.inference_mode():
                for index in range(10):
                    policy.reset()
                    policy._get_action_chunk(rp.copy.deepcopy(batch), noise=rp.make_noise(policy, {"small": d}, index, "warmup"))
                torch.cuda.reset_peak_memory_stats()
                times = []
                for index in range(30):
                    policy.reset()
                    noise = rp.make_noise(policy, {"small": d}, index, "measure")
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    chunk = policy._get_action_chunk(rp.copy.deepcopy(batch), noise=noise)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - started)
                    if not torch.isfinite(chunk).all():
                        raise RuntimeError("non-finite chunk at d={}".format(d))
            report["generation"][str(d)] = {"host_s": summary(times), "chunk_shape": list(chunk.shape),
                                            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        state = base._init_states[0]
        report["gates"]["G3_model"] = "PASS"
        report["evaluator"] = {}
        for name, h, d in CONDITIONS:
            row = {"h": h, "d": d, "episode_identity": {"suite": "libero_spatial", "task_id": 0,
                                                        "state_slot": "official0", "policy_seed": SMOKE_SEED}}
            result = rollout(row, env, policy, pre, post, env_pre, env_post, state)
            report["evaluator"][name] = {"h": h, "d": d, "smoke_outcome_not_evidence": result["outcome"],
                                         "action_shape": list(result["actions"].shape), "wall_s": result["timing"]["rollout_wall_s"],
                                         "query_host_s": summary(result["timing"]["query_host_s"]), "memory": result["memory"]}
        env.close()
        report["gates"]["G4_evaluator"] = "PASS"
        action_comparison(args, policy, pre, post, report)
    finally:
        stop.set()
        sampler.join()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["resources"] = {"max_rss_gib": usage.ru_maxrss / 2**20, "cpu_user_s": usage.ru_utime, "cpu_sys_s": usage.ru_stime,
                               "gpu_samples_n": len(samples),
                               "gpu_used_mib_max": max((x[1] for x in samples), default=None),
                               "gpu_util_pct_median": statistics.median(x[2] for x in samples) if samples else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch_guard.check("preflight")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"model": "smolvla", "gates": {"G3_model": "NOT_REACHED", "G4_evaluator": "NOT_REACHED"}, "started_unix": time.time()}
    try:
        run(args, report)
        report["status"] = "completed"
    except BaseException as error:  # A failed run is the record; never replaced.
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["finished_unix"] = time.time()
        atomic_write_json(args.output / "smolvla_small.json", report)


if __name__ == "__main__":
    main()
