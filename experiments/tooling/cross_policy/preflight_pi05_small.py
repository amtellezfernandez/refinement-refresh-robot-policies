#!/usr/bin/env python3
"""Smallest useful pi0.5 GPU preflight, generation half. Frozen in review/GPU_PREFLIGHT_PREREG.md. Python 3.11."""
import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_guard  # noqa: E402
from common import atomic_write_json  # noqa: E402


def summary(values):
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered), "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "min": ordered[0], "max": ordered[-1]}


def sampler(stop, samples):
    while not stop.wait(0.5):
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                          "--format=csv,noheader,nounits"], universal_newlines=True)
        samples.append([float(x) for x in output.strip().split(",")])


def run(args, report):
    # Must precede the first jax import. Keeps the measurement honest: the stock 75% preallocation
    # would report the pool size, not what the model needs.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    import jaxlib
    import numpy as np
    from importlib import metadata
    packages = {}
    for name in ("jax", "jaxlib", "flax", "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12", "nvidia-cuda-runtime-cu12"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    devices = [str(x) for x in jax.local_devices()]
    report["device"] = {"packages": packages, "jax_devices": devices, "backend": jax.default_backend(), "jaxlib_path": str(jaxlib.__file__)}
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX fell back to {}".format(jax.default_backend()))
    x = jax.numpy.ones((512, 512), dtype=jax.numpy.float32)
    # Fails here if cuBLAS and this GPU are incompatible, before the checkpoint loads.
    report["device"]["matmul_finite"] = bool(jax.numpy.isfinite((x @ x).block_until_ready()).all())
    from server_openpi import ALLOWED_D, KeyedPolicy, load, memory_stats
    from openpi.policies.libero_policy import make_libero_example
    begin = time.perf_counter()
    policy = KeyedPolicy(load(args.checkpoint))
    report["load"] = {"seconds": time.perf_counter() - begin, "memory_after": memory_stats()}
    example = make_libero_example()
    report["generation"] = {}
    for d in ALLOWED_D:
        request = dict(example, __d__=d)
        compile_started = time.perf_counter()
        for index in range(3):
            policy.infer(dict(request, __noise_key__=json.dumps(["small_warm", d, index])))
        warm_s = time.perf_counter() - compile_started
        times, digests = [], set()
        for index in range(30):
            result = policy.infer(dict(request, __noise_key__=json.dumps(["small_measure", d, index])))
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise RuntimeError("unsupported_refinement_setting at d={}: {}".format(d, actions.shape))
            times.append(result["host_generation_s"])
            digests.add(hashlib.sha256(actions.tobytes()).hexdigest())
        report["generation"][str(d)] = {"warmup_3_calls_s": warm_s, "host_s": summary(times), "chunk_shape": list(actions.shape),
                                        "distinct_chunks": len(digests), "memory": result["memory"]}
    report["gates"]["G3_model"] = "PASS"
    report["not_measured"] = {"cuda_event": "JAX exposes no CUDA event timer (R5)",
                              "xla_arithmetic": "XLA cost analysis not implemented; no pi0.5 arithmetic count exists (R13)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch_guard.check("preflight")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model": "pi05", "gates": {"G3_model": "NOT_REACHED"}, "started_unix": time.time()}
    stop, samples = threading.Event(), []
    thread = threading.Thread(target=sampler, args=(stop, samples), daemon=True)
    thread.start()
    try:
        run(args, report)
        report["status"] = "completed"
    except BaseException as error:  # A failed run is the record; never replaced.
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        stop.set()
        thread.join()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["resources"] = {"max_rss_gib": usage.ru_maxrss / 2**20, "cpu_user_s": usage.ru_utime, "cpu_sys_s": usage.ru_stime,
                               "gpu_samples_n": len(samples), "gpu_used_mib_max": max((x[0] for x in samples), default=None),
                               "gpu_util_pct_median": statistics.median(x[1] for x in samples) if samples else None}
        report["finished_unix"] = time.time()
        atomic_write_json(args.output / "pi05_small.json", report)


if __name__ == "__main__":
    main()
