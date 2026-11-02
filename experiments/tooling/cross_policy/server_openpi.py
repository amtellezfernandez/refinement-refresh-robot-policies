#!/usr/bin/env python3
"""OpenPI policy server with per-request d and keyed noise. Python 3.11. NOT YET EXECUTED ON A GPU.

Uses the official create_trained_policy and WebsocketPolicyServer unchanged. One thin server-side
wrapper is needed, and Alba must review it (audit R4), because the stock server calls
policy.infer(obs) with no noise argument and draws from one process-wide key. That cannot pair
noise by identity and tick. The wrapper only passes num_steps and noise to documented parameters
of Policy.infer and sample_actions. It adds no skipping, truncation, interpolation or action reuse.

    profile mode: 100 warm-up + 100 measured chunk generations at d=2, 5, 10 on a fixed input
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_guard  # noqa: E402
from common import OPENPI_NORM_SHA256, atomic_write_json, file_sha, keyed_int  # noqa: E402

NATIVE_D = 10
ALLOWED_D = (2, 5, NATIVE_D)
ACTION_HORIZON, ACTION_DIM = 10, 32
CUDA_EVENT_UNAVAILABLE = ("JAX exposes no CUDA event timer. Host time spans a blocking conversion to NumPy. "
                          "Not substituted; Alba's specification asks for this field, see audit R5.")


def noise_for(key):
    import numpy as np
    generator = np.random.default_rng(keyed_int("pi05_noise", key))
    return generator.standard_normal((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)


def memory_stats():
    import jax
    stats = jax.local_devices()[0].memory_stats() or {}
    return {k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}


class KeyedPolicy:
    """BasePolicy wrapper. Requests carry only __d__ and __noise_key__, and no model input changes."""
    def __init__(self, policy):
        self._policy = policy
        self.metadata = policy.metadata
        self._received = None
        sampler = policy._sample_actions

        def recording_sampler(rng, observation, **kwargs):
            # What actually reaches model.sample_actions, as distinct from what was asked for.
            self._received = kwargs.get("num_steps")
            return sampler(rng, observation, **kwargs)
        policy._sample_actions = recording_sampler

    def infer(self, obs):
        obs = dict(obs)
        d, key = int(obs.pop("__d__")), json.loads(obs.pop("__noise_key__"))
        if d not in ALLOWED_D:
            raise ValueError("unsupported_refinement_setting: d={}".format(d))
        self._policy._sample_kwargs = {"num_steps": d}
        self._received = None
        started = time.perf_counter()
        result = self._policy.infer(obs, noise=noise_for(key))
        result["host_generation_s"] = time.perf_counter() - started
        result["memory"] = memory_stats()
        result["received_num_steps"] = self._received
        return result

    def reset(self):
        return None


def verify_loaded_normalization(policy, norm_path):
    """The hash covers the file. This covers what the loaded policy actually applies to its actions."""
    import numpy as np
    expected = json.loads(Path(norm_path).read_text("utf-8"))["norm_stats"]
    found = [t for t in policy._output_transform.transforms if type(t).__name__ == "Unnormalize"]
    if len(found) != 1:
        raise SystemExit("expected one Unnormalize transform, found {}".format(len(found)))
    if not found[0].use_quantiles:
        raise SystemExit("policy does not use quantile normalization")
    for group, values in expected.items():
        for field, numbers in values.items():
            if not np.array_equal(np.asarray(getattr(found[0].norm_stats[group], field), dtype=np.float64),
                                  np.asarray(numbers, dtype=np.float64)):
                raise SystemExit("loaded normalization differs at {}.{}".format(group, field))
    return {"use_quantiles": True, "groups": sorted(expected)}


def load(checkpoint):
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    norm = Path(checkpoint) / "assets/physical-intelligence/libero/norm_stats.json"
    actual = file_sha(norm)
    if actual != OPENPI_NORM_SHA256:
        raise SystemExit("norm_stats.json hash mismatch: {}".format(actual))
    policy = policy_config.create_trained_policy(openpi_config.get_config("pi05_libero"), str(checkpoint))
    policy.normalization_loaded = verify_loaded_normalization(policy, norm)
    return policy


def write_provenance(args, normalization_loaded=None):
    import provenance as pv
    norm = Path(args.checkpoint) / "assets/physical-intelligence/libero/norm_stats.json"
    args.output.mkdir(parents=True, exist_ok=False)
    record = pv.capture("pi05_server", args, {"checkpoint_dir": str(args.checkpoint), "normalization_sha256": file_sha(norm),
                                              "normalization_loaded": normalization_loaded})
    atomic_write_json(args.output / "provenance.json", record)


def profile(args):
    import jax
    import numpy as np
    from openpi.policies.libero_policy import make_libero_example
    loaded = load(args.checkpoint)
    write_provenance(args, normalization_loaded=loaded.normalization_loaded)
    policy = KeyedPolicy(loaded)
    example = make_libero_example()
    report = {"cuda_event": CUDA_EVENT_UNAVAILABLE, "jax_devices": [str(x) for x in jax.local_devices()], "points": {}}
    for d in ALLOWED_D:
        request = dict(example, __d__=d)
        for index in range(100):
            warm = policy.infer(dict(request, __noise_key__=json.dumps(["warmup", d, index])))
        host, chunks, received = [], set(), []
        # Same input and noise at every d: the chunks must differ if d reaches the integrator.
        fixed = np.asarray(policy.infer(dict(request, __noise_key__=json.dumps(["fixed-noise"])))["actions"])
        for index in range(100):
            result = policy.infer(dict(request, __noise_key__=json.dumps(["measure", d, index])))
            received.append(result["received_num_steps"])
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                report["status"] = "unsupported_refinement_setting"
                report["failure"] = {"d": d, "shape": list(actions.shape), "finite": bool(np.isfinite(actions).all())}
                atomic_write_json(args.output / "profile.json", report)
                raise SystemExit("unsupported_refinement_setting at d={}".format(d))
            host.append(result["host_generation_s"])
            chunks.add(hashlib.sha256(actions.tobytes()).hexdigest())
        report["points"][str(d)] = {"host_s": host, "chunk_shape": list(actions.shape), "memory": result["memory"],
                                    "distinct_chunks": len(chunks), "warm_shape": list(np.asarray(warm["actions"]).shape),
                                    "received_num_steps": received,
                                    "fixed_noise_chunk_sha256": hashlib.sha256(fixed.tobytes()).hexdigest()}
    report["status"] = "completed"
    atomic_write_json(args.output / "profile.json", report)


def serve(args):
    from openpi.serving import websocket_policy_server
    loaded = load(args.checkpoint)
    if args.output is not None:
        write_provenance(args, normalization_loaded=loaded.normalization_loaded)
    policy = KeyedPolicy(loaded)
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port,
                                                  metadata=policy.metadata).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("serve", "profile"))
    parser.add_argument("--phase", default="preflight", choices=("preflight",))  # The client enforces the release.
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    launch_guard.check(args.phase)
    if args.mode == "profile":
        profile(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
