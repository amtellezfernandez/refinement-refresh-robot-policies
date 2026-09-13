"""Fail-closed, batch-one synchronous LIBERO instrumentation (ledger v2).

CUDA events measure stream elapsed time, not utilization/energy. Step hashes
prove replay only when compared across repeated runs of the SAME condition.
"""
import hashlib
import json
import random
import dataclasses
import importlib.metadata
import sys
import os
import copy
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def array_record(value):
    a = np.ascontiguousarray(value)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite state/action")
    header = json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":"))
    return {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(),
            "sha256": hashlib.sha256(header.encode() + a.tobytes()).hexdigest()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed):
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        seed_all(seed)
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


class Ledger:
    def __init__(self, policy, output, suite, warmup=20, config=None):
        if warmup < 1:
            raise ValueError("At least one warm-up pass required")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        self.policy, self.suite, self.warmup = policy, suite, warmup
        self.phase, self.episode, self.tick = "measured", None, 0
        self.warmed = False
        self.reference_check = os.environ.get("ROBOPAD_REFERENCE_CHECK") == "1"
        if config is None or config.env.max_parallel_tasks != 1:
            raise ValueError("Full config and sequential tasks required")
        checkpoint = Path(config.policy.pretrained_path)
        if not checkpoint.is_dir():
            raise ValueError("Local pinned checkpoint required")
        def file_hash(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        import lerobot.envs.libero as libero_module
        import lerobot.policies.smolvla.modeling_smolvla as policy_module
        cfg = json.loads(json.dumps(dataclasses.asdict(config), default=str))
        for key in ("output_dir", "job_name"):
            cfg.pop(key, None)
        contract = {"config": cfg,
                    "checkpoint": {str(p.relative_to(checkpoint)): file_hash(p)
                                   for p in sorted(checkpoint.rglob("*")) if p.is_file()
                                   and ".cache" not in p.relative_to(checkpoint).parts},
                    "software": {name: importlib.metadata.version(name) for name in
                                 ("torch", "numpy", "mujoco", "robosuite", "transformers")},
                    "source": {"ledger": file_hash(__file__), "evaluator": file_hash(sys.argv[0]), "libero": file_hash(libero_module.__file__),
                               "policy": file_hash(policy_module.__file__)}}
        contract["reference_check"] = self.reference_check
        self.data = {"schema_version": 3, "suite": suite, "contract": contract, "complete": False,
                     "reference_steps": 0, "reference_sampler_calls": 0,
                     "h": policy.config.n_action_steps, "denoising_steps": policy.config.num_steps,
                     "warmup_passes": warmup, "episodes": [], "inferences": [],
                     "sampler_calls": {"warmup": 0, "measured": 0},
                     "chunk_calls": {"warmup": 0, "measured": 0}}
        Path(output).mkdir(parents=True, exist_ok=True)
        self.journal = (Path(output) / "ledger_events.jsonl").open("x", buffering=1)
        original_chunk = policy._get_action_chunk
        original_sample = policy.model.sample_actions
        original_select = policy.select_action

        def sample(*args, **kwargs):
            self.data["sampler_calls"][self.phase] += 1
            return original_sample(*args, **kwargs)

        def chunk(*args, **kwargs):
            self.data["chunk_calls"][self.phase] += 1
            device = next(policy.parameters()).device
            if device.type != "cuda":
                raise RuntimeError("CUDA timing requires a CUDA policy")
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)  # exclude pending preprocessing
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                result = original_chunk(*args, **kwargs)
                end.record()
                end.synchronize()
                event = {"phase": self.phase, "episode": self.episode["identity"],
                         "step": self.tick, "cuda_ms": start.elapsed_time(end)}
            self.data["inferences"].append(event)
            self.emit("inference", event)
            return result

        def select(batch, *args, **kwargs):
            if not self.warmed:
                self.phase = "warmup"
                with isolated_rng(self.episode["seeds"]["torch"]):
                    for _ in range(self.warmup):
                        policy.reset()
                        original_select(dict(batch), *args, **kwargs)
                policy.reset()
                self.phase, self.warmed = "measured", True
            # Tick-keyed policy RNG avoids condition-dependent call-count drift.
            key = json.dumps([self.episode["identity"], self.tick], sort_keys=True)
            noise_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
            before = copy.deepcopy(policy._queues) if self.reference_check else None
            with isolated_rng(noise_seed):
                result = original_select(batch, *args, **kwargs)
            if self.reference_check:
                after = policy._queues
                policy._queues = before
                policy._get_action_chunk = original_chunk
                def reference_sample(*a, **kw):
                    self.data["reference_sampler_calls"] += 1
                    return original_sample(*a, **kw)
                policy.model.sample_actions = reference_sample
                try:
                    with isolated_rng(noise_seed):
                        reference = original_select(dict(batch), *args, **kwargs)
                    if not torch.equal(result, reference):
                        raise ValueError("Instrumented/reference action mismatch")
                    self.data["reference_steps"] += 1
                finally:
                    policy._queues = after
                    policy._get_action_chunk = chunk
                    policy.model.sample_actions = sample
            self.episode["select_calls"] += 1
            return result

        policy.model.sample_actions = sample
        policy._get_action_chunk = chunk
        policy.select_action = select

    def emit(self, kind, event):
        self.journal.write(json.dumps({"kind": kind, "event": event}) + "\n")

    def before_episode(self, env, seeds):
        if env.num_envs != 1 or not hasattr(env, "envs") or seeds is None or len(seeds) != 1:
            raise ValueError("Ledger requires seeded synchronous batch-one evaluation")
        self.pending_seed = int(seeds[0])
        seed_all(self.pending_seed)  # before LIBERO reset, not merely once per job
        base = env.envs[0].unwrapped
        if not base.init_states:
            raise ValueError("Predefined LIBERO initial states required")
        self.pending_state_id = base.init_state_id % len(base._init_states)
        if not hasattr(env, "_robopad_original_step"):
            env._robopad_original_step = env.step

            def step(action):
                record = {"step": self.tick, "action": array_record(action)}
                result = env._robopad_original_step(action)
                _, reward, terminated, truncated, info = result
                final = info.get("final_info", info)
                if not isinstance(final, dict):
                    final = final[0] if final[0] is not None else {}
                success = bool(np.asarray(final.get("is_success", False)).reshape(-1)[0])
                record.update(reward=float(np.asarray(reward).reshape(-1)[0]),
                              terminated=bool(np.asarray(terminated).reshape(-1)[0]),
                              truncated=bool(np.asarray(truncated).reshape(-1)[0]), success=success)
                self.episode["environment_steps"].append(record)
                self.emit("environment_step", {"identity": self.episode["identity"], **record})
                self.tick += 1
                if record["terminated"] or record["truncated"] or self.tick == self.episode["max_steps"]:
                    self.episode["complete"] = True
                    self.episode["success"] = any(x["success"] for x in self.episode["environment_steps"])
                    self.emit("episode_complete", {"identity": self.episode["identity"],
                                                  "steps": self.tick, "success": self.episode["success"]})
                return result

            env.step = step

    def after_episode_reset(self, env):
        base = env.envs[0].unwrapped
        initial = array_record(np.asarray(base._env.sim.get_state().flatten()))
        seed = self.pending_seed
        self.episode = {"identity": [self.suite, int(base.task_id), int(self.pending_state_id), seed],
                        "seeds": {k: seed for k in ("python", "numpy", "torch", "libero")},
                        "initial_state": initial, "select_calls": 0, "complete": False,
                        "max_steps": int(env.call("_max_episode_steps")[0]),
                        "producer_steps": [], "environment_steps": []}
        self.tick = 0
        self.data["episodes"].append(self.episode)
        self.emit("episode", self.episode)

    def record_action(self, action):
        record = {"step": self.tick, "action": array_record(action)}
        self.episode["producer_steps"].append(record)
        self.emit("producer_step", {"identity": self.episode["identity"], **record})

    def finalize(self, info):
        if not all(e["complete"] for e in self.data["episodes"]):
            raise ValueError("Cannot finalize incomplete episodes")
        self.data["complete"] = True
        from validate_ledger import validate
        validate({**info, "robopad_ledger": self.data})
        self.emit("run_complete", {"episodes": len(self.data["episodes"])})
        self.journal.close()
        return self.data
