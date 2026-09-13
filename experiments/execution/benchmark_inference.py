#!/usr/bin/env python3
"""Paired, fixed-LIBERO-observation SmolVLA inference microbenchmark.

Example (inside the pinned CUDA/LeRobot environment)::

    python benchmark_inference.py --output-dir /opt/robopad/results/inference_full

Default allocation: 5 randomized blocks x 2 solver settings x 2 instrumentation
modes x (20 warm-up + 100 measured calls) = 2400 *actual* chunk generations.
There is no execution-horizon factor: this measures complete chunk prediction,
not rollouts or action dispatch. The same explicit Gaussian noise is reconstructed
for each on/off pair. No generated or randomly synthesized image is used.

Both modes retain common outer timing and output collection. 'On' additionally
installs model-entry/NFE/vision/prefill/expert probes; 'off' removes those probes.
This estimates the overhead of THESE probes, not the full rollout evaluator.
"""

from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
LEROBOT_REVISION = "b6ec0060779550c0a157ae34feb89e0cf86012a8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing files are never overwritten.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/opt/robopad/checkpoints/smolvla_libero"))
    parser.add_argument("--lerobot-source", type=Path, default=Path("/opt/robopad/src/lerobot"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--warmup-calls", type=int, default=20, help="Per cell in every block; saved separately.")
    parser.add_argument("--calls", type=int, default=100, help="Measured calls per cell in every block.")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--max-total-calls", type=int, default=2400, help="Reject a larger allocation before loading anything.")
    parser.add_argument("--max-seconds", type=float, default=3600, help="Overall wall budget, checked between calls.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--order-seed", type=int, default=271828)
    parser.add_argument("--environment-seed", type=int, default=1000)
    parser.add_argument("--suite", default="libero_spatial", choices=["libero_spatial"])
    parser.add_argument("--task-id", type=int, default=0, choices=[0], help="Development observation only; no held-out tasks.")
    parser.add_argument("--init-state-id", type=int, default=0, choices=[0])
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    if min(args.calls, args.blocks, args.max_total_calls, args.image_size) < 1 or args.warmup_calls < 0:
        parser.error("Positive counts/image size and nonnegative warm-up are required.")
    if args.num_steps != sorted(set(args.num_steps)) or any(step not in (5, 10) for step in args.num_steps):
        parser.error("--num-steps must be a sorted, unique subset of 5 10.")
    if (not all(math.isfinite(value) for value in (args.max_seconds, args.rtol, args.atol))
            or args.max_seconds <= 0 or min(args.rtol, args.atol) < 0):
        parser.error("Wall budget must be finite and positive; tolerances finite and nonnegative.")
    args.planned_calls = args.blocks * len(args.num_steps) * 2 * (args.warmup_calls + args.calls)
    if args.planned_calls > args.max_total_calls:
        parser.error(f"Requested {args.planned_calls} calls exceeds cap {args.max_total_calls}.")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def jsonl_write(stream: Any, value: Any) -> None:
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def tensor_hash(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tree_hashes(tree: Any, prefix: str = "") -> dict[str, Any]:
    import numpy as np
    import torch

    if isinstance(tree, dict):
        result = {}
        for key in sorted(tree):
            result.update(tree_hashes(tree[key], f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(tree, torch.Tensor):
        return {prefix: {"sha256": tensor_hash(tree), "shape": list(tree.shape), "dtype": str(tree.dtype)}}
    if isinstance(tree, np.ndarray):
        return tree_hashes(torch.from_numpy(np.ascontiguousarray(tree)), prefix)
    return {prefix: {"value": tree}}


def tree_digest(tree: Any) -> tuple[str, dict[str, Any]]:
    parts = tree_hashes(tree)
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest(), parts


def clone_batch(batch: dict[str, Any], cpu: bool = False) -> dict[str, Any]:
    import torch

    return {
        key: (value.detach().cpu().clone() if cpu else value.clone())
        if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def noise_seed(base: int, block: int, phase: str, index: int) -> int:
    # Solver effort and on/off mode deliberately excluded: same inputs in each pair.
    key = f"robopad-fixed-observation-v1:{base}:{block}:{phase}:{index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63 - 1)


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": ordered[0], "max": ordered[-1],
            "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]}


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Record actual resolved tokenizer contents without claiming a missing revision."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None else None
    vocab_json = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    return {"class": type(tokenizer).__name__, "name_or_path": tokenizer.name_or_path,
            "resolved_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "vocabulary_sha256": hashlib.sha256(vocab_json.encode()).hexdigest(),
            "backend_sha256": hashlib.sha256(backend_json.encode()).hexdigest() if backend_json else None,
            "special_tokens_map": tokenizer.special_tokens_map,
            "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side}


class ModelProbes:
    """Temporary probes; off-mode has no model wrappers or stage CUDA events.

    Stage intervals nest: vision_encoder lies within embed_prefix. Do not sum
    all stage intervals into total runtime. NFE counts actual denoise entry.
    """

    def __init__(self, policy: Any):
        self.policy = policy
        self.restores: list[tuple[Any, str, Any, bool]] = []
        self.counts: collections.Counter[str] = collections.Counter()
        self.events: list[tuple[str, Any, Any]] = []

    def reset(self) -> None:
        self.counts.clear()
        self.events.clear()

    def wrap(self, owner: Any, name: str, label: str, timed: bool = False, classifier: Any = None) -> None:
        import torch

        original = getattr(owner, name)
        had_local = name in owner.__dict__

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            event_label = classifier(args, kwargs) if classifier else label
            self.counts[event_label] += 1
            if not timed:
                return original(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.events.append((event_label, start, end))
            return result

        self.restores.append((owner, name, original, had_local))
        setattr(owner, name, wrapped)

    def __enter__(self) -> "ModelProbes":
        model = self.policy.model

        def classify_forward(args: Any, kwargs: Any) -> str:
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                raise RuntimeError("Pinned SmolVLA forward no longer passes named inputs_embeds.")
            if embeds[0] is not None and embeds[1] is None:
                return "prefill_forwards"
            if embeds[0] is None and embeds[1] is not None:
                return "action_expert_forwards"
            raise RuntimeError("Unexpected mixed forward in inference benchmark.")

        try:
            self.wrap(model, "sample_actions", "chunk_entries")
            self.wrap(model, "denoise_step", "denoise_nfe")
            self.wrap(model, "embed_prefix", "embed_prefix", timed=True)
            self.wrap(model.vlm_with_expert.get_vlm_model().vision_model, "forward", "vision_encoder_calls", timed=True)
            self.wrap(model.vlm_with_expert, "forward", "", timed=True, classifier=classify_forward)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for owner, name, original, had_local in reversed(self.restores):
            if had_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.restores.clear()

    def result(self, expected_steps: int) -> tuple[dict[str, int], dict[str, float]]:
        counts = {key: self.counts[key] for key in (
            "chunk_entries", "denoise_nfe", "vision_encoder_calls", "prefill_forwards", "action_expert_forwards")}
        if counts["chunk_entries"] != 1 or counts["denoise_nfe"] != expected_steps:
            raise RuntimeError(f"Actual generation/NFE mismatch: {counts}")
        if counts["prefill_forwards"] != 1 or counts["action_expert_forwards"] != expected_steps or counts["vision_encoder_calls"] < 1:
            raise RuntimeError(f"Actual encoder/prefill/expert mismatch: {counts}")
        label_map = {"vision_encoder_calls": "vision_encoder", "prefill_forwards": "prefill",
                     "action_expert_forwards": "action_expert", "embed_prefix": "embed_prefix"}
        stages: dict[str, float] = collections.defaultdict(float)
        for label, start, end in self.events:
            stages[label_map[label]] += float(start.elapsed_time(end))
        return counts, dict(stages)


def versions(torch: Any, source: Path) -> dict[str, Any]:
    packages = {}
    for package in ("lerobot", "torch", "torchvision", "transformers", "safetensors", "gymnasium", "mujoco", "robosuite", "libero", "numpy", "huggingface-hub"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not_installed_as_distribution"
    try:
        gpu_state = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv"], capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        gpu_state = None
    source_files = ("policies/smolvla/modeling_smolvla.py", "policies/smolvla/smolvlm_with_expert.py",
                    "policies/common/flow_matching.py", "policies/pretrained.py", "envs/libero.py", "envs/utils.py")
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(), "gpu_state_at_start": gpu_state,
            "lerobot_declared_revision": LEROBOT_REVISION,
            "lerobot_source_hashes": {rel: file_sha256(source / "src/lerobot" / rel) for rel in source_files},
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "execution_environment": {key: os.environ.get(key) for key in (
                "MUJOCO_GL", "PYOPENGL_PLATFORM", "HF_HOME", "HF_HUB_OFFLINE",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
            "script_sha256": file_sha256(Path(__file__))}


def acquire_observation(args: Any, policy: Any, checkpoint: Path, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors

    cfg = LiberoEnv(task=args.suite, task_ids=[args.task_id], init_states=True, hard_reset=True,
                    observation_height=args.image_size, observation_width=args.image_size, control_mode="relative")
    env = make_env(cfg, n_envs=1, use_async_envs=False)[args.suite][args.task_id]
    try:
        base = env.envs[0].unwrapped
        if base._init_states is None or not 0 <= args.init_state_id < len(base._init_states):
            raise RuntimeError("Requested real LIBERO initial state is unavailable.")
        initial_hash, _ = tree_digest({"initial_state": base._init_states[args.init_state_id]})
        base.init_state_id = args.init_state_id
        reset_start = time.perf_counter()
        raw, _ = env.reset(seed=[args.environment_seed], options={NEW_ROLLOUT_OPTION: True})
        reset_seconds = time.perf_counter() - reset_start
        if base.init_state_id != args.init_state_id + base._reset_stride:
            raise RuntimeError("LIBERO reset did not consume exactly the requested initial state.")
        description = list(env.call("task_description"))
        raw_hash, raw_parts = tree_digest(raw)
        torch.save(raw, args.output_dir / "raw_libero_observation.pt")
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device},
                                    "rename_observations_processor": {"rename_map": {}}})
        env_preprocessor, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
        torch.cuda.synchronize()
        prep_start = time.perf_counter()
        batch = preprocess_observation(copy.deepcopy(raw))
        batch["task"] = description
        batch = preprocessor(env_preprocessor(batch))
        torch.cuda.synchronize()
        prep_seconds = time.perf_counter() - prep_start
        batch_hash, batch_parts = tree_digest(batch)
        tokenizers = [{"processor_step": type(step).__name__, **tokenizer_identity(step.input_tokenizer)}
                      for step in preprocessor.steps if getattr(step, "input_tokenizer", None) is not None]
        if len(tokenizers) != 1:
            raise RuntimeError(f"Expected one actual policy input tokenizer, found {len(tokenizers)}.")
        torch.save(clone_batch(batch, cpu=True), args.output_dir / "processed_observation.pt")
        return batch, {"suite": args.suite, "task_id": args.task_id, "init_state_id": args.init_state_id,
                       "environment_seed": args.environment_seed, "task_description": description,
                       "image_size": args.image_size, "initial_state_sha256": initial_hash,
                       "raw_observation_sha256": raw_hash, "raw_components": raw_parts,
                       "observation_sha256": batch_hash, "processed_components": batch_parts,
                       "reset_wall_seconds": reset_seconds, "preprocessing_wall_seconds": prep_seconds,
                       "reset_internal_settle_steps": base.num_steps_wait,
                       "policy_input_tokenizers": tokenizers,
                       "policy_dispatched_actions": 0, "environment_config": dataclasses.asdict(cfg)}
    finally:
        env.close()


def run(args: Any) -> int:
    run_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema_version": 1, "status": "initializing", "config": vars(args),
                                "checkpoint_repository": CHECKPOINT, "checkpoint_revision": REVISION,
                                "planned_actual_chunk_generations": args.planned_calls}
    json_write(args.output_dir / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    saved_actions: list[Any] = []
    pending: dict[str, tuple[dict[str, Any], Any]] = {}
    issued_prediction_calls = 0
    active_trial = None
    error = None
    try:
        # These must be set before torch/MuJoCo imports. No automatic CPU fallback.
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        source = args.lerobot_source.resolve()
        sys.path.insert(0, str(source / "src"))
        import numpy as np
        import torch
        import lerobot.policies.smolvla.modeling_smolvla as smolvla_module
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from run_paired import verify_source_tree

        if not Path(smolvla_module.__file__).resolve().is_relative_to(source / "src"):
            raise RuntimeError("Imported LeRobot does not come from the declared pinned source tree.")
        manifest["verified_pinned_source_sha256"] = verify_source_tree()
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required; this benchmark never substitutes CPU or random output.")
        torch.cuda.set_device(torch.device(args.device))
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        checkpoint = args.checkpoint_dir.resolve()
        marker = checkpoint / ".robopad_revision"
        if checkpoint.name == REVISION:
            revision_evidence = "HF snapshot directory name"
        elif marker.is_file() and marker.read_text().strip() == REVISION:
            revision_evidence = "caller-supplied .robopad_revision marker; actual files also hashed"
        else:
            raise RuntimeError(f"Checkpoint must be the pinned snapshot, or contain .robopad_revision with {REVISION}.")
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Real complete checkpoint weights missing: {weights}")
        checkpoint_hashes = {str(path.relative_to(checkpoint)): file_sha256(path)
                             for path in sorted(checkpoint.rglob("*"))
                             if path.is_file() and path.suffix in (".json", ".safetensors") and ".cache" not in path.parts}
        cfg = SmolVLAConfig.from_pretrained(str(checkpoint), local_files_only=True)
        native = {"n_action_steps": cfg.n_action_steps, "num_steps": cfg.num_steps, "chunk_size": cfg.chunk_size}
        if native != {"n_action_steps": 1, "num_steps": 10, "chunk_size": 50}:
            raise RuntimeError(f"Unexpected native checkpoint configuration: {native}")
        if cfg.rtc_config is not None or cfg.n_obs_steps != 1:
            raise RuntimeError("Benchmark requires this checkpoint's memoryless, RTC-disabled inference path.")
        cfg.device, cfg.compile_model, cfg.load_vlm_weights = args.device, False, False
        load_start = time.perf_counter()
        policy = smolvla_module.SmolVLAPolicy.from_pretrained(str(checkpoint), config=cfg, strict=True, local_files_only=True)
        policy.eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_start
        vlm = policy.model.vlm_with_expert
        base_config = vlm.vlm.config.to_dict()
        json_write(args.output_dir / "resolved_vlm_config.json", base_config)
        manifest.update({"strict_checkpoint_loading": True, "checkpoint_load_wall_seconds": load_seconds,
                         "checkpoint_revision_evidence": revision_evidence, "checkpoint_file_hashes": checkpoint_hashes,
                         "native_configuration": native, "runtime": versions(torch, source),
                         "parameter_dtypes": sorted({str(p.dtype) for p in policy.parameters()}),
                         "parameter_count": sum(p.numel() for p in policy.parameters())})
        manifest["vlm_assets"] = {"configured_model_name": cfg.vlm_model_name,
                                  "resolved_config_commit": getattr(vlm.vlm.config, "_commit_hash", None),
                                  "config_artifact": "resolved_vlm_config.json",
                                  "config_sha256": file_sha256(args.output_dir / "resolved_vlm_config.json"),
                                  "model_processor_tokenizer": tokenizer_identity(vlm.processor.tokenizer),
                                  "revision_limit": "Upstream AutoConfig/AutoProcessor/AutoTokenizer resolve their configured model name. Content hashes are recorded; a null resolved commit is unknown, not a pin."}
        random.seed(args.environment_seed)
        np.random.seed(args.environment_seed)
        batch, observation = acquire_observation(args, policy, checkpoint, torch)
        manifest["observation"] = observation
        manifest["clock_boundaries"] = {
            "cuda_elapsed_ms": "Current CUDA stream interval around predict_action_chunk; includes stream idle gaps; not occupancy or energy.",
            "host_elapsed_ms": "perf_counter around predict_action_chunk plus end-event synchronization; pre-call synchronization excluded.",
            "common_exclusions": ["environment reset/settling", "external environment/policy processor and tokenization (once before benchmark)", "batch cloning", "explicit noise generation",
                                  "postprocessing/action unnormalization", "device-to-host action copy", "hashing", "JSONL writing", "stage event resolution"],
            "on_only": "Temporary actual chunk/NFE/vision/transformer probes and stage CUDA event recording inside prediction.",
            "off_mode": "No model probes. Outer host/CUDA timing and peak-memory collection remain common measurement overhead.",
            "internal_preprocessing": "SmolVLAPolicy.prepare_images/prepare_state inside predict_action_chunk remain inside both outer clock intervals.",
            "prefill": "Measured afresh in every on-mode chunk; no cross-call prefix cache. embed_prefix contains vision_encoder and must not be summed with it.",
            "memory": "Per-call peak allocated/reserved bytes after reset_peak_memory_stats; includes loaded model and fixed observation, not incremental-only memory."}
        manifest["output_units"] = "Complete predicted policy-normalized action chunk; not simulator-dispatched/unnormalized actions."
        manifest["noise_contract"] = "Independent CUDA torch.Generator with SHA-256-keyed seed; full FP32 Gaussian noise passed explicitly; on/off absent from key."
        manifest["status"] = "running"
        json_write(args.output_dir / "manifest.json", manifest)
        schedule_rng = random.Random(args.order_seed)
        schedule = []
        for block in range(args.blocks):
            cells = [(steps, mode) for steps in args.num_steps for mode in ("off", "on")]
            schedule_rng.shuffle(cells)
            schedule.extend({"block": block, "cell_order": order, "num_steps": steps, "instrumentation": mode}
                            for order, (steps, mode) in enumerate(cells))
        json_write(args.output_dir / "schedule.json", schedule)
        with (args.output_dir / "trials.jsonl").open("x", encoding="utf-8") as trial_file, (args.output_dir / "pairs.jsonl").open("x", encoding="utf-8") as pair_file:
            for cell in schedule:
                policy.config.num_steps = cell["num_steps"]
                probes = ModelProbes(policy) if cell["instrumentation"] == "on" else None
                if probes:
                    probes.__enter__()
                cell_start = time.perf_counter()
                try:
                    for phase, count in (("warmup", args.warmup_calls), ("measurement", args.calls)):
                        for index in range(count):
                            if time.perf_counter() - run_start > args.max_seconds:
                                raise TimeoutError("Declared wall budget exhausted; partial rows retained, no completed benchmark claim.")
                            prepared = clone_batch(batch)
                            seed = noise_seed(args.seed, cell["block"], phase, index)
                            generator = torch.Generator(device=args.device).manual_seed(seed)
                            noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator, device=args.device, dtype=torch.float32)
                            noise_hash = tensor_hash(noise)
                            policy.reset()
                            if probes:
                                probes.reset()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            allocated_before = torch.cuda.memory_allocated()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            active_trial = {**cell, "phase": phase, "call_index": index, "noise_seed": seed,
                                            "noise_sha256": noise_hash}
                            issued_prediction_calls += 1
                            host_start = time.perf_counter()
                            start.record()
                            with torch.inference_mode():
                                action = policy.predict_action_chunk(prepared, noise=noise)
                            end.record()
                            end.synchronize()
                            host_ms = (time.perf_counter() - host_start) * 1000
                            cuda_ms = float(start.elapsed_time(end))
                            peak_allocated = torch.cuda.max_memory_allocated()
                            peak_reserved = torch.cuda.max_memory_reserved()
                            counts, stages = probes.result(cell["num_steps"]) if probes else (None, None)
                            action_cpu = action.detach().cpu().contiguous()
                            expected_shape = (1, policy.config.chunk_size, policy.config.action_feature.shape[0])
                            if tuple(action_cpu.shape) != expected_shape:
                                raise RuntimeError(f"Expected a complete action chunk of shape {expected_shape}, got {tuple(action_cpu.shape)}.")
                            if not torch.isfinite(action_cpu).all():
                                raise RuntimeError("Non-finite actual policy output.")
                            output_hash = tensor_hash(action_cpu)
                            pair_id = f"b{cell['block']}:d{cell['num_steps']}:{phase}:{index}"
                            record = {"record_type": "inference_trial", "schema_version": 1, **cell, "phase": phase,
                                      "call_index": index, "pair_id": pair_id, "noise_seed": seed, "noise_sha256": noise_hash,
                                      "observation_sha256": observation["observation_sha256"], "action_sha256": output_hash,
                                      "action_shape": list(action_cpu.shape), "action_dtype": str(action_cpu.dtype),
                                      "action_index": len(saved_actions), "cuda_elapsed_ms": cuda_ms, "host_elapsed_ms": host_ms,
                                      "allocated_before_bytes": allocated_before, "peak_allocated_bytes": peak_allocated,
                                      "peak_reserved_bytes": peak_reserved, "instrumentation_counts": counts, "stage_cuda_ms": stages}
                            rows.append(record)
                            saved_actions.append(action_cpu.clone())
                            jsonl_write(trial_file, record)
                            array = action_cpu.float().numpy().copy()
                            if pair_id not in pending:
                                pending[pair_id] = (record, array)
                            else:
                                other, other_array = pending.pop(pair_id)
                                if other["noise_sha256"] != noise_hash or other["instrumentation"] == cell["instrumentation"]:
                                    raise RuntimeError("Invalid on/off pair identity or explicit noise mismatch.")
                                off, on = (record, other) if cell["instrumentation"] == "off" else (other, record)
                                off_array, on_array = (array, other_array) if cell["instrumentation"] == "off" else (other_array, array)
                                equal = output_hash == other["action_sha256"]
                                close = bool(np.allclose(on_array, off_array, rtol=args.rtol, atol=args.atol, equal_nan=False))
                                pair = {"record_type": "inference_pair", "schema_version": 1, "pair_id": pair_id,
                                        "block": cell["block"], "num_steps": cell["num_steps"], "phase": phase, "call_index": index,
                                        "off_action_sha256": off["action_sha256"], "on_action_sha256": on["action_sha256"],
                                        "exact_equal": equal, "allclose": close, "max_abs_diff": float(np.max(np.abs(array - other_array))),
                                        "rtol": args.rtol, "atol": args.atol,
                                        "on_host_elapsed_ms": on["host_elapsed_ms"], "off_host_elapsed_ms": off["host_elapsed_ms"],
                                        "on_cuda_elapsed_ms": on["cuda_elapsed_ms"], "off_cuda_elapsed_ms": off["cuda_elapsed_ms"],
                                        "host_delta_ms": on["host_elapsed_ms"] - off["host_elapsed_ms"],
                                        "cuda_delta_ms": on["cuda_elapsed_ms"] - off["cuda_elapsed_ms"]}
                                pairs.append(pair)
                                jsonl_write(pair_file, pair)
                                if not close:
                                    raise RuntimeError(f"Instrumentation changes predictions beyond tolerance: {pair_id}")
                            del action, action_cpu, prepared, noise
                            active_trial = None
                finally:
                    if probes:
                        probes.__exit__()
                print(json.dumps({"completed_cell": cell, "cell_wall_seconds": time.perf_counter() - cell_start,
                                  "completed_calls": len(rows), "completed_pairs": len(pairs)}), flush=True)
        if pending or len(rows) != args.planned_calls:
            raise RuntimeError("Incomplete benchmark/pairs at nominal completion.")
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
        manifest["error"] = error
        manifest["failure_traceback"] = traceback.format_exc()
        manifest["active_trial_at_failure"] = active_trial
        print(json.dumps({"status": manifest["status"], "error": error}), file=sys.stderr, flush=True)
    finally:
        measured = [row for row in rows if row["phase"] == "measurement"]
        per_cell = []
        for steps in args.num_steps:
            for mode in ("off", "on"):
                subset = [row for row in measured if row["num_steps"] == steps and row["instrumentation"] == mode]
                per_cell.append({"num_steps": steps, "instrumentation": mode, "measured_calls": len(subset),
                                 "cuda_elapsed_ms": distribution([row["cuda_elapsed_ms"] for row in subset]),
                                 "host_elapsed_ms": distribution([row["host_elapsed_ms"] for row in subset]),
                                 "peak_allocated_bytes": max((row["peak_allocated_bytes"] for row in subset), default=None)})
        manifest["total_wall_seconds"] = time.perf_counter() - run_start
        if saved_actions:
            import torch

            # Exact dtype/value tensors in trial order; no padded or synthetic rows.
            torch.save(saved_actions, args.output_dir / "actions.pt")
            manifest["actions_artifact"] = {"path": "actions.pt", "sha256": file_sha256(args.output_dir / "actions.pt"),
                                            "count": len(saved_actions), "index_field": "action_index"}
        summary = {**manifest, "counts": {"issued_prediction_calls": issued_prediction_calls,
                   "completed_actual_chunk_generations": len(rows),
                   "warmup_calls": sum(row["phase"] == "warmup" for row in rows), "measured_calls": len(measured),
                   "completed_pairs": len(pairs), "unpaired_calls": len(pending),
                   "pairs_exact_equal": sum(pair["exact_equal"] for pair in pairs),
                   "pairs_allclose": sum(pair["allclose"] for pair in pairs)}, "per_cell": per_cell,
                   "analysis_limit": "One fixed development observation; block-aware paired analysis is required. No task-quality, rollout, horizon, or general efficiency conclusion."}
        json_write(args.output_dir / "manifest.json", manifest)
        json_write(args.output_dir / "summary.json", summary)
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
