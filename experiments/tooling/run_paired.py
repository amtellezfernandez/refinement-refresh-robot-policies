#!/usr/bin/env python3
"""One-loaded-checkpoint paired LIBERO evaluation against pinned LeRobot.

Runs real inference; requires Linux, CUDA, LeRobot+LIBERO and a local checkpoint.
Writes exclusive output artifacts and refuses partial blocks. A completed run
certifies execution of its recorded subset, not that reserved IDs were unseen in
unarchived studies. No real-time, energy, or safety claim is inferred.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

EXPECTED_REVISION = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
CHECKPOINT_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
PINNED_SOURCE_HASHES = {
    "policies/smolvla/modeling_smolvla.py": "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d",
    "policies/smolvla/smolvlm_with_expert.py": "996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf",
    "policies/common/flow_matching.py": "ce1cff1aab02f971dacfc6f78779c6c6813a404165d2c16bf66c3ad2174049eb",
    "envs/libero.py": "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71",
    "envs/utils.py": "4a37ce5d9ac70a22f88e63c02b84ef3e30be72725e52fe6a03ec7e3407c143e7",
    "processor/env_processor.py": "41cf74048b233652b2b03d38c79b30d6f1d341592d553a5625f971de555f3a3a",
    "scripts/lerobot_eval.py": "fddec10c709401dd863814091d525fcd54d41cc6ccf51f7ef8300459cf69b873",
}
# The runner supports a frozen, plan-declared finite grid.  The historical
# four-cell pilot remains its own plan; do not silently apply its condition set
# to a later discovery study.
DEFAULT_CONDITIONS = {(1, 5), (1, 10), (5, 5), (5, 10)}
CONDITIONS = DEFAULT_CONDITIONS  # public historical test/analysis compatibility


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=str).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def keyed_seed(identity, tick=None, domain="policy_noise"):
    key = [domain, identity] if tick is None else [domain, identity, tick]
    return int.from_bytes(hashlib.sha256(canonical(key)).digest()[:4], "little")


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_checkpoint_marker(checkpoint):
    marker = Path(checkpoint) / ".robopad_revision"
    require(marker.is_file(), f"checkpoint revision marker missing: {marker}")
    require(marker.read_text(encoding="utf-8").strip() == CHECKPOINT_REVISION,
            "checkpoint revision marker disagrees with pinned revision")


def verify_source_tree(package_root=None):
    if package_root is None:
        from lerobot.policies.smolvla import modeling_smolvla
        package_root = Path(inspect.getfile(modeling_smolvla)).parents[2]
    actual = {}
    for relative, expected in PINNED_SOURCE_HASHES.items():
        path = Path(package_root) / relative
        require(path.is_file(), f"pinned source missing: {path}")
        actual[relative] = file_sha(path)
        require(actual[relative] == expected, f"source differs from pinned revision: {path}")
    return actual


def batch_one_success(info):
    """Read the evaluator-facing vector info independently of the base-env hook."""
    if "final_info" in info:
        final = info["final_info"]
        if isinstance(final, dict):
            return batch_one_success(final)
        require(len(final) == 1 and final[0] is not None, "invalid batch-one final_info")
        return batch_one_success(final[0])
    require("is_success" in info, "environment success field missing")
    value = info["is_success"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        require(len(value) == 1, "success info must be batch one")
        value = value[0]
    return bool(value)


def select_schedule(plan, max_blocks=None):
    """Select whole state/seed pairs, equally per task, before outcomes exist."""
    rows = plan["schedule"]
    require(rows and sha(rows) == plan["scheduling"]["schedule_sha256"],
            "plan schedule digest missing or incorrect")
    declared = plan.get("design", {}).get("conditions")
    require(isinstance(declared, list) and declared, "plan must declare nonempty conditions")
    conditions = {(row.get("h"), row.get("denoising_steps")) for row in declared}
    require(all(type(h) is int and h > 0 and type(d) is int and d > 0
                for h, d in conditions), "h and denoising_steps must be positive integers")
    require(len(conditions) == len(declared), "duplicate declared conditions")
    per_block = len(conditions)
    require(plan["design"].get("conditions_per_block") == per_block,
            "conditions_per_block disagrees with declared conditions")
    blocks = {}
    order = []
    for row in rows:
        bid = row["block_id"]
        if bid not in blocks:
            blocks[bid] = []
            order.append(bid)
        blocks[bid].append(row)
    for bid, block in blocks.items():
        require(len(block) == per_block and {(r["h"], r["denoising_steps"]) for r in block} == conditions,
                f"incomplete or mismatched condition block {bid} (historical four-condition or declared grid)")
        require(all(r["episode_identity"] == block[0]["episode_identity"] for r in block),
                f"mixed identities in block {bid}")
    require([row["block_id"] for row in rows] ==
            [bid for bid in order for _ in range(per_block)], "condition blocks must be contiguous")
    require(len({r["planned_run_id"] for r in rows}) == len(rows), "duplicate planned_run_id")
    if max_blocks is None:
        return copy.deepcopy(rows)
    require(type(max_blocks) is int and 0 < max_blocks <= len(blocks), "invalid max-blocks")
    if max_blocks == len(blocks):
        return copy.deepcopy(rows)
    task_states = defaultdict(dict)
    for bid in order:
        e = blocks[bid][0]["episode_identity"]
        task = (e["suite"], e["task_id"])
        state = e["initial_state_id"]
        task_states[task].setdefault(state, []).append(bid)
    seeds = set(plan["design"]["policy_seeds"])
    divisor = len(task_states) * len(seeds)
    require(max_blocks % divisor == 0,
            f"--max-blocks must be a multiple of {divisor} to preserve task/state/seed balance")
    states_per_task = max_blocks // divisor
    selected = set()
    for task, states in task_states.items():
        require(len(states) >= states_per_task, f"not enough states for {task}")
        for state, bids in list(states.items())[:states_per_task]:
            actual = {blocks[bid][0]["episode_identity"]["policy_seed"] for bid in bids}
            require(actual == seeds and len(bids) == len(seeds), "incomplete policy-seed pair")
            selected.update(bids)
    selected_rows = [row for row in rows if row["block_id"] in selected]
    require(len(selected_rows) == max_blocks * per_block, "balanced selection size mismatch")
    return copy.deepcopy(selected_rows)


def development_schedule():
    identity = {"suite": "libero_spatial", "task_id": 0,
                "initial_state_id": 0, "policy_seed": 1000}
    bid = "development.libero_spatial.task000.state000.seed1000"
    return [{"schedule_index": index, "block_index": 0, "condition_index": index,
             "block_id": bid, "episode_identity": identity.copy(),
             "condition_id": f"h{h}_d{d}", "planned_run_id": f"{bid}.h{h}_d{d}",
             "h": h, "denoising_steps": d, "native_reference": (h, d) == (1, 10)}
            for index, (h, d) in enumerate(((1, 10), (5, 5), (1, 5), (5, 10)))]


def array_record(value):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(value))
    require(a.dtype.kind in "fiu" and np.isfinite(a).all(), "nonfinite/non-numeric array")
    header = json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode()
    return {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(),
            "sha256": hashlib.sha256(header + a.tobytes()).hexdigest()}


def observation_sha(observation):
    import numpy as np
    def encode(value):
        if isinstance(value, dict):
            return {key: encode(item) for key, item in sorted(value.items())}
        if isinstance(value, np.ndarray):
            return array_record(value)["sha256"]
        return value
    return sha(encode(observation))


def observation_field_shas(observation):
    """Return compact recursive hashes for audit logs without storing inputs."""
    import numpy as np
    if isinstance(observation, dict):
        return {key: observation_field_shas(value)
                for key, value in sorted(observation.items())}
    if isinstance(observation, np.ndarray):
        return array_record(observation)["sha256"]
    return sha(observation)


def seed_environment(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_noise(policy, identity, tick, domain="policy_noise"):
    import torch
    generator = torch.Generator(device=policy.config.device)
    generator.manual_seed(keyed_seed(identity, tick, domain))
    return torch.normal(0.0, 1.0,
                        size=(1, policy.config.chunk_size, policy.config.max_action_dim),
                        generator=generator, device=policy.config.device, dtype=torch.float32)


def prepare_observation(observation, env, env_preprocessor, preprocessor):
    from lerobot.envs import preprocess_observation
    batch = preprocess_observation(observation)
    batch["task"] = list(env.call("task_description"))
    return preprocessor(env_preprocessor(batch))


def load_policy_and_processors(checkpoint):
    """Strict complete weight load; no factory path that defaults to strict=False."""
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    checkpoint = Path(checkpoint).resolve()
    verify_checkpoint_marker(checkpoint)
    verify_source_tree()
    require((checkpoint / "model.safetensors").is_file(), "local model.safetensors missing")
    config = SmolVLAConfig.from_pretrained(checkpoint)
    native = {"h": config.n_action_steps, "d": config.num_steps, "chunk_size": config.chunk_size}
    require(native == {"h": 1, "d": 10, "chunk_size": 50},
            f"unexpected checkpoint native configuration: {native}")
    require(config.rtc_config is None or not config.rtc_config.enabled,
            "runner requires RTC disabled")
    config.device = "cuda"
    config.compile_model = False
    config.load_vlm_weights = False  # full final checkpoint is loaded strictly below
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True,
                                         local_files_only=True)
    policy.eval()
    require(policy.config is policy.model.config, "policy/model config alias changed")
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {}},
        })
    return policy, pre, post, native


class Counters:
    """Actual Python model-boundary counts plus CUDA stream timings, not FLOPs."""
    def __init__(self, policy):
        self.policy = policy
        self.originals = []
        self.reset()
        self._wrap(policy.model, "denoise_step", "denoise_steps")
        self._wrap(policy.model.vlm_with_expert, "embed_image", "vision_encoder_calls")
        original = policy.model.vlm_with_expert.forward
        self.originals.append((policy.model.vlm_with_expert, "forward", original,
                               "forward" in policy.model.vlm_with_expert.__dict__))
        def forward(*args, **kwargs):
            inputs = kwargs.get("inputs_embeds")
            if inputs is not None:
                if inputs[0] is not None and inputs[1] is None:
                    self.counts["prefill_calls"] += 1
                elif inputs[0] is None and inputs[1] is not None:
                    self.counts["action_expert_calls"] += 1
            return original(*args, **kwargs)
        policy.model.vlm_with_expert.forward = forward
        original_chunk = policy._get_action_chunk
        self.originals.append((policy, "_get_action_chunk", original_chunk,
                               "_get_action_chunk" in policy.__dict__))
        def chunk(*args, **kwargs):
            import torch
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter()
            start.record()
            result = original_chunk(*args, **kwargs)
            end.record()
            end.synchronize()
            self.host_s.append(time.perf_counter() - host_start)
            self.cuda_ms.append(float(start.elapsed_time(end)))
            self.counts["generations"] += 1
            return result
        policy._get_action_chunk = chunk

    def _wrap(self, owner, name, metric):
        original = getattr(owner, name)
        self.originals.append((owner, name, original, name in owner.__dict__))
        def call(*args, **kwargs):
            self.counts[metric] += 1
            return original(*args, **kwargs)
        setattr(owner, name, call)

    def reset(self):
        self.counts = {name: 0 for name in (
            "generations", "denoise_steps", "vision_encoder_calls",
            "prefill_calls", "action_expert_calls")}
        self.host_s, self.cuda_ms = [], []

    def restore(self):
        for owner, name, original, had_local in reversed(self.originals):
            if had_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.originals.clear()


def json_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False, default=str) + "\n")
    stream.flush()


def directory_manifest(root):
    """Hash every file under a resolved runtime dependency directory."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"runtime dependency directory is missing: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha(path),
                "symlink_target": os.readlink(path) if path.is_symlink() else None,
            })
    if not entries:
        raise RuntimeError(f"runtime dependency directory is empty: {root}")
    return {"root": str(root), "file_count": len(entries),
            "tree_sha256": sha(entries), "files": entries}


def runtime_manifest(args, policy, native, schedule, load_wall_s):
    import torch
    import numpy as np
    import lerobot
    import libero
    import mujoco
    from libero.libero import get_libero_path
    from lerobot.policies.smolvla import modeling_smolvla
    from lerobot.envs import libero as libero_env_module
    libero_assets = directory_manifest(get_libero_path("assets"))
    files = {}
    for path in Path(args.checkpoint).resolve().rglob("*"):
        if path.is_file() and ".cache" not in path.parts and not path.name.startswith("."):
            files[str(path.relative_to(Path(args.checkpoint).resolve()))] = file_sha(path)
    # Keep every visible distribution record. Multiple site-packages roots can
    # expose more than one dist-info record for a package; a dict comprehension
    # silently let the last one overwrite the others.
    distribution_records = defaultdict(list)
    for dist in importlib.metadata.distributions():
        display_name = dist.metadata.get("Name")
        if not display_name:
            continue
        key = display_name.lower().replace("_", "-").replace(".", "-")
        distribution_records[key].append({
            "name": display_name,
            "version": dist.version,
            "metadata_path": str(getattr(dist, "_path", "")),
            "install_root": str(Path(dist.locate_file("")).resolve()),
        })
    distribution_records = {
        name: sorted(records, key=lambda row: (row["version"], row["metadata_path"]))
        for name, records in sorted(distribution_records.items())
    }
    # The imported module is authoritative for what this process executes.
    # Require matching distribution metadata and record its path/version too.
    imported_modules = {}
    for name, module in (("numpy", np), ("torch", torch), ("mujoco", mujoco)):
        module_version = str(module.__version__)
        module_file = str(Path(module.__file__).resolve())
        base_version = module_version.split("+", 1)[0]
        candidates = distribution_records.get(name, [])
        if not any(row["version"].split("+", 1)[0] == base_version for row in candidates):
            raise RuntimeError(
                f"imported {name} {module_version} from {module_file} has no matching "
                "installed distribution metadata"
            )
        imported_modules[name] = {"version": module_version, "file": module_file}
    # Preserve a simple mapping only where metadata is unambiguous. Ambiguous
    # package names are represented by null and fully listed below.
    packages = {
        name: (records[0]["version"] if len({row["version"] for row in records}) == 1 else None)
        for name, records in distribution_records.items()
    }
    duplicate_distribution_records = {
        name: records for name, records in distribution_records.items() if len(records) > 1
    }
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "nvidia-smi unavailable"
    return {"schema_version": 2, "kind": "robopad_paired_runtime",
            "status_at_creation": "scheduled_not_completed",
            "mode": "development" if args.development else "planned",
            "argv": vars(args), "python": sys.version, "platform": platform.platform(),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "precision": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "autocast_enabled": policy.config.use_amp},
            "numpy_version": np.__version__, "mujoco_version": mujoco.__version__,
            "gpu": gpu, "device_name": torch.cuda.get_device_name(0),
            "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "packages": packages, "package_distribution_candidates": duplicate_distribution_records,
            "imported_modules": imported_modules, "native_checkpoint_config": native,
            "effective_policy_config": dataclasses.asdict(policy.config),
            "checkpoint_expected_revision": CHECKPOINT_REVISION,
            "checkpoint_revision_marker_verified": True,
            "checkpoint_files_sha256": files,
            "lerobot_expected_revision": EXPECTED_REVISION,
            "verified_pinned_source_sha256": verify_source_tree(),
            "source_paths": {"lerobot": str(lerobot.__file__), "libero": str(libero.__file__)},
            "libero_assets": libero_assets,
            "source_sha256": {str(path): file_sha(path) for path in (
                Path(__file__), Path(inspect.getfile(modeling_smolvla)),
                Path(inspect.getfile(libero_env_module)))},
            "model_loading_wall_s": load_wall_s,
            "execution_environment": {key: os.environ.get(key) for key in (
                "MUJOCO_GL", "PYOPENGL_PLATFORM", "HF_HOME", "HF_HUB_OFFLINE", "LIBERO_CONFIG_PATH",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
            "selected_schedule_sha256": sha(schedule), "selected_schedule": schedule,
            "external_holdout_status": "not_certified_against_unarchived_runs",
            "control": {"camera_resolution": [args.image_size, args.image_size],
                        "camera_names": ["agentview_image", "robot0_eye_in_hand_image"],
                        "control_mode": "relative", "hard_reset": True, "batch_size": 1,
                        "synchronous_simulation": True, "nominal_hz": 20},
            "noise_contract": "Explicit float32 torch.normal generator keyed by full episode identity and simulation tick; h,d excluded.",
            "cost_boundary": "Rollout wall includes preprocessing, noise, generation, actions, simulator/rendering and journal writes; excludes reset, warmup, model load and environment creation.",
            "unmeasured": ["energy", "real_time_deadlines", "safety", "hardware_kernel_FLOPs"]}


def parity_report(reference, initial_state, initial_observation_sha256, actions, success):
    action_records = [array_record(action) for action in actions]
    stream_digest = sha([record["sha256"] for record in action_records])
    checks = {
        "initial_state_equal": initial_state["sha256"] == reference["initial_state"]["sha256"],
        "initial_observation_equal": initial_observation_sha256 == reference["initial_observation_sha256"],
        "action_stream_exact": stream_digest == reference["action_stream_sha256"],
        "episode_steps_equal": len(action_records) == reference["steps"],
        "success_equal": bool(success) == reference["success"],
    }
    return {"schema_version": 1, "status": "passed" if all(checks.values()) else "failed",
            "checks": checks, "reference_run_id": reference["run_id"],
            "episode_identity": reference["episode_identity"],
            "h": reference["h"], "denoising_steps": reference["denoising_steps"],
            "upstream_steps": len(action_records), "upstream_success": bool(success),
            "upstream_initial_state": initial_state,
            "upstream_initial_observation_sha256": initial_observation_sha256,
            "upstream_action_stream_sha256": stream_digest, "upstream_actions": action_records,
            "scope": "Pinned upstream rollout loop with same explicit tick-keyed noise and state; does not assert the stock global-RNG trajectory is identical."}


def run_upstream_parity(reference, env, policy, pre, post, env_pre, env_post):
    """Run the actual upstream loop, with counters removed by the caller."""
    import torch
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.constants import ACTION
    base = env.envs[0].unwrapped
    identity = reference["episode_identity"]
    state_id = identity["initial_state_id"]
    require(0 <= state_id < len(base._init_states), "parity state out of range")
    base.init_state_id = state_id
    policy.config.n_action_steps = reference["h"]
    policy.config.num_steps = reference["denoising_steps"]
    policy.reset()
    for processor in (pre, post, env_pre, env_post):
        if hasattr(processor, "reset"):
            processor.reset()
    seed_environment(reference["environment_seed"])
    tick, capture = 0, {}
    original_select, original_reset = policy.select_action, env.reset
    def select(batch, *args, **kwargs):
        nonlocal tick
        require(not args and not kwargs, "unexpected upstream select_action arguments")
        noise = make_noise(policy, identity, tick) if len(policy._queues[ACTION]) == 0 else None
        action = original_select(batch, noise=noise)
        tick += 1
        return action
    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        capture["state"] = array_record(base._env.sim.get_state().flatten())
        capture["observation_sha256"] = observation_sha(result[0])
        return result
    policy.select_action, env.reset = select, reset
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda") if policy.config.use_amp else nullcontext():
            result = rollout(env, policy, env_pre, env_post, pre, post,
                             seeds=[reference["environment_seed"]])
    finally:
        policy.select_action, env.reset = original_select, original_reset
    elapsed = time.perf_counter() - started
    report = parity_report(reference, capture["state"], capture["observation_sha256"],
                           result[ACTION][0].cpu().numpy(), bool(result["success"].any().item()))
    report.update(upstream_wall_s=elapsed,
                  upstream_evaluator_source_sha256=file_sha(inspect.getfile(rollout)),
                  instrumentation="Model and environment counters removed; only reset capture and explicit-noise injection remain.")
    return report


def run_episode(row, env, policy, pre, post, env_pre, env_post, counters,
                warmed, args, step_stream, runtime_sha, environment_creation_wall_s,
                expected_pair=None):
    import numpy as np
    import torch
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.utils.constants import ACTION
    identity = row["episode_identity"]
    state_id = identity["initial_state_id"]
    base = env.envs[0].unwrapped
    require(type(state_id) is int and 0 <= state_id < len(base._init_states),
            f"requested state {state_id} out of range; modulo fallback forbidden")
    require(base.task_id == identity["task_id"], "actual task differs from schedule")
    require(base.hard_reset and base.init_states, "hard reset with explicit initial states required")
    environment_key = [identity["suite"], identity["task_id"], state_id]
    environment_seed = keyed_seed(environment_key, domain="environment")
    policy.config.n_action_steps = row["h"]
    require(1 <= row["h"] <= policy.config.chunk_size, "prefix exceeds predicted chunk")
    policy.config.num_steps = row["denoising_steps"]
    policy.reset()
    for processor in (pre, post, env_pre, env_post):
        if hasattr(processor, "reset"):
            processor.reset()
    base.init_state_id = state_id
    seed_environment(environment_seed)
    reset_start = time.perf_counter()
    observation, _ = env.reset(seed=[environment_seed], options={NEW_ROLLOUT_OPTION: True})
    reset_wall_s = time.perf_counter() - reset_start
    initial = array_record(base._env.sim.get_state().flatten())
    initial_obs_sha = observation_sha(observation)
    if expected_pair is not None:
        require(expected_pair == (initial["sha256"], initial_obs_sha),
                "paired reset state/observation mismatch before policy execution")
    max_steps = int(base._max_episode_steps)
    prepared_start = time.perf_counter()
    batch = prepare_observation(observation, env, env_pre, pre)
    initial_preprocess_wall = time.perf_counter() - prepared_start
    require(batch["observation.state"].shape[-1] == 8, "expected eight state features")
    counters.reset()
    warmup_start = time.perf_counter()
    condition = (row["h"], row["denoising_steps"])
    amp = lambda: torch.autocast(device_type="cuda") if policy.config.use_amp else nullcontext()
    if condition not in warmed:
        with torch.inference_mode(), amp():
            for index in range(args.warmup):
                policy.reset()
                policy.select_action(copy.deepcopy(batch),
                                     noise=make_noise(policy, identity, index, "warmup"))
        policy.reset()
        warmed.add(condition)
    warmup_wall_s = time.perf_counter() - warmup_start
    warmup_counts = counters.counts.copy()
    counters.reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    driver_records = []
    original_step = base.step
    def independent_step(action):
        consumed = array_record(action)
        result = original_step(action)
        _, reward, terminated, truncated, info = result
        driver_records.append({"action": consumed, "reward": float(reward),
                               "terminated": bool(terminated), "truncated": bool(truncated),
                               "is_success": bool(info["is_success"])})
        return result
    base.step = independent_step
    run_id = row["planned_run_id"] + ".attempt_" + args.attempt_namespace
    json_line(step_stream, {"kind": "episode_start", "run_id": run_id,
                           "episode_identity": identity, "initial_state": initial,
                           "initial_observation_sha256": initial_obs_sha})
    action_hashes = []
    env_wall_s, preprocess_wall_s = 0.0, 0.0
    success, terminated, truncated = False, False, False
    generation_origin_tick = 0
    supplied_observation = copy.deepcopy(observation)
    supplied_observation_tick = 0
    rollout_start = time.perf_counter()
    try:
        with torch.inference_mode(), amp():
            for tick in range(max_steps):
                started = time.perf_counter()
                freshness = getattr(args, "observation_refresh", 1)
                input_refreshed = tick % freshness == 0
                if input_refreshed:
                    supplied_observation = copy.deepcopy(observation)
                    supplied_observation_tick = tick
                current_observation_sha = observation_sha(observation)
                supplied_observation_sha = observation_sha(supplied_observation)
                batch = prepare_observation(supplied_observation, env, env_pre, pre)
                preprocess_wall_s += time.perf_counter() - started
                generates = len(policy._queues[ACTION]) == 0
                noise = None
                if generates:
                    generation_origin_tick = tick
                    noise = make_noise(policy, identity, tick)
                action = policy.select_action(batch, noise=noise)
                action = post(action)
                action = env_post({ACTION: action})[ACTION]
                action_numpy = action.detach().to("cpu").numpy()
                require(action_numpy.shape == (1, 7), "expected one seven-dimensional action")
                produced = array_record(action_numpy[0])
                started = time.perf_counter()
                observation, reward, terminated_array, truncated_array, info = env.step(action_numpy)
                env_wall_s += time.perf_counter() - started
                require(len(driver_records) == tick + 1, "independent environment step count mismatch")
                driver = driver_records[-1]
                require(produced["sha256"] == driver["action"]["sha256"], "producer/consumer action mismatch")
                require(np.isclose(float(reward[0]), driver["reward"], rtol=0, atol=0),
                        "driver/evaluator reward disagreement")
                terminated, truncated = bool(terminated_array[0]), bool(truncated_array[0])
                require(terminated == driver["terminated"], "termination disagreement")
                evaluator_success = batch_one_success(info)
                require(evaluator_success == driver["is_success"], "driver/evaluator success disagreement")
                success = success or evaluator_success
                action_hashes.append(produced["sha256"])
                json_line(step_stream, {
                    "kind": "step", "run_id": run_id, "tick": tick,
                    "input_refreshed": input_refreshed,
                    "current_observation_sha256": current_observation_sha,
                    "supplied_observation_sha256": supplied_observation_sha,
                    "supplied_observation_field_sha256": observation_field_shas(supplied_observation),
                    "supplied_observation_tick": supplied_observation_tick,
                    "supplied_observation_age_ticks": tick - supplied_observation_tick,
                    "generation_origin_tick": generation_origin_tick,
                    "action_age_ticks": tick - generation_origin_tick,
                    "new_generation": generates, "noise_seed": keyed_seed(identity, tick) if generates else None,
                    "producer_action": produced, "environment_action_sha256": driver["action"]["sha256"],
                    "reward": driver["reward"], "driver_terminated": driver["terminated"],
                    "driver_truncated": driver["truncated"], "driver_is_success": driver["is_success"],
                    "vector_terminated": terminated, "vector_truncated": truncated,
                    "vector_is_success": evaluator_success})
                if terminated or truncated:
                    break
        torch.cuda.synchronize()
        rollout_wall_s = time.perf_counter() - rollout_start
    finally:
        base.step = original_step
    steps = len(action_hashes)
    require(steps > 0 and len(driver_records) == steps, "empty/inconsistent episode")
    require(counters.counts["generations"] == (steps + row["h"] - 1) // row["h"], "generation schedule disagreement")
    require(counters.counts["denoise_steps"] == counters.counts["generations"] * row["denoising_steps"],
            "actual denoising evaluations do not match configured solver")
    require(counters.counts["prefill_calls"] == counters.counts["generations"], "prefill count disagreement")
    require(counters.counts["action_expert_calls"] == counters.counts["denoise_steps"], "expert/NFE mismatch")
    reason = "success" if success else "terminated" if terminated else "truncated" if truncated else "task_horizon"
    record = {"schema_version": 1, "status": "completed", "run_id": run_id,
              "planned_run_id": row["planned_run_id"], "block_id": row["block_id"],
              "episode_identity": identity, "h": row["h"], "denoising_steps": row["denoising_steps"],
              "native_reference": row["native_reference"], "success": success,
              "termination_reason": reason, "steps": steps, "task_horizon": max_steps,
              "environment_seed": environment_seed, "initial_state": initial,
              "initial_observation_sha256": initial_obs_sha,
              "initial_state_set_sha256": array_record(np.asarray(base._init_states))["sha256"],
              "task_bddl_sha256": file_sha(base._task_bddl_file),
              "action_stream_sha256": sha(action_hashes), "counts": counters.counts.copy(),
              "warmup_counts": warmup_counts,
              "timing": {"environment_creation_wall_s": environment_creation_wall_s,
                         "reset_wall_s": reset_wall_s, "warmup_wall_s": warmup_wall_s,
                         "warmup_input_preprocess_wall_s": initial_preprocess_wall,
                         "rollout_wall_s": rollout_wall_s, "preprocess_wall_s": preprocess_wall_s,
                         "env_step_wall_s": env_wall_s,
                         "generation_host_s": list(counters.host_s),
                         "generation_cuda_ms": list(counters.cuda_ms)},
              "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
              "independent_driver": {"steps": len(driver_records), "success": any(x["is_success"] for x in driver_records),
                                     "terminal_record": {k: v for k, v in driver_records[-1].items() if k != "action"}},
              "runtime_manifest_sha256": runtime_sha}
    json_line(step_stream, {"kind": "episode_completed", "run_id": run_id,
                           "success": success, "steps": steps, "action_stream_sha256": record["action_stream_sha256"]})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(__file__).parents[1] / "planned_evaluation.json")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--upstream-parity", action="store_true",
                        help="development-only native episode check against upstream rollout")
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--observation-refresh", type=int, default=1,
                        help="Diagnostic only: refresh all inputs (images and proprioception) every N ticks")
    args = parser.parse_args()
    require(args.warmup >= 1 and args.image_size > 0, "positive warmup/image size required")
    require(args.observation_refresh >= 1, "positive observation refresh required")
    require(not args.upstream_parity or args.development, "upstream parity is a development gate")
    if args.development:
        require(args.max_blocks in (None, 1), "development has exactly one complete block")
        schedule = development_schedule()
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        schedule = select_schedule(plan, args.max_blocks)
    args.output.mkdir(parents=True, exist_ok=False)
    args.attempt_namespace = uuid.uuid4().hex
    (args.output / "selected_schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    try:
        execute(args, schedule)
    except BaseException as error:
        failure_path = args.output / "failure.json"
        if not failure_path.exists():
            failure = {"status": "failed_or_interrupted", "phase": "runtime_setup",
                       "completed_attempts": 0, "expected_attempts": len(schedule),
                       "error": repr(error), "traceback": traceback.format_exc()}
            failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


def execute(args, schedule):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    require(torch.cuda.is_available(), "CUDA GPU required")
    require(torch.cuda.device_count() >= 1, "no visible CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_environment(0)
    started = time.perf_counter()
    policy, pre, post, native = load_policy_and_processors(args.checkpoint)
    load_wall_s = time.perf_counter() - started
    runtime = runtime_manifest(args, policy, native, schedule, load_wall_s)
    runtime_digest = sha(runtime)
    (args.output / "runtime_manifest.json").write_text(
        json.dumps(runtime, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    counters = Counters(policy)
    env, env_key, env_pre, env_post = None, None, None, None
    completed, paired_starts, warmed = [], {}, set()
    try:
        with (args.output / "episodes.jsonl").open("x", buffering=1) as episode_stream, \
             (args.output / "steps.jsonl").open("x", buffering=1) as step_stream:
            for row in schedule:
                key = (row["episode_identity"]["suite"], row["episode_identity"]["task_id"])
                creation_wall_s = 0.0
                if key != env_key:
                    if env is not None:
                        env.close()
                    started = time.perf_counter()
                    cfg = LiberoEnv(task=key[0], task_ids=[key[1]], init_states=True,
                                    hard_reset=True, control_mode="relative",
                                    observation_height=args.image_size, observation_width=args.image_size)
                    env = make_env(cfg, n_envs=1, use_async_envs=False)[key[0]][key[1]]
                    env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
                    creation_wall_s = time.perf_counter() - started
                    env_key = key
                print(f"START {len(completed)+1}/{len(schedule)} {row['planned_run_id']}", flush=True)
                result = run_episode(row, env, policy, pre, post, env_pre, env_post,
                                     counters, warmed, args, step_stream, runtime_digest, creation_wall_s,
                                     expected_pair=paired_starts.get(row["block_id"]))
                pair = (result["initial_state"]["sha256"], result["initial_observation_sha256"])
                if row["block_id"] in paired_starts:
                    require(paired_starts[row["block_id"]] == pair, "paired reset state/observation mismatch")
                else:
                    paired_starts[row["block_id"]] = pair
                result["paired_initial_state_verified"] = True
                result["pairing_scope"] = "all completed conditions in this block; full block checked at final completion"
                json_line(episode_stream, result)
                completed.append(result)
                print(f"DONE success={result['success']} steps={result['steps']} "
                      f"chunks={result['counts']['generations']} NFE={result['counts']['denoise_steps']} "
                      f"wall={result['timing']['rollout_wall_s']:.3f}s", flush=True)
                if args.upstream_parity and row["native_reference"]:
                    print("CHECK upstream native-loop parity with controlled noise", flush=True)
                    counters.restore()
                    try:
                        parity = run_upstream_parity(result, env, policy, pre, post, env_pre, env_post)
                    finally:
                        counters = Counters(policy)
                    (args.output / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
                    require(parity["status"] == "passed", f"upstream-loop parity failed: {parity['checks']}")
                    print("PASS upstream native-loop parity", flush=True)
        summary = {"schema_version": 1, "status": "completed", "mode": runtime["mode"],
                   "runtime_manifest_sha256": runtime_digest, "schedule_sha256": sha(schedule),
                   "completed_attempts": len(completed), "completed_blocks": len(paired_starts),
                   "expected_attempts": len(schedule), "episodes": completed,
                   "aggregate_successes": sum(row["success"] for row in completed),
                   "performance_claim_scope": "selected finite task/state/seed set only"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    except BaseException as error:
        failure = {"status": "failed_or_interrupted", "completed_attempts": len(completed),
                   "expected_attempts": len(schedule), "error": repr(error),
                   "active_planned_run_id": row["planned_run_id"] if "row" in locals() else None,
                   "traceback": traceback.format_exc(), "runtime_manifest_sha256": runtime_digest}
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        counters.restore()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
