"""SmolVLA Protocol B adapter. Python 3.12 (SmolVLA stack). NEVER RUN.

Native stack (02 B3, 07 H2): LeRobot `LiberoEnv`, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1, OSMesa on the
RTX 5090 WSL host (D4). Policy loading, noise, observation preparation and success reading reuse the frozen
`run_paired.py` / `runner_smolvla.py` helpers unchanged. Differences from the frozen rollout:
- a FRESH environment per attempt (the frozen attempts() reused one env per task, which 02 B3 forbids);
- D7 classification: a non-finite action under valid inputs ends the episode as a scientific failure, without
  applying it; a broken action contract, unclear provenance, an invalid input, or a generation whose counted
  denoise steps differ from d is an InstrumentIncident, raised before the action is applied;
- a missing evaluator success field is an InstrumentIncident (unclear outcome provenance);
- a first-query marker is written before the first policy query.
"""
import hashlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from classify import InfraPre, InstrumentIncident  # noqa: E402
from v3common import V2, file_sha, sha  # noqa: E402


def batch_is_valid(batch):
    """Valid input (02 B5): a non-empty batch with a task string and every tensor finite and non-empty."""
    import torch
    if not isinstance(batch, dict) or not batch:
        return False, "batch is not a non-empty dict"
    if "task" not in batch:
        return False, "batch has no task"
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False, "empty tensor {}".format(key)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                return False, "non-finite input {}".format(key)
    return True, None


class SmolVLAAdapter(object):
    model = "smolvla"

    def __init__(self, states_path, checkpoint):
        import numpy as np
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        if os.environ.get("MUJOCO_GL") != "osmesa":
            raise SystemExit("REFUSED: the certified renderer is OSMesa (D4)")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import run_paired as rp  # frozen helpers, imported unchanged
        import runner_smolvla as rs  # frozen v2 module, imported unchanged
        # The run_paired.py determinism settings, as certified in Protocol A (01 A3a determinism_config).
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rp.seed_environment(0)
        self.np, self.torch, self.rp, self.rs = np, torch, rp, rs
        self.states = np.load(states_path)
        self.checkpoint = checkpoint
        self.policy, self.pre, self.post, self.native = rp.load_policy_and_processors(checkpoint)

    def stack_config(self):
        return {"model": "smolvla", "stack": "lerobot_liberoenv", "python": sys.version.split()[0],
                "mujoco_gl": os.environ.get("MUJOCO_GL"), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "deterministic_algorithms": True, "cudnn_benchmark": False, "allow_tf32": False,
                "native": self.native, "image_size": 360}

    def provenance_sha256(self):
        files = {name: file_sha(path) for name, path in (
            ("adapter_smolvla.py", __file__), ("runner_smolvla.py", V2 / "runner_smolvla.py"),
            ("run_paired.py", HERE.parent / "run_paired.py"), ("common.py", V2 / "common.py"))}
        return sha({"files": files, "stack": self.stack_config(), "checkpoint": str(self.checkpoint)})

    def run_attempt(self, context):
        row = context.row
        ident = row["episode_identity"]
        try:
            state = self.states[row["state_key"]]
            if hashlib.sha256(state.tobytes()).hexdigest() != row["state_sha256"]:
                raise InfraPre("state hash mismatch for {}".format(row["state_key"]))
            # Fresh environment for this attempt only (02 B3).
            env, env_pre, env_post = self.rs.make_task_env(ident["suite"], ident["task_id"], self.policy)
        except InfraPre:
            raise
        except Exception as error:
            raise InfraPre("environment build failed: {!r}".format(error))
        try:
            return self._rollout(context, env, env_pre, env_post, state)
        finally:
            env.close()

    def _rollout(self, context, env, env_pre, env_post, state):
        np, torch, rp, rs = self.np, self.torch, self.rp, self.rs
        from lerobot.envs.utils import NEW_ROLLOUT_OPTION
        from lerobot.utils.constants import ACTION
        policy, pre, post = self.policy, self.pre, self.post
        row = context.row
        ident, h, d = row["episode_identity"], row["h"], row["d"]
        base = env.envs[0].unwrapped
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
        context.record_reset(settled, first_observation)
        from attempt_loop import check_reset
        check_reset((settled, first_observation), context.expected_reset)  # before any policy query
        counters = rp.Counters(policy)
        tap = rs.ChunkTap(policy)
        provenance, actions = [], []
        success, terminated, truncated, nonfinite = False, False, False, False
        started = time.perf_counter()
        try:
            with torch.inference_mode(), (torch.autocast(device_type="cuda") if policy.config.use_amp
                                          else _null()):
                for tick in range(int(base._max_episode_steps)):
                    batch = rp.prepare_observation(observation, env, env_pre, pre)
                    valid, why = batch_is_valid(batch)
                    if not valid:
                        raise InstrumentIncident("invalid policy input at tick {}: {}".format(tick, why), tick=tick)
                    generates = len(policy._queues[ACTION]) == 0
                    noise = rp.make_noise(policy, ident, tick) if generates else None
                    before = dict(counters.counts)
                    chunks_before = len(tap.chunks)
                    if not context.first_query_started():
                        context.mark_first_query()
                    action = env_post({ACTION: post(policy.select_action(batch, noise=noise))})[ACTION]
                    array = action.detach().to("cpu").numpy()
                    new_chunks = len(tap.chunks) - chunks_before
                    # Provenance and count, checked before the action can be applied (D7).
                    if generates:
                        if new_chunks != 1 or counters.counts["generations"] - before["generations"] != 1:
                            raise InstrumentIncident("generation not recorded exactly once at tick {}".format(tick))
                        executed = counters.counts["denoise_steps"] - before["denoise_steps"]
                        if executed != d:
                            raise InstrumentIncident("generation executed {} denoise steps, expected d={}".format(
                                executed, d), tick=tick)
                    elif new_chunks != 0:
                        raise InstrumentIncident("unexpected generation at tick {}".format(tick))
                    if not tap.chunks:
                        raise InstrumentIncident("executed action with no generation record")
                    if array.shape != (1, 7) or array.dtype.kind != "f":
                        raise InstrumentIncident("broken action contract: shape {} dtype {}".format(
                            array.shape, array.dtype), tick=tick)
                    if not np.isfinite(array).all():
                        # D7: valid inputs (checked above) and a non-finite action -> scientific failure.
                        nonfinite = True
                        break
                    provenance.append({"tick": tick, "new_generation": generates,
                                       "generation_index": len(tap.chunks) - 1,
                                       "chunk_sha256": tap.chunks[-1]["sha256"],
                                       "action_sha256": rp.array_record(array[0])["sha256"]})
                    actions.append(array[0])
                    observation, _, terminated_array, truncated_array, info = env.step(array)
                    terminated, truncated = bool(terminated_array[0]), bool(truncated_array[0])
                    try:
                        success = success or rp.batch_one_success(info)
                    except ValueError as error:
                        raise InstrumentIncident("evaluator success field unreadable: {}".format(error), tick=tick)
                    if terminated or truncated:
                        break
        finally:
            tap.restore()
            counts = dict(counters.counts)
            counters.restore()
        if counts["denoise_steps"] != counts["generations"] * d:
            raise InstrumentIncident("episode denoise count {} != generations {} x d={}".format(
                counts["denoise_steps"], counts["generations"], d))
        if success:
            reason = "success"
        elif nonfinite:
            reason = "nonfinite_action_valid_inputs"
        else:
            reason = "terminated" if terminated else "truncated" if truncated else "task_horizon"
        array_path = context.arrays_dir / "{}.attempt{:02d}.npy".format(row["planned_run_id"], context.attempt_number)
        np.save(array_path, np.asarray(actions))
        return {"outcome": {"success": bool(success), "episode_length": len(actions), "termination_reason": reason,
                            "success_source": "info['is_success'] via run_paired.batch_one_success, OR over steps"},
                "settled_state_sha256": settled, "first_observation_sha256": first_observation,
                "action_provenance": provenance, "action_array": array_path.name,
                "timing": {"rollout_wall_s": time.perf_counter() - started, "generation_counts": counts}}

    def close(self):
        return None


class _null(object):
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
