"""pi0.5 Protocol B adapter, client side. Python 3.8 (OpenPI LIBERO client stack). NEVER RUN.

Native stack (02 B3, 07 H2): OpenPI LIBERO submodule `OffScreenRenderEnv`, robosuite 1.4.1, MuJoCo 3.2.3,
MUJOCO_GL=osmesa on the RTX 5090 WSL host (D4). Environment, settling, image handling and state vector reuse the
frozen `client_openpi.py` helpers unchanged. Differences from the frozen rollout:
- every request carries `__request_id__` and `__tick__`, and every served chunk's executed-step attestation is
  verified BEFORE its first action is applied (D5); a failure raises InstrumentIncident;
- unsupported refinement settings stop the phase, and malformed chunks stop for instrument review;
- a first-query marker is written before the first request;
- each executed action records the request and attestation it came from.
"""
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attestation as A  # noqa: E402
from classify import InfraPre, InstrumentIncident, ProtocolStop  # noqa: E402
from v3common import V2, file_sha, sha  # noqa: E402

CERTIFIED_XLA_FLAGS = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"  # D8


def checked_chunk(np, response, request_id, tick, h):
    """Return an attested action chunk, or None for a verified model failure."""
    try:
        chunk = np.asarray(response["actions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InstrumentIncident("broken action chunk contract at tick {}".format(tick)) from exc
    if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < h or chunk.dtype.kind != "f":
        raise InstrumentIncident("broken action chunk contract at tick {}".format(tick))
    evidence = response.get("action_diagnostics")
    if (
        not isinstance(evidence, dict) or evidence.get("request_id") != request_id
        or type(evidence.get("tick")) is not int or evidence["tick"] != tick
        or evidence.get("transformed_inputs_finite") is not True
    ):
        raise InstrumentIncident("invalid or unverified transformed inputs at tick {}".format(tick))
    if not np.isfinite(chunk).all():
        if evidence.get("raw_actions_finite") is not False:
            raise InstrumentIncident("non-finite action has unclear attribution at tick {}".format(tick))
        return None
    return chunk


class Pi05ClientAdapter(object):
    model = "pi05"

    def __init__(self, states_path, port, startup_gate):
        import numpy as np
        import client_openpi as co  # frozen v2 module, imported unchanged
        from libero.libero import benchmark
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        if os.environ.get("MUJOCO_GL") != "osmesa":
            raise SystemExit("REFUSED: the certified renderer is OSMesa (D4)")
        self.np, self.co = np, co
        self.states = np.load(str(states_path))
        self.suites = benchmark.get_benchmark_dict()
        self.client = co.connect(port)
        metadata = self.client.get_server_metadata()
        gate = metadata.get("startup_gate_sha256")
        # 09 C4 check 6: the value comes from the server's startup record on disk, which must say GATE_OK.
        record = json.loads(Path(startup_gate).read_text("utf-8"))
        if record.get("GATE_OK") is not True or file_sha(startup_gate) != gate:
            raise SystemExit("REFUSED: startup gate record missing, not GATE_OK, or not the served one")
        if record.get("server_code_sha256") != metadata.get("server_code_sha256"):
            raise SystemExit("REFUSED: startup gate record was written by different server code")
        if metadata.get("xla_flags") != CERTIFIED_XLA_FLAGS:
            raise SystemExit("REFUSED: server XLA flags {!r} are not the certified flags".format(metadata.get("xla_flags")))
        self.verifier = A.Verifier(gate, CERTIFIED_XLA_FLAGS)
        self.server_metadata = {"startup_gate_sha256": gate, "xla_flags": metadata.get("xla_flags"),
                                "server_code_sha256": metadata.get("server_code_sha256")}

    def stack_config(self):
        return {"model": "pi05", "stack": "openpi_libero_client", "python": sys.version.split()[0],
                "mujoco_gl": os.environ.get("MUJOCO_GL"), "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                "max_steps": self.co.MAX_STEPS, "wait_steps": self.co.WAIT_STEPS, "env_seed": self.co.EXAMPLE_ENV_SEED,
                "resolution": self.co.ENV_RESOLUTION, "server": self.server_metadata}

    def provenance_sha256(self):
        files = {name: file_sha(path) for name, path in (
            ("adapter_pi05.py", __file__), ("attestation.py", A.__file__),
            ("client_openpi.py", V2 / "client_openpi.py"), ("common.py", V2 / "common.py"))}
        return sha({"files": files, "stack": self.stack_config()})

    def run_attempt(self, context):
        co = self.co
        row = context.row
        ident, h, d = row["episode_identity"], row["h"], row["d"]
        try:
            state = self.states[row["state_key"]]
            if hashlib.sha256(state.tobytes()).hexdigest() != row["state_sha256"]:
                raise InfraPre("state hash mismatch for {}".format(row["state_key"]))
            # Fresh environment for this attempt only (02 B3).
            env, prompt = co.make_env(self.suites[ident["suite"]]().get_task(ident["task_id"]))
        except InfraPre:
            raise
        except Exception as error:
            raise InfraPre("environment build failed: {!r}".format(error))
        try:
            result = self._rollout(context, env, prompt, state, ident, h, d)
        except BaseException:
            _close_quietly(env)
            raise
        # A close failure after a finished episode must not turn its outcome into a missing value.
        result["timing"]["env_close_error"] = _close_quietly(env)
        return result

    def _rollout(self, context, env, prompt, state, ident, h, d):
        np, co = self.np, self.co
        row = context.row
        env.reset()
        obs = env.set_init_state(state)
        for _ in range(co.WAIT_STEPS):
            obs, _, _, _ = env.step(co.DUMMY_ACTION)
        settled = co.digest(np.asarray(env.sim.get_state().flatten()))
        first_observation = co.input_hash(co.policy_input(obs, prompt))
        context.record_reset(settled, first_observation)
        co.check_reset((settled, first_observation), context.expected_reset)  # before any policy query
        plan, provenance, actions, done = collections.deque(), [], [], False
        started = time.perf_counter()
        nonfinite = False
        for tick in range(co.MAX_STEPS[ident["suite"]]):
            current = co.policy_input(obs, prompt)
            if not plan:
                rid = A.request_id(row["planned_run_id"], context.attempt_number, tick)
                request = dict(current, __d__=d, __noise_key__=json.dumps([ident, tick]),
                               __request_id__=rid, __tick__=tick)
                if not context.first_query_started():
                    context.mark_first_query()
                response = self.client.infer(request)
                # D5 first: no action from this chunk is applied unless its attestation verifies.
                digest = self.verifier.verify(response.get("attestation"), rid, tick, d)
                if response.get("server_configured_num_steps") != d:
                    raise ProtocolStop("unsupported_refinement_setting: server did not attest configured d={}".format(d))
                if response.get("received_num_steps") != d:
                    raise ProtocolStop("unsupported_refinement_setting: sampler received {}".format(
                        response.get("received_num_steps")))
                chunk = checked_chunk(np, response, rid, tick, h)
                if chunk is None:
                    nonfinite = True
                    break
                chunk_record = {"request_id": rid, "tick": tick, "attestation_sha256": digest,
                                "chunk_sha256": co.digest(chunk), "shape": list(chunk.shape)}
                plan.extend((chunk_record, index, chunk[index]) for index in range(h))
            chunk_record, index, action = plan.popleft()
            if chunk_record is None or chunk_record.get("attestation_sha256") is None:
                raise InstrumentIncident("executed action without an attested chunk record")
            provenance.append({"tick": tick, "request_id": chunk_record["request_id"],
                               "request_tick": chunk_record["tick"], "chunk_index": index,
                               "chunk_sha256": chunk_record["chunk_sha256"],
                               "attestation_sha256": chunk_record["attestation_sha256"],
                               "action_sha256": co.digest(np.asarray(action))})
            actions.append(np.asarray(action))
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
        array_path = context.arrays_dir / "{}.attempt{:02d}.npy".format(row["planned_run_id"], context.attempt_number)
        np.save(str(array_path), np.asarray(actions))
        return {"outcome": {"success": bool(done), "episode_length": len(actions),
                            "termination_reason": ("success" if done else "nonfinite_action_valid_inputs"
                                                   if nonfinite else "task_horizon"),
                            "success_source": "LIBERO done (OffScreenRenderEnv.step)"},
                "settled_state_sha256": settled, "first_observation_sha256": first_observation,
                "action_provenance": provenance, "action_array": array_path.name,
                "timing": {"rollout_wall_s": time.perf_counter() - started}}

    def close(self):
        return None


def _close_quietly(env):
    try:
        env.close()
        return None
    except Exception as error:  # recorded, never raised: the outcome is already fixed
        return repr(error)
