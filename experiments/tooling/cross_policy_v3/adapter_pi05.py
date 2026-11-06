"""pi0.5 Protocol B adapter, client side. Python 3.8 (OpenPI LIBERO client stack). NEVER RUN.

Native stack (02 B3, 07 H2): OpenPI LIBERO submodule `OffScreenRenderEnv`, robosuite 1.4.1, MuJoCo 3.2.3,
MUJOCO_GL=osmesa on the RTX 5090 WSL host (D4). Environment, settling, image handling and state vector reuse the
frozen `client_openpi.py` helpers unchanged. Differences from the frozen rollout:
- every request carries `__request_id__` and `__tick__`, and every served chunk's executed-step attestation is
  verified BEFORE its first action is applied (D5); a failure raises InstrumentIncident;
- the v1 stop rule (server echo, shape, finiteness, chunk length) raises ProtocolStop, as in v1;
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


class Pi05ClientAdapter(object):
    model = "pi05"

    def __init__(self, states_path, port):
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
            return self._rollout(context, env, prompt, state, ident, h, d)
        finally:
            env.close()

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
                # v1 stop_rule_pi05, unchanged (02 B5).
                if response.get("server_configured_num_steps") != d:
                    raise ProtocolStop("unsupported_refinement_setting: server did not attest configured d={}".format(d))
                chunk = np.asarray(response["actions"])
                if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < h or not np.isfinite(chunk).all():
                    raise ProtocolStop("unsupported_refinement_setting: chunk {} finite={}".format(
                        chunk.shape, bool(np.isfinite(chunk).all())))
                if response.get("received_num_steps") != d:
                    raise ProtocolStop("unsupported_refinement_setting: sampler received {}".format(
                        response.get("received_num_steps")))
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
                            "termination_reason": "success" if done else "task_horizon",
                            "success_source": "LIBERO done (OffScreenRenderEnv.step)"},
                "settled_state_sha256": settled, "first_observation_sha256": first_observation,
                "action_provenance": provenance, "action_array": array_path.name,
                "timing": {"rollout_wall_s": time.perf_counter() - started}}

    def close(self):
        return None
