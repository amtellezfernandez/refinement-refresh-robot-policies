#!/usr/bin/env python3
"""pi0.5 attesting policy server (D5; 09 C2, C4). Python 3.11, OpenPI env. NEVER RUN on any host.

Wraps the frozen `server_openpi.py` (load, KeyedPolicy, noise_for, count_denoise_loop), imported unchanged.
1. Instruments the served compiled sampler with two data-dependent jax.debug.callbacks, at trace time:
   the output of `action_out_proj` (G02 counting site; signature-unique, asserted) and the time operand of
   `embed_suffix` (time grid).
2. Startup gate on the SERVED object, before the port is bound: the G02 predicate (count = d, bitwise equal to
   a static-k reference, d-1 and d+1 references differ, bitwise equal to an uninstrumented compile [PRES],
   in-process repeat) for stock, P and the mutants M1 M2 M3 M5 M6 K W1 on cases A and B at d = 2, 5, 10;
   CNT-IND (eager count under disable_jit via the frozen count_denoise_loop); the time grid for stock.
   Any wrong verdict: exit 1, port never bound.
3. Serves through the frozen WebsocketPolicyServer. Every response carries an attestation built from the
   callback counters only (attestation.build), bound to the request's __request_id__ and __tick__.

Modes:
    serve    --checkpoint C --schedule S       Protocol B (Alba's release + qualification + reservation)
    serve    --checkpoint C --authorisation F  Protocol A (Alba's authorisation, scope PROTOCOL_A_PI05_SERVER)
    gate     --checkpoint C --authorisation F --output DIR   startup gate only, record written, no port
    overhead --checkpoint C --authorisation F --output DIR   CNT-T attestation overhead, recorded, no threshold
The deterministic XLA flags (D8) must be in the environment before JAX is imported; otherwise it refuses.

The sampler builder below is copied from the G02 prototype
(review/preflight_v2/diagnostic_recovery_20261105/drivers/pdr_g2_gate.py, sha256 e533a605...); the faithful
sampler P and the mutants are unchanged from it.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attestation as A  # noqa: E402
from v3common import REPO_ROOT, V2, atomic_write_json, file_sha  # noqa: E402

CERTIFIED_XLA_FLAGS = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
PROTOCOL_A_DOCUMENT = REPO_ROOT / "review/preflight_v2/protocol_v3_revision_20261106/01_PREFLIGHT_V3_DRAFT_REV2.md"
DS = (2, 5, 10)
KINDS = ("stock", "P", "M1", "M2", "M3", "M5", "M6", "K", "W1")
EXPECT_PASS = ("stock", "P")
CASES = (("A", 20261105, ["fixed-noise"]), ("B", 20261106, ["fixed-noise-2"]))

TRACE = {"on": True}            # read at TRACE time: decides whether a compile carries the callbacks
COUNTS = {"head": 0, "times": []}
SIG = {}


def require_flags():
    if os.environ.get("XLA_FLAGS") != CERTIFIED_XLA_FLAGS:
        raise SystemExit("REFUSED: XLA_FLAGS must be exactly {!r} (D8)".format(CERTIFIED_XLA_FLAGS))


def _head_tick(_value):
    COUNTS["head"] += 1


def _time_tick(value):
    import numpy as np
    COUNTS["times"].append(float(np.asarray(value, dtype=np.float32).reshape(-1)[0]))


def install_instrumentation(model):
    """Class-level patches, active only in compiles traced while TRACE['on'] is True."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    original_linear = nnx.Linear.__call__
    model_class = type(model)
    original_embed = model_class.embed_suffix

    def linear(self, x, *args, **kwargs):
        y = original_linear(self, x, *args, **kwargs)
        if TRACE["on"] and (self.in_features, self.out_features) == SIG["sig"]:
            jax.debug.callback(_head_tick, jnp.sum(y.astype(jnp.float32)))
        return y

    def embed_suffix(self, *args, **kwargs):
        if TRACE["on"]:
            timestep = args[2] if len(args) > 2 else kwargs["timestep"]
            jax.debug.callback(_time_tick, jnp.reshape(timestep, (-1,))[0].astype(jnp.float32))
        return original_embed(self, *args, **kwargs)

    SIG["sig"] = (model.action_out_proj.in_features, model.action_out_proj.out_features)
    modules = model.iter_modules() if hasattr(model, "iter_modules") else nnx.iter_graph(model)
    matches = sum(1 for _, m in modules if isinstance(m, nnx.Linear) and (m.in_features, m.out_features) == SIG["sig"])
    if matches != 1:
        raise SystemExit("REFUSED: action_out_proj signature is not unique ({} matches); counting would be ambiguous"
                         .format(matches))
    nnx.Linear.__call__ = linear
    model_class.embed_suffix = embed_suffix


def build(model, kind, k_static=None):
    """G02 prototype sampler builder (unchanged logic): P is faithful; the others are mutants."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models import pi0 as pi0mod

    def sample(self, rng, observation, *, num_steps=10, noise=None):
        observation = _model.preprocess_observation(None, observation, train=False)
        bsz = observation.state.shape[0]
        pt, pm, par = self.embed_prefix(observation)
        _, kv = self.PaliGemma.llm([pt, None], mask=pi0mod.make_attn_mask(pm, par), positions=jnp.cumsum(pm, axis=1) - 1)

        def backbone(x_t, t):
            st, sm, sar, ada = self.embed_suffix(observation, x_t, jnp.broadcast_to(t, bsz))
            fam = jnp.concatenate([einops.repeat(pm, "b p -> b s p", s=st.shape[1]), pi0mod.make_attn_mask(sm, sar)], axis=-1)
            pos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
            (_, so_), _ = self.PaliGemma.llm([None, st], mask=fam, positions=pos, kv_cache=kv, adarms_cond=[None, ada])
            return so_[:, -self.action_horizon:]

        head = self.action_out_proj
        n = k_static if k_static is not None else num_steps
        n = {"M1": 10, "M2": n + 1}.get(kind, n)
        dt = (-1.0 / 10) if kind == "M3" else (-1.0 / n)
        x0 = noise
        if kind == "M6":
            v0 = head(backbone(x0, jnp.float32(1.0)))
            return jax.lax.fori_loop(0, n, lambda i, c: (c[0] + dt * v0, c[1] + dt), (x0, jnp.float32(1.0)))[0]
        if kind == "K":
            def body_k(i, c):
                v = head(backbone(c[0], c[1]))
                vu = jnp.where(i == 0, v, c[2])
                return c[0] + dt * vu, c[1] + dt, vu
            return jax.lax.fori_loop(0, n, body_k, (x0, jnp.float32(1.0), jnp.zeros_like(x0)))[0]
        if kind == "W1":
            def body_w(i, c):
                hb = jnp.where(i == n - 1, c[2], backbone(c[0], c[1]))
                return c[0] + dt * head(hb), c[1] + dt, hb
            return jax.lax.fori_loop(0, n, body_w, (x0, jnp.float32(1.0), backbone(x0, jnp.float32(1.0))))[0]

        def body(i, c):
            v = head(backbone(c[0], c[1]))
            if kind == "M5":
                _ = head(backbone(c[0], c[1]))
            return c[0] + dt * v, c[1] + dt
        return jax.lax.fori_loop(0, n, body, (x0, jnp.float32(1.0)))[0]
    return types.MethodType(sample, model)


def _digest(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()


class Harness(object):
    """Installs a sampler into the loaded policy exactly as Policy does, and drives it through KeyedPolicy."""

    def __init__(self, so, loaded, served_jitted):
        self.so, self.loaded, self.served = so, loaded, served_jitted

    def keyed_with(self, jitted):
        captured = {}

        def capturing(rng, observation, **kwargs):
            out = jitted(rng, observation, **kwargs)
            captured["obs"], captured["x0"] = observation, out
            return out
        self.loaded._sample_actions, self.loaded._sample_kwargs = capturing, {}
        return self.so.KeyedPolicy(self.loaded), captured

    def jit(self, kind, instrumented):
        from openpi.shared import nnx_utils
        TRACE["on"] = instrumented
        method = self.loaded._model.sample_actions if kind == "stock" else build(self.loaded._model, kind)
        return nnx_utils.module_jit(method)

    def run(self, keyed, captured, example, d, noise_key, instrumented):
        """One call; with instrumented=True also returns (head count, visited times) for this call only."""
        import jax
        import numpy as np
        TRACE["on"] = instrumented
        COUNTS["head"], COUNTS["times"] = 0, []
        keyed.infer(dict(example, __d__=d, __noise_key__=noise_key))
        jax.effects_barrier()
        return np.asarray(captured["x0"]), COUNTS["head"], list(COUNTS["times"])


def startup_gate(so, loaded, served_jitted, checkpoint):
    """09 C2 / C4 startup gate on the served object. Returns (record, ok)."""
    import jax
    import numpy as np
    from openpi.policies.libero_policy import make_libero_example
    from openpi.shared import nnx_utils
    harness = Harness(so, loaded, served_jitted)
    model = loaded._model
    record = {"xla_flags": os.environ.get("XLA_FLAGS"), "jax": jax.__version__, "devices": [str(x) for x in jax.local_devices()],
              "checkpoint": str(checkpoint), "server_code_sha256": file_sha(__file__),
              "server_openpi_sha256": file_sha(V2 / "server_openpi.py"), "cases": {}, "rows": {}, "time_grid": {},
              "cnt_ind": {}}
    ok = True
    for name, seed, key in CASES:
        np.random.seed(seed)
        example = make_libero_example()
        noise_key = json.dumps(key)
        record["cases"][name] = {k: (_digest(v) if not isinstance(v, str) else v) for k, v in example.items()}
        keyed, captured = harness.keyed_with(harness.jit("stock", False))
        keyed.infer(dict(example, __d__=2, __noise_key__=noise_key))
        observation = captured["obs"]
        noise = jax.numpy.asarray(so.noise_for(json.loads(noise_key)))[None]
        TRACE["on"] = False
        reference = {k: np.asarray(nnx_utils.module_jit(build(model, "P", k_static=k))(jax.random.key(0), observation,
                                                                                         noise=noise))
                     for k in range(1, 12)}
        for kind in KINDS:
            keyed_u, captured_u = harness.keyed_with(harness.jit(kind, False))
            uninstrumented = {d: harness.run(keyed_u, captured_u, example, d, noise_key, False)[0] for d in DS}
            # The served object is gated as itself: "stock" uses served_jitted, never a re-jit.
            jitted = served_jitted if kind == "stock" else harness.jit(kind, True)
            keyed_i, captured_i = harness.keyed_with(jitted)
            for d in DS:
                x, count, times = harness.run(keyed_i, captured_i, example, d, noise_key, True)
                x2 = harness.run(keyed_i, captured_i, example, d, noise_key, True)[0]
                row = {"count": count, "eq_ref_d": bool(np.array_equal(x, reference[d])),
                       "ne_ref_dm1": not np.array_equal(x, reference[d - 1]),
                       "ne_ref_dp1": not np.array_equal(x, reference[d + 1]),
                       "eq_uninstrumented": bool(np.array_equal(x, uninstrumented[d])),
                       "repeat": bool(np.array_equal(x, x2))}
                row["GATE_PASS"] = (count == d and row["eq_ref_d"] and row["ne_ref_dm1"] and row["ne_ref_dp1"]
                                    and row["eq_uninstrumented"] and row["repeat"])
                expected = kind in EXPECT_PASS or (kind in ("M1", "M3") and d == 10)
                row["expected_pass"], row["verdict_ok"] = expected, row["GATE_PASS"] == expected
                ok &= row["verdict_ok"]
                if kind == "stock":
                    grid = sorted(A.f32_hex(t) for t in times)
                    grid_ok = grid == sorted(A.f32_hex(t) for t in A.expected_time_values(d))
                    record["time_grid"]["{}_d{}".format(name, d)] = {"ok": grid_ok, "visited": grid}
                    ok &= grid_ok
                    ok &= row["eq_uninstrumented"]  # PRES on the served object
                record["rows"]["{}_{}_d{}".format(name, kind, d)] = row
    # CNT-IND: a different instrument on a different program (eager, disable_jit), case A only (SC1).
    TRACE["on"] = False
    np.random.seed(CASES[0][1])
    loaded._sample_actions, loaded._sample_kwargs = served_jitted, {}
    keyed = so.KeyedPolicy(loaded)
    counts = so.count_denoise_loop(keyed, make_libero_example())
    record["cnt_ind"] = counts
    ok &= all(counts.get(str(d)) == d for d in DS)
    TRACE["on"] = True
    record["GATE_OK"] = bool(ok)
    return record, bool(ok)


class AttestingPolicy(object):
    """BasePolicy-compatible. Pops __request_id__ and __tick__, then calls the frozen KeyedPolicy path."""

    def __init__(self, keyed, startup_gate_sha256):
        self._keyed, self._lock, self._sequence = keyed, threading.Lock(), 0
        self._gate = startup_gate_sha256
        self.metadata = dict(keyed.metadata or {}, startup_gate_sha256=startup_gate_sha256,
                             xla_flags=os.environ.get("XLA_FLAGS"), server_code_sha256=file_sha(__file__))

    def infer(self, obs):
        import jax
        obs = dict(obs)
        request_id, tick = obs.pop("__request_id__", None), obs.pop("__tick__", None)
        d = int(obs["__d__"])
        with self._lock:
            TRACE["on"] = True
            COUNTS["head"], COUNTS["times"] = 0, []
            result = self._keyed.infer(obs)
            jax.effects_barrier()
            self._sequence += 1
            # Counts come only from the compiled callbacks. A missing request id or tick is passed through
            # unchanged, so the client's binding check fails rather than the server inventing a binding.
            result["attestation"] = A.build(request_id, tick if tick is not None else -1, d, COUNTS["head"],
                                            COUNTS["times"], self._sequence, self._gate, os.environ.get("XLA_FLAGS"))
        return result

    def reset(self):
        return None


def load_served(checkpoint):
    import server_openpi as so  # frozen v2 module, imported unchanged
    loaded = so.load(checkpoint)
    install_instrumentation(loaded._model)
    from openpi.shared import nnx_utils
    TRACE["on"] = True
    served = nnx_utils.module_jit(loaded._model.sample_actions)
    return so, loaded, served


def gate_and_record(checkpoint, output):
    so, loaded, served = load_served(checkpoint)
    record, ok = startup_gate(so, loaded, served, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "startup_gate.json"
    if path.exists():
        raise SystemExit("REFUSED: a startup gate record already exists in {}".format(output))
    atomic_write_json(path, record)
    return so, loaded, served, path, ok


def serve(args):
    from openpi.serving import websocket_policy_server
    so, loaded, served, path, ok = gate_and_record(args.checkpoint, args.output)
    if not ok:
        raise SystemExit(1)  # the port is never bound
    loaded._sample_actions, loaded._sample_kwargs = served, {}
    policy = AttestingPolicy(so.KeyedPolicy(loaded), file_sha(path))
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port,
                                                  metadata=policy.metadata).serve_forever()


def overhead(args):
    """CNT-T (D5): per-call host time with and without the attestation callbacks. Recorded, no threshold."""
    import numpy as np
    from openpi.policies.libero_policy import make_libero_example
    so, loaded, served = load_served(args.checkpoint)
    harness = Harness(so, loaded, served)
    np.random.seed(CASES[0][1])
    example = make_libero_example()
    report = {"xla_flags": os.environ.get("XLA_FLAGS"), "points": {}}
    for label, instrumented in (("attested", True), ("plain", False)):
        keyed, captured = harness.keyed_with(served if instrumented else harness.jit("stock", False))
        for d in DS:
            for index in range(100):
                harness.run(keyed, captured, example, d, json.dumps(["warmup", d, index]), instrumented)
            times = []
            for index in range(100):
                started = time.perf_counter()
                harness.run(keyed, captured, example, d, json.dumps(["measure", d, index]), instrumented)
                times.append(time.perf_counter() - started)
            report["points"]["{}_d{}".format(label, d)] = {"host_s": times}
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "cnt_t_attestation.json", report)


def main():
    parser = argparse.ArgumentParser(description="pi0.5 attesting server (never run).")
    parser.add_argument("mode", choices=("serve", "gate", "overhead"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--schedule", type=Path, help="Protocol B: the released pi0.5 schedule")
    parser.add_argument("--authorisation", type=Path, help="Protocol A: Alba's authorisation file")
    args = parser.parse_args()
    import release
    if args.mode == "serve" and args.schedule is not None:
        import launch_guard
        import qualification
        release.check_release(args.schedule, "pi05")
        qualification.require_qualified("pi05")
        launch_guard.need_reservation(os.environ)
        if launch_guard.compute_processes():
            raise SystemExit("REFUSED: GPU compute processes present")
    else:
        release.check_authorisation(args.authorisation, PROTOCOL_A_DOCUMENT, {"server_openpi_v3.py": __file__},
                                    scope="PROTOCOL_A_PI05_SERVER")
    require_flags()  # before JAX is imported anywhere below
    if args.mode == "serve":
        serve(args)
    elif args.mode == "gate":
        _, _, _, path, ok = gate_and_record(args.checkpoint, args.output)
        print(json.dumps({"startup_gate": str(path), "GATE_OK": ok}))
        raise SystemExit(0 if ok else 1)
    else:
        overhead(args)


if __name__ == "__main__":
    main()
