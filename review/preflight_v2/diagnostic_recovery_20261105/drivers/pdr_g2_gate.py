#!/usr/bin/env python3
"""PDR G02: counting gate, revision after review round 2. NON-COUNTING DIAGNOSTIC.

One gate predicate for every sampler (stock, positive control P, mutants), each installed as the policy's
sampler exactly as openpi Policy builds it (module_jit of a bound method) and driven through KeyedPolicy.infer.
gate(sampler, d, case) = I1 count == d  AND  served output bitwise == static-k reference R(d)
                          AND R(d-1), R(d+1) bitwise different  AND  bitwise equal to the same sampler compiled
                          without instrumentation  AND  in-process repeat bitwise.
Echo (received_num_steps) is reported only: it is set by the wrapper and equal by construction.
Expected: stock, P pass; M1 (d<10), M2, M3 (d<10), M5, M6, K, W1 fail.  Two cases (seeded input + noise key).
usage: pdr_g2_gate.py CODE_DIR CKPT OUT_JSON
"""
import hashlib, json, os, sys, types

C, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import jax, jax.numpy as jnp, numpy as np, einops  # noqa: E402
from flax import nnx  # noqa: E402
import server_openpi as so  # noqa: E402
from openpi.models import pi0 as pi0mod, model as _model  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from openpi.policies.libero_policy import make_libero_example  # noqa: E402

sha = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()  # noqa: E731
CB = {"on": False, "n": 0}
_orig = nnx.Linear.__call__
SIG = {}


def _tick(_x):
    CB["n"] += 1


def linear_cb(self, x, *a, **kw):
    y = _orig(self, x, *a, **kw)
    if CB["on"] and (self.in_features, self.out_features) == SIG.get("sig"):
        jax.debug.callback(_tick, jnp.sum(y.astype(jnp.float32)))
    return y


nnx.Linear.__call__ = linear_cb
loaded = so.load(CKPT)
model = loaded._model
SIG["sig"] = (model.action_out_proj.in_features, model.action_out_proj.out_features)
mods = model.iter_modules() if hasattr(model, "iter_modules") else nnx.iter_graph(model)
assert sum(1 for _, m in mods if isinstance(m, nnx.Linear) and (m.in_features, m.out_features) == SIG["sig"]) == 1
DS = (2, 5, 10)


def build(kind, k_static=None):
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
        if kind == "K":  # head evaluated every step (count d); the step-0 velocity is used throughout
            def body(i, c):
                v = head(backbone(c[0], c[1]))
                vu = jnp.where(i == 0, v, c[2])
                return c[0] + dt * vu, c[1] + dt, vu
            return jax.lax.fori_loop(0, n, body, (x0, jnp.float32(1.0), jnp.zeros_like(x0)))[0]
        if kind == "W1":  # last step uses the previous step's backbone features
            def body(i, c):
                hb = jnp.where(i == n - 1, c[2], backbone(c[0], c[1]))
                return c[0] + dt * head(hb), c[1] + dt, hb
            return jax.lax.fori_loop(0, n, body, (x0, jnp.float32(1.0), backbone(x0, jnp.float32(1.0))))[0]

        def body(i, c):
            v = head(backbone(c[0], c[1]))
            if kind == "M5":
                _ = head(backbone(c[0], c[1]))
            return c[0] + dt * v, c[1] + dt
        return jax.lax.fori_loop(0, n, body, (x0, jnp.float32(1.0)))[0]
    return sample


def method(kind):
    return model.sample_actions if kind == "stock" else types.MethodType(build(kind), model)


def serve_with(kind, cb):
    """Install a sampler as Policy does (module_jit of a bound method) and wrap it like the server."""
    CB["on"] = cb
    jitted = nnx_utils.module_jit(method(kind))
    cap = {}

    def capturing(rng, observation, **kw):
        out = jitted(rng, observation, **kw)
        cap["obs"], cap["x0"] = observation, out
        return out
    loaded._sample_actions, loaded._sample_kwargs = capturing, {}
    return so.KeyedPolicy(loaded), cap


CASES = {}
for name, seed, nk in (("A", 20261105, ["fixed-noise"]), ("B", 20261106, ["fixed-noise-2"])):
    np.random.seed(seed)
    ex = make_libero_example()
    CASES[name] = (ex, json.dumps(nk), {k: (sha(v) if not isinstance(v, str) else v) for k, v in ex.items()})

R = {"XLA_FLAGS": os.environ.get("XLA_FLAGS", ""), "cases": {k: v[2] for k, v in CASES.items()}, "rows": {}, "hashes": {}}
EXPECT_PASS = {"stock", "P"}
ok = True
for cname, (ex, nk, _) in CASES.items():
    # references: static k, uninstrumented, on this case's captured observation and noise
    kp, cap = serve_with("stock", False)
    kp.infer(dict(ex, __d__=2, __noise_key__=nk))
    obs = cap["obs"]
    noise = jnp.asarray(so.noise_for(json.loads(nk)))[None]
    CB["on"] = False
    ref = {k: np.asarray(nnx_utils.module_jit(types.MethodType(build("P", k_static=k), model))(jax.random.key(0), obs, noise=noise))
           for k in range(1, 12)}
    for kind in ("stock", "P", "M1", "M2", "M3", "M5", "M6", "K", "W1"):
        kp_u, cap_u = serve_with(kind, False)
        unin = {}
        for d in DS:
            kp_u.infer(dict(ex, __d__=d, __noise_key__=nk))
            unin[d] = np.asarray(cap_u["x0"])
        kp_i, cap_i = serve_with(kind, True)
        for d in DS:
            CB["n"] = 0
            res = kp_i.infer(dict(ex, __d__=d, __noise_key__=nk))
            jax.effects_barrier()
            x, c = np.asarray(cap_i["x0"]), CB["n"]
            kp_i.infer(dict(ex, __d__=d, __noise_key__=nk))
            jax.effects_barrier()
            x2 = np.asarray(cap_i["x0"])
            row = {"I1_count": c, "echo_tautological": res["received_num_steps"],
                   "eq_ref_d": bool(np.array_equal(x, ref[d])), "max_abs_vs_ref_d": float(np.abs(x - ref[d]).max()),
                   "ne_ref_dm1": not np.array_equal(x, ref[d - 1]), "ne_ref_dp1": not np.array_equal(x, ref[d + 1]),
                   "eq_uninstrumented": bool(np.array_equal(x, unin[d])), "repeat": bool(np.array_equal(x, x2))}
            row["GATE_PASS"] = (c == d and row["eq_ref_d"] and row["ne_ref_dm1"] and row["ne_ref_dp1"]
                                and row["eq_uninstrumented"] and row["repeat"])
            correct = kind in EXPECT_PASS or (kind in ("M1", "M3") and d == 10)
            row["expected_pass"] = correct
            row["verdict_ok"] = row["GATE_PASS"] == correct
            ok &= row["verdict_ok"]
            R["rows"]["%s_%s_d%d" % (cname, kind, d)] = row
            if kind == "stock":
                R["hashes"]["%s_d%d" % (cname, d)] = sha(x)
        print("STATUS", cname, kind, [R["rows"]["%s_%s_d%d" % (cname, kind, d)]["GATE_PASS"] for d in DS], flush=True)
R["GATE_OK"] = bool(ok)
json.dump(R, open(OUT, "w"), indent=1)
print("STATUS completed GATE_OK=%s" % ok, flush=True)
sys.exit(0 if ok else 1)
