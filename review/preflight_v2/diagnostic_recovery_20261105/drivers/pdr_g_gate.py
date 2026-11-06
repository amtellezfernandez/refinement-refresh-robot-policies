#!/usr/bin/env python3
"""PDR G01: single-process CODED counting gate prototype (after independent review). NON-COUNTING DIAGNOSTIC.

Changes from B01/F01, each from the review:
  - I1 callback is DATA-DEPENDENT (operand = sum of the head output), so an evaluation must actually run to be counted
  - I1 is compiled into the SERVED object (loaded._sample_actions, traced on first call), not a re-jit
  - plain comparator = a separate uninstrumented module_jit in the same process; must be bitwise equal to served
  - reference R(k): k a Python constant (static fori_loop, dt=-1/k folded), independent of traced num_steps plumbing
  - CNT-2 neighbours: R(d-1), R(d+1) must NOT equal served output (bitwise)
  - all mutants in ONE process; coded verdict; exit 1 if any verdict is wrong
Mutants: M1 ignore num_steps; M2 d+1; M3 wrong dt; M5 head twice; M6 NFEs skipped; K evaluate each step but use the
first velocity; W1 last step uses stale backbone output (head re-applied).
Run with XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0". Seeded input (20261105).
usage: pdr_g_gate.py CODE_DIR CKPT OUT_JSON
"""
import functools, hashlib, json, os, sys, types

C, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import jax, jax.numpy as jnp, numpy as np, einops  # noqa: E402
from flax import nnx  # noqa: E402
import server_openpi as so  # noqa: E402
from openpi.models import pi0 as pi0mod, model as _model  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from openpi.policies.libero_policy import make_libero_example  # noqa: E402

sha = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()  # noqa: E731
np.random.seed(20261105)
example = make_libero_example()
CB = {"on": True, "n": 0}
_orig = nnx.Linear.__call__
SIG = {}


def _tick(_x):
    CB["n"] += 1


def linear_cb(self, x, *a, **kw):
    y = _orig(self, x, *a, **kw)
    if CB["on"] and (self.in_features, self.out_features) == SIG.get("sig"):
        jax.debug.callback(_tick, jnp.sum(y.astype(jnp.float32)))  # data-dependent
    return y


nnx.Linear.__call__ = linear_cb
loaded = so.load(CKPT)
model = loaded._model
SIG["sig"] = (model.action_out_proj.in_features, model.action_out_proj.out_features)
mods = model.iter_modules() if hasattr(model, "iter_modules") else nnx.iter_graph(model)
assert sum(1 for _, m in mods if isinstance(m, nnx.Linear) and (m.in_features, m.out_features) == SIG["sig"]) == 1
R = {"XLA_FLAGS": os.environ.get("XLA_FLAGS", ""), "example_sha": sha(example["observation/state"]), "verdicts": {}}
FK = json.dumps(["fixed-noise"])
DS = (2, 5, 10)

# ---- served object (instrumented at its first trace) ----
CAP = {}
served_fn = loaded._sample_actions


def cap(rng, observation, **kw):
    out = served_fn(rng, observation, **kw)
    CAP["obs"], CAP["x0"] = observation, out
    return out


loaded._sample_actions = cap
keyed = so.KeyedPolicy(loaded)
served, counts, echo = {}, {}, {}
for d in DS:
    CB["n"] = 0
    res = keyed.infer(dict(example, __d__=d, __noise_key__=FK))
    jax.effects_barrier()
    counts[d], served[d], echo[d] = CB["n"], np.asarray(CAP["x0"]), res["received_num_steps"]
OBS = CAP["obs"]
noise = jnp.asarray(so.noise_for(["fixed-noise"]))[None]

# ---- plain comparator (uninstrumented) ----
CB["on"] = False
plain_jit = nnx_utils.module_jit(model.sample_actions)
plain = {d: np.asarray(plain_jit(jax.random.key(0), OBS, num_steps=d, noise=noise)) for d in DS}
plain2 = {d: np.asarray(plain_jit(jax.random.key(0), OBS, num_steps=d, noise=noise)) for d in DS}


# ---- loop builders ----
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
        if kind == "M6":
            v0 = head(backbone(noise, jnp.float32(1.0)))
            body = lambda i, c: (c[0] + dt * v0, c[1] + dt)  # noqa: E731
            init = (noise, jnp.float32(1.0))
        elif kind == "K":
            v0 = head(backbone(noise, jnp.float32(1.0)))

            def body(i, c):
                _ = head(backbone(c[0], c[1]))
                return c[0] + dt * v0, c[1] + dt
            init = (noise, jnp.float32(1.0))
        elif kind == "W1":
            def body(i, c):
                x, t, h = c
                hb = jnp.where(i == n - 1, h, backbone(x, t))  # last step: stale backbone features
                return x + dt * head(hb), t + dt, hb
            init = (noise, jnp.float32(1.0), backbone(noise, jnp.float32(1.0)))  # prologue features (head not applied, not counted)
        else:
            def body(i, c):
                v = head(backbone(c[0], c[1]))
                if kind == "M5":
                    _ = head(backbone(c[0], c[1]))
                return c[0] + dt * v, c[1] + dt
            init = (noise, jnp.float32(1.0))
        out = jax.lax.fori_loop(0, n, body, init)
        return out[0]
    return sample


def run(kind, d, cb, k_static=None):
    CB["on"] = cb
    f = nnx_utils.module_jit(types.MethodType(build(kind, k_static), model))
    CB["n"] = 0
    x = np.asarray(f(jax.random.key(0), OBS, num_steps=d, noise=noise))
    jax.effects_barrier()
    CB["on"] = False
    return CB["n"], x


ref = {k: run("P", 0, False, k_static=k)[1] for k in range(1, 12)}
ref_traced = {d: run("P", d, False)[1] for d in DS}

V = R["verdicts"]
ok = True
for d in DS:
    s = served[d]
    v = {"I1_count": counts[d], "echo": echo[d],
         "served_eq_plain": bool(np.array_equal(s, plain[d])), "plain_repeat": bool(np.array_equal(plain[d], plain2[d])),
         "served_eq_ref_d": bool(np.array_equal(s, ref[d])), "served_max_abs_ref_d": float(np.abs(s - ref[d]).max()),
         "served_eq_ref_traced_d": bool(np.array_equal(s, ref_traced[d])),
         "served_ne_ref_dm1": not np.array_equal(s, ref[d - 1]), "served_ne_ref_dp1": not np.array_equal(s, ref[d + 1])}
    v["PASS"] = (v["I1_count"] == d and v["echo"] == d and v["served_eq_plain"] and v["plain_repeat"]
                 and v["served_eq_ref_d"] and v["served_ne_ref_dm1"] and v["served_ne_ref_dp1"])
    V["stock_d%d" % d] = v
    ok &= v["PASS"]
    for kind in ("M1", "M2", "M3", "M5", "M6", "K", "W1"):
        if (kind in ("M1", "M3")) and d == 10:
            V["%s_d%d" % (kind, d)] = {"not_a_mutant": "identical program to stock at d=10 by construction"}
            continue
        c, x = run(kind, d, True)
        eq = bool(np.array_equal(x, ref[d]))
        gate_pass = (c == d) and eq
        is_correct_program = False
        row = {"I1_count": c, "eq_ref_d": eq, "max_abs_vs_ref_d": float(np.abs(x - ref[d]).max()),
               "gate_pass": gate_pass, "is_correct_program": is_correct_program}
        row["verdict_ok"] = (gate_pass == is_correct_program)
        V["%s_d%d" % (kind, d)] = row
        ok &= row["verdict_ok"]
R["ref_neighbour_min_max_abs"] = {str(d): float(min(np.abs(ref[d] - ref[d - 1]).max(), np.abs(ref[d] - ref[d + 1]).max())) for d in DS}
R["GATE_OK"] = bool(ok)
json.dump(R, open(OUT, "w"), indent=1)
print("STATUS completed GATE_OK=%s" % ok, flush=True)
sys.exit(0 if ok else 1)
