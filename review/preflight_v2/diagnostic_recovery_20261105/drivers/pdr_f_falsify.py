#!/usr/bin/env python3
"""PDR F: attempt to falsify the proposed compiled-path counter (I1) and identifier (I2). NON-COUNTING DIAGNOSTIC.

Seeded synthetic input (np.random.seed(20261105)). Frozen server_openpi imported unchanged.
Mutants built from the pinned pi0.py loop body, run through module_jit + Policy.infer like the real sampler:
  P   faithful re-implementation (fori_loop, d steps, dt=-1/d)           expect I1=d, I2 -> k=d
  M3  d steps, wrong dt (-1/10): right count, wrong integration           I1 cannot see it; I2 must
  M5  action head called twice per step, second result unused            I1 over-counts (2d): safe-direction reject
  M6  network evaluated once, v_t reused for later steps (skipped NFEs)   I1 must report 1
usage: pdr_f_falsify.py CODE_DIR CKPT OUT_JSON
"""
import hashlib, json, sys, types

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
loaded = so.load(CKPT)
model = loaded._model
WIDTH, ADIM = model.action_out_proj.in_features, model.action_out_proj.out_features
CB = {"on": False, "n": 0}
_orig = nnx.Linear.__call__


def _tick():
    CB["n"] += 1


def linear_cb(self, x, *a, **kw):
    y = _orig(self, x, *a, **kw)
    if CB["on"] and self.in_features == WIDTH and self.out_features == ADIM:
        jax.debug.callback(_tick)
    return y


nnx.Linear.__call__ = linear_cb


def body_factory(self, observation, prefix_mask, kv_cache, bsz):
    def v_of(x_t, t, twice=False):
        st, sm, sar, ada = self.embed_suffix(observation, x_t, jnp.broadcast_to(t, bsz))
        sam = pi0mod.make_attn_mask(sm, sar)
        pam = einops.repeat(prefix_mask, "b p -> b s p", s=st.shape[1])
        fam = jnp.concatenate([pam, sam], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, so_), _ = self.PaliGemma.llm([None, st], mask=fam, positions=pos, kv_cache=kv_cache, adarms_cond=[None, ada])
        v = self.action_out_proj(so_[:, -self.action_horizon:])
        if twice:
            _ = self.action_out_proj(so_[:, -self.action_horizon:])
        return v
    return v_of


def make(kind):
    def sample(self, rng, observation, *, num_steps=10, noise=None):
        observation = _model.preprocess_observation(None, observation, train=False)
        bsz = observation.state.shape[0]
        pt, pm, par = self.embed_prefix(observation)
        _, kv = self.PaliGemma.llm([pt, None], mask=pi0mod.make_attn_mask(pm, par), positions=jnp.cumsum(pm, axis=1) - 1)
        v_of = body_factory(self, observation, pm, kv, bsz)
        dt = (-1.0 / 10) if kind == "M3" else (-1.0 / num_steps)
        if kind == "M6":
            v0 = v_of(noise, jnp.float32(1.0))

            def b(i, c):
                return c[0] + dt * v0, c[1] + dt
        else:
            def b(i, c):
                return c[0] + dt * v_of(c[0], c[1], twice=(kind == "M5")), c[1] + dt
        x0, _ = jax.lax.fori_loop(0, num_steps, b, (noise, jnp.float32(1.0)))
        return x0
    return sample


def run(kind, cb):
    CB["on"] = cb
    if kind == "plain":
        jitted = nnx_utils.module_jit(model.sample_actions)
    else:
        jitted = nnx_utils.module_jit(types.MethodType(make(kind), model))
    CAP = {}

    def capturing(rng, observation, **kw):
        out = jitted(rng, observation, **kw)
        CAP["x0"] = out
        return out
    loaded._sample_actions = capturing
    loaded._sample_kwargs = {}
    kp = so.KeyedPolicy(loaded)
    rows = {}
    for d in (2, 5, 10):
        CB["n"] = 0
        kp.infer(dict(example, __d__=d, __noise_key__=json.dumps(["fixed-noise"])))
        jax.effects_barrier()
        rows[d] = (CB["n"], np.asarray(CAP["x0"]))
    return rows


R = {"example_state_sha": sha(example["observation/state"])}
base = run("plain", False)
res = {k: run(k, True) for k in ("P", "M3", "M5", "M6")}
R["I1_counts"] = {k: {str(d): v[d][0] for d in v} for k, v in res.items()}
R["dist_to_plain"] = {k: {str(d): float(np.abs(v[d][1] - base[d][1]).max()) for d in v} for k, v in res.items()}
R["bitwise_plain"] = {k: {str(d): bool(np.array_equal(v[d][1], base[d][1])) for d in v} for k, v in res.items()}
R["plain_neighbour_gap"] = {"2v5": float(np.abs(base[2][1] - base[5][1]).max()), "5v10": float(np.abs(base[5][1] - base[10][1]).max())}
R["status"] = "completed"
json.dump(R, open(OUT, "w"), indent=1)
print("STATUS completed", json.dumps(R["I1_counts"]), flush=True)
