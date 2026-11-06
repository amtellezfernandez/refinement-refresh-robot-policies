#!/usr/bin/env python3
"""PDR Phases B+D: compiled-path step counting with negative controls; stock vs wrapper. NON-COUNTING DIAGNOSTIC.

Synthetic input only (openpi make_libero_example). Frozen server_openpi imported unchanged.
Instruments on the COMPILED (jit) path:
  I1 jax.debug.callback inside action_out_proj (identified by signature, asserted unique) -> executed-step count
  I2 output identification: deployed raw x0 vs a separately jitted explicit k-step integrator R(k, dt)
Variants: plain (uninstrumented), cb (I1 on), M1 (ignores num_steps: always 10), M2 (runs num_steps+1).
Phase D: stock Policy.infer(obs, noise=n) with config default sample kwargs vs KeyedPolicy.infer(d=10, same n).
usage: pdr_b_compiled.py CODE_DIR CKPT OUT_JSON
"""
import hashlib, json, sys, time, types

C, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import jax, jax.numpy as jnp, numpy as np, einops  # noqa: E402
from flax import nnx  # noqa: E402
import server_openpi as so  # noqa: E402
from openpi.models import pi0 as pi0mod, model as _model  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from openpi.policies.libero_policy import make_libero_example  # noqa: E402

sha = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()  # noqa: E731
R = {"jax": jax.__version__, "device": str(jax.local_devices()[0])}
loaded = so.load(CKPT)
model = loaded._model
STOCK_SAMPLER = loaded._sample_actions
R["stock_sample_kwargs"] = dict(loaded._sample_kwargs)
R["sample_actions_default_num_steps"] = pi0mod.Pi0.sample_actions.__kwdefaults__.get("num_steps")
R["input_transforms"] = [type(t).__name__ for t in loaded._input_transform.transforms]
R["output_transforms"] = [type(t).__name__ for t in loaded._output_transform.transforms]
example = make_libero_example()
FIXED = so.noise_for(["fixed-noise"])

# ---------------- Phase D: stock vs wrapper (before any patching) ----------------
stock = loaded.infer(dict(example), noise=FIXED)
stock2 = loaded.infer(dict(example), noise=FIXED)
stock_rng = loaded.infer(dict(example))
keyed = so.KeyedPolicy(loaded)
wrap = keyed.infer(dict(example, __d__=10, __noise_key__=json.dumps(["fixed-noise"])))
R["D"] = {"stock_keys": sorted(stock), "wrapper_keys": sorted(wrap),
          "stock_shape": list(np.shape(stock["actions"])), "stock_dtype": str(np.asarray(stock["actions"]).dtype),
          "wrapper_shape": list(np.shape(wrap["actions"])),
          "stock_vs_wrapper_d10_bitwise": bool(np.array_equal(stock["actions"], wrap["actions"])),
          "stock_vs_wrapper_max_abs": float(np.abs(np.asarray(stock["actions"]) - np.asarray(wrap["actions"])).max()),
          "stock_repeat_bitwise": bool(np.array_equal(stock["actions"], stock2["actions"])),
          "stock_rng_noise_differs": not bool(np.array_equal(stock["actions"], stock_rng["actions"])),
          "wrapper_received_num_steps": wrap["received_num_steps"], "stock_sha": sha(stock["actions"]),
          "wrapper_sha": sha(wrap["actions"]), "action_range_min": np.asarray(wrap["actions"]).min(0).tolist(),
          "action_range_max": np.asarray(wrap["actions"]).max(0).tolist()}
print("STATUS D_done", R["D"]["stock_vs_wrapper_d10_bitwise"], flush=True)

# ---------------- I1 instrumentation (class-level, toggled at trace time) ----------------
WIDTH, ADIM = model.action_out_proj.in_features, model.action_out_proj.out_features
_mods = model.iter_modules() if hasattr(model, "iter_modules") else nnx.iter_graph(model)
sig = [p for p, m in _mods if isinstance(m, nnx.Linear) and m.in_features == WIDTH and m.out_features == ADIM]
R["I1_signature_matches"] = [str(p) for p in sig]
assert len(sig) == 1, sig
CB = {"on": False, "n": 0}
_orig_linear = nnx.Linear.__call__


def _tick():
    CB["n"] += 1


def linear_cb(self, x, *a, **kw):
    y = _orig_linear(self, x, *a, **kw)
    if CB["on"] and self.in_features == WIDTH and self.out_features == ADIM:
        jax.debug.callback(_tick)
    return y


nnx.Linear.__call__ = linear_cb


def m1(self, rng, observation, *, num_steps=10, noise=None):
    return pi0mod.Pi0.sample_actions(self, rng, observation, num_steps=10, noise=noise)


def m2(self, rng, observation, *, num_steps=10, noise=None):
    return pi0mod.Pi0.sample_actions(self, rng, observation, num_steps=num_steps + 1, noise=noise)


CAP = {}


def build(variant):
    CB["on"] = variant != "plain"
    meth = {"plain": model.sample_actions, "cb": model.sample_actions,
            "M1": types.MethodType(m1, model), "M2": types.MethodType(m2, model)}[variant]
    jitted = nnx_utils.module_jit(meth)

    def capturing(rng, observation, **kw):
        out = jitted(rng, observation, **kw)
        CAP["obs"], CAP["noise"], CAP["x0"] = observation, kw.get("noise"), out
        return out
    loaded._sample_actions = capturing
    loaded._sample_kwargs = {}
    return so.KeyedPolicy(loaded)


V = {}
for variant in ("plain", "cb", "M1", "M2"):
    kp = build(variant)
    rows = {}
    for d in (2, 5, 10):
        reps = []
        for rep in range(2):
            CB["n"] = 0
            t0 = time.perf_counter()
            res = kp.infer(dict(example, __d__=d, __noise_key__=json.dumps(["fixed-noise"])))
            jax.effects_barrier()
            reps.append({"count": CB["n"], "s": time.perf_counter() - t0, "sha": sha(res["actions"]),
                         "raw_sha": sha(CAP["x0"]), "received": res["received_num_steps"]})
            if rep == 0:
                V[(variant, d)] = np.asarray(CAP["x0"])
                if variant == "plain" and d == 2:
                    OBS = CAP["obs"]
        rows[str(d)] = {"reps": reps, "repeat_bitwise": reps[0]["raw_sha"] == reps[1]["raw_sha"]}
    R.setdefault("variants", {})[variant] = rows
    print("STATUS variant", variant, {d: [r["count"] for r in v["reps"]] for d, v in rows.items()}, flush=True)
CB["on"] = False
R["cb_vs_plain_raw_bitwise"] = {str(d): bool(np.array_equal(V[("cb", d)], V[("plain", d)])) for d in (2, 5, 10)}


# ---------------- I2 reference integrator (explicit k, no time-based cond), copied from pinned pi0.py ----------------
def ref_sample(self, observation, noise, k, dt):
    observation = _model.preprocess_observation(None, observation, train=False)
    bsz = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = pi0mod.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def body(i, carry):
        x_t, t = carry
        st, sm, sar, ada = self.embed_suffix(observation, x_t, jnp.broadcast_to(t, bsz))
        sam = pi0mod.make_attn_mask(sm, sar)
        pam = einops.repeat(prefix_mask, "b p -> b s p", s=st.shape[1])
        fam = jnp.concatenate([pam, sam], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, so_), _ = self.PaliGemma.llm([None, st], mask=fam, positions=pos, kv_cache=kv_cache, adarms_cond=[None, ada])
        v_t = self.action_out_proj(so_[:, -self.action_horizon:])
        return x_t + dt * v_t, t + dt
    x0, _ = jax.lax.fori_loop(0, k, body, (noise, jnp.float32(1.0)))
    return x0


ref = nnx_utils.module_jit(types.MethodType(ref_sample, model))
noise_b = jnp.asarray(FIXED)[None]
REF = {}


def r(k, dd):
    key = (k, dd)
    if key not in REF:
        REF[key] = np.asarray(ref(OBS, noise_b, k, jnp.float32(-1.0) / jnp.float32(dd)))
    return REF[key]


I2 = {}
for variant in ("plain", "cb", "M1", "M2"):
    for d in (2, 5, 10):
        x = V[(variant, d)]
        # full integrations with k steps (k=1..12), plus truncated/overshot loops at the requested dt
        cands = {"k%d_dt1/%d" % (k, k): r(k, k) for k in range(1, 13)}
        cands.update({"k%d_dt1/%d" % (k, d): r(k, d) for k in (d - 1, d + 1)})
        dist = {n: float(np.abs(x - c).max()) for n, c in cands.items()}
        best = min(dist, key=dist.get)
        others = sorted(v for n, v in dist.items() if n != best and not np.array_equal(cands[n], cands[best]))
        I2["%s_d%d" % (variant, d)] = {"dist": dist, "identified": best,
                                        "margin_ratio": (others[0] / max(dist[best], 1e-30)) if others else None}
R["I2"] = I2
R["ref_self_check_k_d_vs_plain_bitwise"] = {str(d): bool(np.array_equal(r(d, d), V[("plain", d)])) for d in (2, 5, 10)}
R["status"] = "completed"
json.dump(R, open(OUT, "w"), indent=1, default=str)
print("STATUS completed", flush=True)
