import hashlib, json, os, platform, time
import numpy as np, jax
from openpi.policies import policy_config, libero_policy
from openpi.training import config as C
B = os.environ["HOME"] + "/alba-macprobe-20261102"; ck = B + "/ckpt/checkpoints/pi05_libero"; R = B + "/result.json"
out = {"status": "RUNNING", "host": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "jax": jax.__version__, "devices": [str(x) for x in jax.devices()], "started_unix": time.time(), "points": {}}
save = lambda: json.dump(out, open(R, "w"), indent=1)
norm = ck + "/assets/physical-intelligence/libero/norm_stats.json"
out["norm_sha256"] = hashlib.sha256(open(norm, "rb").read()).hexdigest(); save()
t = time.time(); p = policy_config.create_trained_policy(C.get_config("pi05_libero"), ck); out["load_s"] = time.time() - t; print("loaded", out["load_s"], flush=True)
un = [x for x in p._output_transform.transforms if type(x).__name__ == "Unnormalize"]; exp = json.load(open(norm))["norm_stats"]
out["unnormalize_found"] = len(un); out["use_quantiles"] = bool(un[0].use_quantiles)
out["normalization_fields_compared"] = {g + "." + f: bool(np.array_equal(np.asarray(getattr(un[0].norm_stats[g], f), dtype=np.float64), np.asarray(v, dtype=np.float64))) for g, s in exp.items() for f, v in s.items()}
out["loaded_stats_equal_file"] = all(out["normalization_fields_compared"].values()); save()
print({k: out[k] for k in ("norm_sha256", "unnormalize_found", "use_quantiles", "loaded_stats_equal_file")}, flush=True)
m = p._model; tgt = m.action_out_proj; L = type(tgt); orig = L.__call__; n = [0]
def counting(self, *a, **k):
    if self is tgt: n[0] += 1
    return orig(self, *a, **k)
L.__call__ = counting; p._sample_actions = lambda r, o, **k: m.sample_actions(r, o, **k)
ex = libero_policy.make_libero_example(); noise = np.random.default_rng(20261102).standard_normal((10, 32), dtype=np.float32)
with jax.disable_jit():
    for d in (2, 5, 10):
        n[0] = 0; p._sample_kwargs = {"num_steps": d}; t = time.time(); a = np.asarray(p.infer(dict(ex), noise=noise)["actions"])
        out["points"][str(d)] = {"loop_calls": n[0], "shape": list(a.shape), "finite": bool(np.isfinite(a).all()), "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "min": a.min(0).tolist(), "max": a.max(0).tolist(), "wall_s": time.time() - t}
        save(); print("d", d, {k: out["points"][str(d)][k] for k in ("loop_calls", "shape", "finite", "wall_s")}, flush=True)
out["distinct_chunks"] = len({v["sha256"] for v in out["points"].values()}); out["status"] = "COMPLETED"; out["finished_unix"] = time.time(); save(); print("distinct", out["distinct_chunks"], "COMPLETED", flush=True)
