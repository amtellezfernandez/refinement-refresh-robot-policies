W=$HOME/alba-cross-policy-20261101; OUT=$HOME/jobs/alba-v3-pi05-cuda-probe; mkdir -p $OUT
exec 9>$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo "gpu.lock held"; exit 3; }
echo "== alloc recheck"; timeout 120 $W/envs/smolvla/bin/python -c "
import torch;x=torch.empty(int(20e9)//4,device='cuda');x.fill_(1.0);print('alloc_20GB_all_ones',bool((x==1).all().item()),'numel',x.numel());del x;torch.cuda.empty_cache()"
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 6 ]; then echo "WATCHDOG: MemAvailable ${a} GiB, killing probe"; pkill -f alba-v3-pi05-marker; fi; done ) & WD=$!
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 OUTF=$OUT/result.json
cd $W/src/openpi; echo "openpi_commit $(git rev-parse HEAD 2>/dev/null)"
taskset -c 0-2 timeout 3600 $W/envs/openpi/bin/python - alba-v3-pi05-marker <<'PYEOF' 2>&1 | grep -v -E 'WARNING|slow_operation' | tail -30 | cut -c1-260
import hashlib, json, os, time, numpy as np, jax
from openpi.policies import policy_config, libero_policy
from openpi.training import config as C
W = os.environ["HOME"] + "/alba-cross-policy-20261101"; ck = W + "/ckpt/checkpoints/pi05_libero"; OUTF = os.environ["OUTF"]
out = {"devices": [str(x) for x in jax.devices()], "jax": jax.__version__}
def dump(): json.dump(out, open(OUTF, "w"), indent=1)
norm = ck + "/assets/physical-intelligence/libero/norm_stats.json"
out["norm_sha256"] = hashlib.sha256(open(norm, "rb").read()).hexdigest()
t = time.time(); p = policy_config.create_trained_policy(C.get_config("pi05_libero"), ck); out["load_s"] = time.time() - t; print("loaded", round(out["load_s"],1), out["devices"], flush=True)
un = [x for x in p._output_transform.transforms if type(x).__name__ == "Unnormalize"]; exp = json.load(open(norm))["norm_stats"]
out["unnormalize_found"] = len(un); out["use_quantiles"] = bool(un[0].use_quantiles)
out["fields_compared"] = sum(1 for g, s in exp.items() for f in s)
out["loaded_stats_equal_file"] = all(np.array_equal(np.asarray(getattr(un[0].norm_stats[g], f), dtype=np.float64), np.asarray(v, dtype=np.float64)) for g, s in exp.items() for f, v in s.items())
# negative control for the equality check: perturb one value, must be False
g0 = next(iter(exp)); f0 = next(iter(exp[g0])); pert = np.asarray(exp[g0][f0], dtype=np.float64).copy(); pert.flat[0] += 1e-6
out["negctl_perturbed_equal"] = bool(np.array_equal(np.asarray(getattr(un[0].norm_stats[g0], f0), dtype=np.float64), pert))
print({k: out[k] for k in ("norm_sha256", "unnormalize_found", "use_quantiles", "fields_compared", "loaded_stats_equal_file", "negctl_perturbed_equal")}, flush=True); dump()
m = p._model; tgt = m.action_out_proj; L = type(tgt); orig = L.__call__; n = [0]; jitted = p._sample_actions
def counting(self, *a, **k):
    if self is tgt: n[0] += 1
    return orig(self, *a, **k)
L.__call__ = counting; p._sample_actions = lambda r, o, **k: m.sample_actions(r, o, **k)
ex = libero_policy.make_libero_example(); noise = np.random.default_rng(20261102).standard_normal((10, 32), dtype=np.float32); out["points"] = {}
with jax.disable_jit():
    for d in (2, 5, 10):
        n[0] = 0; p._sample_kwargs = {"num_steps": d}; t = time.time(); a = np.asarray(p.infer(dict(ex), noise=noise)["actions"])
        out["points"][d] = {"loop_calls": n[0], "shape": list(a.shape), "finite": bool(np.isfinite(a).all()), "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "min": a.min(0).tolist(), "max": a.max(0).tolist(), "wall_s": time.time() - t}
        out["points"][d]["_a"] = a.tolist()
        print("d", d, {k: out["points"][d][k] for k in ("loop_calls", "shape", "finite", "wall_s")}, flush=True); dump()
    n[0] = 0; p._sample_kwargs = {}; p.infer(dict(ex), noise=noise); out["negctl_no_num_steps_loop_calls"] = n[0]; print("negctl default loop_calls", n[0], flush=True)
out["distinct_chunks"] = len({v["sha256"] for v in out["points"].values()}); print("distinct_chunks", out["distinct_chunks"], flush=True)
L.__call__ = orig; p._sample_actions = jitted; out["jit"] = {}
for d in (2, 5, 10):
    p._sample_kwargs = {"num_steps": d}; ts = []
    for i in range(6):
        t = time.time(); a = np.asarray(p.infer(dict(ex), noise=noise)["actions"]); ts.append(time.time() - t)
    ref = np.asarray(out["points"][d]["_a"]); out["jit"][d] = {"first_call_s": ts[0], "warm_median_s": float(np.median(ts[1:])), "max_abs_diff_vs_nojit": float(np.abs(a - ref).max())}
    print("jit d", d, out["jit"][d], flush=True); dump()
for d in out["points"]: out["points"][d].pop("_a")
dump(); print("DONE", flush=True)
PYEOF
kill $WD; sha256sum $OUTF; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
