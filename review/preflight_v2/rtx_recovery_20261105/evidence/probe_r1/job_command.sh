set -o pipefail
W=$HOME/alba-cross-policy-20261101; OUT=$HOME/jobs/alba-v3-pi05-cuda-probe-r1; mkdir -p $OUT
exec 9>$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo "gpu.lock held"; exit 3; }
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 6 ]; then echo "WATCHDOG: MemAvailable ${a} GiB"; pkill -f alba-v3-pi05-marker; fi; done ) & WD=$!
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 OUT
cd $W/src/openpi
cat > $OUT/probe.py <<'PYEOF'
import hashlib, json, os, sys, time, numpy as np, jax
from openpi.policies import policy_config, libero_policy
from openpi.training import config as C
MODE = sys.argv[2]; W = os.environ["HOME"] + "/alba-cross-policy-20261101"; ck = W + "/ckpt/checkpoints/pi05_libero"; OUTF = os.environ["OUT"] + f"/result_{MODE}.json"
out = {"mode": MODE, "devices": [str(x) for x in jax.devices()]}
def mem(): s = jax.devices()[0].memory_stats() or {}; return {k: round(s.get(k, 0) / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}
p = policy_config.create_trained_policy(C.get_config("pi05_libero"), ck); out["mem_after_load"] = mem()
m = p._model; tgt = m.action_out_proj; L = type(tgt); orig = L.__call__; n = [0]; jitted = p._sample_actions
def counting(self, *a, **k):
    if self is tgt: n[0] += 1
    return orig(self, *a, **k)
ex = libero_policy.make_libero_example(); noise = np.random.default_rng(20261102).standard_normal((10, 32), dtype=np.float32)
if MODE in ("d5", "d10", "neg"):
    L.__call__ = counting; p._sample_actions = lambda r, o, **k: m.sample_actions(r, o, **k)
    p._sample_kwargs = {} if MODE == "neg" else {"num_steps": int(MODE[1:])}
    with jax.disable_jit():
        t = time.time(); a = np.asarray(p.infer(dict(ex), noise=noise)["actions"])
    out.update(loop_calls=n[0], shape=list(a.shape), finite=bool(np.isfinite(a).all()), sha256=hashlib.sha256(a.tobytes()).hexdigest(), wall_s=time.time() - t, min=a.min(0).tolist(), max=a.max(0).tolist())
    np.save(os.environ["OUT"] + f"/actions_{MODE}.npy", a)
else:
    out["jit"] = {}
    for d in (2, 5, 10):
        p._sample_kwargs = {"num_steps": d}; ts = []
        for i in range(6):
            t = time.time(); a = np.asarray(p.infer(dict(ex), noise=noise)["actions"]); ts.append(time.time() - t)
        out["jit"][d] = {"first_call_s": ts[0], "warm_median_s": float(np.median(ts[1:])), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        np.save(os.environ["OUT"] + f"/actions_jit_d{d}.npy", a)
out["mem_end"] = mem(); json.dump(out, open(OUTF, "w"), indent=1); print("RESULT", json.dumps({k: v for k, v in out.items() if k not in ("min", "max")}), flush=True)
PYEOF
rc_all=0
for MODE in d5 d10 neg jit; do
  echo "== $MODE"; taskset -c 0-2 timeout 1500 $W/envs/openpi/bin/python $OUT/probe.py alba-v3-pi05-marker $MODE 2>&1 | grep -E "RESULT|Error|RESOURCE|Traceback" | cut -c1-600; rc=$?; echo "rc_$MODE=${PIPESTATUS[0]}"
done
kill $WD; sha256sum $OUT/*.json $OUT/*.npy; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
