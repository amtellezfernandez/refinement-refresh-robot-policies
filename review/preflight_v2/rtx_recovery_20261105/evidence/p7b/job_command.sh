set -o pipefail
W=$HOME/alba-cross-policy-20261101; C=$W/code-078e166/experiments/tooling/cross_policy; JD=$HOME/jobs/alba-v3-p7b-diag-d5; OUT=$JD/out; mkdir -p $JD
exec 9>$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo GPU_LOCK_HELD; exit 3; }
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 6 ]; then echo "WATCHDOG_FIRED MemAvailable=${a}GiB $(date -u +%T)" >> $JD/watchdog.txt; pkill -f p7_diag.py; fi; done ) & WD=$!
( while sleep 3; do echo "$(date -u +%T) memavail_gib=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $JD/resources.txt; done ) & RS=$!
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$W/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false ALBA_GPU_RESERVATION=R017

env | grep -E '^(XLA|JAX|CUDA|OMP|MKL|OPENBLAS|HF_|TRANSFORMERS|ALBA|LD_LIB|MUJOCO|PYOPENGL|CUBLAS)' | sort > $JD/env.txt
cat $C/../../../BUNDLE_COMMIT 2>/dev/null; sha256sum $C/server_openpi.py $C/common.py $C/launch_guard.py $C/provenance.py > $JD/code_sha256.txt
echo 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 | base64 -d | gunzip > $JD/p7_diag.py; echo "driver_sha256 $(sha256sum $JD/p7_diag.py | cut -c1-64) expected f92586cfda406a9c78b942f400f1789f72a9d7a705bbee33b69efe3bdf6c35e5"
[ "$(sha256sum $JD/p7_diag.py | cut -c1-64)" = "f92586cfda406a9c78b942f400f1789f72a9d7a705bbee33b69efe3bdf6c35e5" ] || { echo DRIVER_HASH_MISMATCH; exit 7; }
mkdir -p $OUT
taskset -c 0-2 timeout 2400 $W/envs/openpi/bin/python $JD/p7_diag.py $C $W/ckpt/checkpoints/pi05_libero $OUT/diag.json 5 > $JD/python.log 2>&1
rc=$?; echo "python_rc=$rc" > $JD/rc.txt; cat $JD/rc.txt
kill $WD $RS 2>/dev/null
grep -E "STATUS|RESOURCE_EXHAUSTED" $JD/python.log | tail -4 | cut -c1-300
[ -f $OUT/profile.json ] && python3 - $OUT/profile.json > $JD/summary.txt <<'PY'
import json, statistics, sys
p = json.load(open(sys.argv[1])); print("status", p.get("status")); print("denoise_loop_calls", p.get("denoise_loop_calls")); print("devices", p.get("jax_devices"))
fx = []
for d, v in sorted(p.get("points", {}).items(), key=lambda kv: int(kv[0])):
    r = v["received_num_steps"]; fx.append(v["fixed_noise_chunk_sha256"])
    print("d", d, "received_eq_d", sum(1 for x in r if x == int(d)), "/", len(r), "chunk_shape", v["chunk_shape"], "host_median_s", round(statistics.median(v["host_s"]), 4), "uncompiled_vs_compiled_max_abs", v["uncompiled_vs_compiled_max_abs"], "fixed", v["fixed_noise_chunk_sha256"][:16])
print("distinct_fixed_noise_chunks", len(set(fx)))
PY
cat $JD/summary.txt 2>/dev/null; tail -3 $JD/resources.txt; sort -t= -k3 -n $JD/resources.txt | tail -1 | sed 's/^/peak_line /'
find $JD -type f | sort | xargs sha256sum
exit $rc

