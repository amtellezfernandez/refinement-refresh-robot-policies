set -o pipefail
W=$HOME/alba-cross-policy-20261101; C=$W/code-078e166/experiments/tooling/cross_policy; JD=$HOME/jobs/alba-v3-p5-pi05-profile-memfrac100; OUT=$JD/out; mkdir -p $JD
exec 9>$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo GPU_LOCK_HELD; exit 3; }
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 6 ]; then echo "WATCHDOG_FIRED MemAvailable=${a}GiB $(date -u +%T)" >> $JD/watchdog.txt; pkill -f "server_openpi.py profile"; fi; done ) & WD=$!
( while sleep 3; do echo "$(date -u +%T) memavail_gib=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $JD/resources.txt; done ) & RS=$!
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$W/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false ALBA_GPU_RESERVATION=R016
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
env | grep -E '^(XLA|JAX|CUDA|OMP|MKL|OPENBLAS|HF_|TRANSFORMERS|ALBA|LD_LIB|MUJOCO|PYOPENGL|CUBLAS)' | sort > $JD/env.txt
cat $C/../../../BUNDLE_COMMIT 2>/dev/null; sha256sum $C/server_openpi.py $C/common.py $C/launch_guard.py $C/provenance.py > $JD/code_sha256.txt
taskset -c 0-2 timeout 5400 $W/envs/openpi/bin/python $C/server_openpi.py profile --checkpoint $W/ckpt/checkpoints/pi05_libero --output $OUT > $JD/python.log 2>&1
rc=$?; echo "python_rc=$rc" > $JD/rc.txt; cat $JD/rc.txt
kill $WD $RS 2>/dev/null
grep -E "RESOURCE_EXHAUSTED|Error|SystemExit|unsupported" $JD/python.log | tail -8 | cut -c1-300
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
