set -o pipefail
W=$HOME/alba-cross-policy-20261101; C=$W/code-078e166/experiments/tooling/cross_policy; JD=$HOME/jobs/pdr-G02; OUT=$JD/out; mkdir -p $JD $OUT
exec 9>$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo GPU_LOCK_HELD; exit 3; }
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 6 ]; then echo "WATCHDOG_FIRED $a $(date -u +%T)" >> $JD/watchdog.txt; pkill -f pdr_g2_gate.py; fi; done ) & WD=$!
( while sleep 3; do echo "$(date -u +%T) memavail_gib=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $JD/resources.txt; done ) & RS=$!
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$W/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false ALBA_GPU_RESERVATION=R018 PDR_RUN_ID=G02
env | grep -E '^(XLA|JAX|CUDA|OMP|MKL|OPENBLAS|HF_|TRANSFORMERS|ALBA|LD_LIB|MUJOCO|PYOPENGL|CUBLAS|PDR)' | sort > $JD/env.txt
sha256sum $C/server_openpi.py $C/common.py $C/launch_guard.py $C/provenance.py > $JD/code_sha256.txt; (cd $W/src/openpi && git rev-parse HEAD && git status --porcelain | wc -l) > $JD/openpi_rev.txt
cp $HOME/jobs/pdr-staged/pdr_g2_gate.py $JD/pdr_g2_gate.py
[ "$(sha256sum $JD/pdr_g2_gate.py | cut -c1-64)" = "e533a605a577f79677a33959ca99f2f23da4f1f4e138eeaefd1a816e5bfa8fff" ] || { echo DRIVER_HASH_MISMATCH; exit 7; }; echo "driver_sha256 e533a605a577f79677a33959ca99f2f23da4f1f4e138eeaefd1a816e5bfa8fff"
XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" taskset -c 0-2 timeout 2400 $W/envs/openpi/bin/python $JD/pdr_g2_gate.py $C $W/ckpt/checkpoints/pi05_libero $OUT/r_det1.json  >> $JD/python.log 2>&1; echo "rc_det1=$?" >> $JD/rcs.txt
XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" taskset -c 0-2 timeout 2400 $W/envs/openpi/bin/python $JD/pdr_g2_gate.py $C $W/ckpt/checkpoints/pi05_libero $OUT/r_det2.json  >> $JD/python.log 2>&1; echo "rc_det2=$?" >> $JD/rcs.txt
XLA_FLAGS="" taskset -c 0-2 timeout 2400 $W/envs/openpi/bin/python $JD/pdr_g2_gate.py $C $W/ckpt/checkpoints/pi05_libero $OUT/r_def.json  >> $JD/python.log 2>&1; echo "rc_def=$?" >> $JD/rcs.txt
cat $JD/rcs.txt; python3 -c "import json,glob; json.dump({f.split('/')[-1]:json.load(open(f)) for f in sorted(glob.glob('$OUT/r_*.json'))}, open('$OUT/result.json','w'))"
rc=$?; echo "python_rc=$rc" > $JD/rc.txt; cat $JD/rc.txt
kill $WD $RS 2>/dev/null
grep -E "STATUS|MARK|RESOURCE_EXHAUSTED|Error" $JD/python.log | tail -40 | cut -c1-300
sort -t= -k3 -n $JD/resources.txt | tail -1 | sed 's/^/peak_line /'
echo BEGIN_RESULT; gzip -9c $OUT/result.json 2>/dev/null | base64 -w0; echo; echo END_RESULT
find $JD -type f | sort | xargs sha256sum
exit $rc