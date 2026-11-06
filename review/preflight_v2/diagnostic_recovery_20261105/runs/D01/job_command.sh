set -o pipefail
W=$HOME/alba-cross-policy-20261101; C=$W/code-078e166/experiments/tooling/cross_policy; JD=$HOME/jobs/pdr-D01; OUT=$JD/out; mkdir -p $JD $OUT
exec 9>$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo GPU_LOCK_HELD; exit 3; }
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 6 ]; then echo "WATCHDOG_FIRED $a $(date -u +%T)" >> $JD/watchdog.txt; pkill -f pdr_d_xproc.py; fi; done ) & WD=$!
( while sleep 3; do echo "$(date -u +%T) memavail_gib=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $JD/resources.txt; done ) & RS=$!
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$W/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false ALBA_GPU_RESERVATION=R018 PDR_RUN_ID=D01
env | grep -E '^(XLA|JAX|CUDA|OMP|MKL|OPENBLAS|HF_|TRANSFORMERS|ALBA|LD_LIB|MUJOCO|PYOPENGL|CUBLAS|PDR)' | sort > $JD/env.txt
sha256sum $C/server_openpi.py $C/common.py $C/launch_guard.py $C/provenance.py > $JD/code_sha256.txt; (cd $W/src/openpi && git rev-parse HEAD && git status --porcelain | wc -l) > $JD/openpi_rev.txt
echo H4sICJkP7WoCA3Bkcl9kX3hwcm9jLnB5AG1TS0/jMBC+51fMhosjBZNmtxwq9VC1LMuC2qot0kpVFbmx2xgS22s70Ajx33cSWkBoc4jm8c03T599u6idvdhKdSHUE5jGF1p9D8IwnE8WMEl6A8itdu7cWJ0L58AKlHidy60spW9A7yDXlZGl4GBkQvuga29q74A4IThaXaN8IbzMQSp0RBSms+n5eHY/Xd1Mr2kwt1IhfPlr1JJZ9vxByHIvtXLAPPBhGvfjXkJhzMrW559lLsAIC3/uRtnPu9H1EpitYNsApoMHvaVB7dheDMBwm/Hs0LZATQPj2eQqm9wsYHw7X8HsfpX9Xs6mbc+BrIy2HgrmilJuY3hwWsWgXYxduCAYx11M3AbBsLVRZvdP694m/lDSz8r3TdDKhvmCSuWE9SRBkuiUSdUVVsQcKANwBkr/ZQO4+pGkJwCGPAmbaSOUkS3Q6S/AndUVvPmp0aXMpXAUqxdWZ53ewJGrYo8iO3rEgVWmFF+4AmWoZYrrirbbI2mSXvZ6ST8KTvjh/1hIFDyKBpeCM9H0thXnXWaCaqkZJ+3YoihYIOIlfF9YOMDZUjw8abWie+HJJ18MYRjhzxUMcS+vKO6kdb5TXoOdtsDxpICkMfRj6CXRIAD8GObANphj1rKGdIXh7HfCEi5zT441x5DhUWRD3gpKSycyhKKh3TnluBdH1pjxIPh55w43URStw+NNotZlW6y7Ajdr55E/2mDy4/VQtKf9S8Ko19vGC0eiiBbiwOVeOE/ew9+a+kzA1skGg0rZwd7rIYu42zPB68NhPIc4UNM+HhIuV6PV/bJ7OKXwgodvp3ts41QjTnNX1q4YrmwtouAfBOBWLPwDAAA= | base64 -d | gunzip > $JD/pdr_d_xproc.py
[ "$(sha256sum $JD/pdr_d_xproc.py | cut -c1-64)" = "fa28f1fb2c2526ab8103fab58f9335cb50f30bb92693a2ce19f9063a4cba9196" ] || { echo DRIVER_HASH_MISMATCH; exit 7; }; echo "driver_sha256 fa28f1fb2c2526ab8103fab58f9335cb50f30bb92693a2ce19f9063a4cba9196"
XLA_FLAGS="" taskset -c 0-2 timeout 1800 $W/envs/openpi/bin/python $JD/pdr_d_xproc.py $C $W/ckpt/checkpoints/pi05_libero $OUT/r_def1.json  >> $JD/python.log 2>&1
XLA_FLAGS="" taskset -c 0-2 timeout 1800 $W/envs/openpi/bin/python $JD/pdr_d_xproc.py $C $W/ckpt/checkpoints/pi05_libero $OUT/r_def2.json  >> $JD/python.log 2>&1
XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" taskset -c 0-2 timeout 1800 $W/envs/openpi/bin/python $JD/pdr_d_xproc.py $C $W/ckpt/checkpoints/pi05_libero $OUT/r_det1.json  >> $JD/python.log 2>&1
XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" taskset -c 0-2 timeout 1800 $W/envs/openpi/bin/python $JD/pdr_d_xproc.py $C $W/ckpt/checkpoints/pi05_libero $OUT/r_det2.json  >> $JD/python.log 2>&1
python3 -c "import json,glob; json.dump({f.split('/')[-1]:json.load(open(f)) for f in sorted(glob.glob('$OUT/r_*.json'))}, open('$OUT/result.json','w'))"
rc=$?; echo "python_rc=$rc" > $JD/rc.txt; cat $JD/rc.txt
kill $WD $RS 2>/dev/null
grep -E "STATUS|MARK|RESOURCE_EXHAUSTED|Error" $JD/python.log | tail -40 | cut -c1-300
sort -t= -k3 -n $JD/resources.txt | tail -1 | sed 's/^/peak_line /'
echo BEGIN_RESULT; gzip -9c $OUT/result.json 2>/dev/null | base64 -w0; echo; echo END_RESULT
find $JD -type f | sort | xargs sha256sum
exit $rc