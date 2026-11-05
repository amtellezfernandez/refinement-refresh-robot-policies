set -o pipefail
W=$HOME/alba-cross-policy-20261101; C=$W/code-078e166/experiments/tooling/cross_policy; JD=$HOME/jobs/alba-v3-p3-smolvla-profile; OUT=$JD/out; mkdir -p $JD
exec 9>$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo GPU_LOCK_HELD; exit 3; }
( while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 6 ]; then echo "WATCHDOG_FIRED MemAvailable=${a}GiB $(date -u +%T)" >> $JD/watchdog.txt; pkill -f p3_profile_only.py; fi; done ) & WD=$!
( while sleep 3; do echo "$(date -u +%T) memavail_gib=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $JD/resources.txt; done ) & RS=$!
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$W/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" LIBERO_CONFIG_PATH=$W/libero-config OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ALBA_GPU_RESERVATION=R015
( while sleep 0.2; do if [ -e $OUT/videos ] || [ -e $OUT/parity_attempts ]; then echo "PARITY_TRIPWIRE_FIRED $(date -u +%T)" >> $JD/tripwire.txt; pkill -9 -f p3_profile_only.py; fi; done ) & TW=$!
env | grep -E '^(OMP|MKL|OPENBLAS|HF_|TRANSFORMERS|ALBA|LD_LIB|MUJOCO|PYOPENGL|CUBLAS|LIBERO)' | sort > $JD/env.txt
sha256sum $C/runner_smolvla.py $C/journal.py $C/launch_guard.py $C/common.py $C/../run_paired.py > $JD/code_sha256.txt
cat > $JD/p3_profile_only.py <<'PY'
# NON-COUNTING probe driver. Runs the frozen runner_smolvla.preflight() unmodified up to its profile step.
# Parity is prevented deterministically: the module-level AttemptJournal, first used at the start of parity
# (runner_smolvla.py:292, after profile.json is written and env.close()), is replaced by a function that raises.
import argparse, json, sys
from pathlib import Path
C = Path(sys.argv[1]); sys.path[:0] = [str(C), str(C.parent)]
import launch_guard, runner_smolvla
class ParityBlocked(SystemExit):
    pass
def blocked(*args, **kwargs):
    raise ParityBlocked("PARITY_BLOCKED_BY_PROBE: AttemptJournal({}) refused; no parity environment or episode created".format(args))
runner_smolvla.AttemptJournal = blocked
print("guard", json.dumps(launch_guard.check("preflight")), flush=True)
args = argparse.Namespace(mode="preflight", phase="preflight", schedule=None, states=None, checkpoint=Path(sys.argv[2]), output=Path(sys.argv[3]))
try:
    runner_smolvla.preflight(args)
    print("UNEXPECTED: preflight returned without reaching parity", flush=True); sys.exit(5)
except ParityBlocked as stop:
    print(stop, flush=True); sys.exit(0)
PY
taskset -c 0-2 timeout 3600 $W/envs/smolvla/bin/python $JD/p3_profile_only.py $C $W/ckpt/smolvla_libero $OUT > $JD/python.log 2>&1
rc=$?; echo "python_rc=$rc" > $JD/rc.txt; cat $JD/rc.txt
kill $WD $RS $TW 2>/dev/null; cat $JD/tripwire.txt 2>/dev/null
grep -E "guard|PARITY_BLOCKED|UNEXPECTED|Error|require" $JD/python.log | tail -6 | cut -c1-300
echo "parity artefacts: $(ls -d $OUT/parity_attempts $OUT/videos 2>/dev/null | wc -l)"; ls $OUT
[ -f $OUT/profile.json ] && python3 - $OUT/profile.json > $JD/summary.txt <<'PY'
import json, statistics, sys
p = json.load(open(sys.argv[1]))
for d in sorted(p, key=int):
    v = p[d]; c = v["counts"]
    print("d", d, "counts", c, "denoise_eq_generations_x_d", c["denoise_steps"] == c["generations"] * int(d), "generations_eq_100", c["generations"] == 100,
          "host_median_s", round(statistics.median(v["host_s"]), 4), "cuda_median_ms", round(statistics.median(v["cuda_ms"]), 2), "n_host", len(v["host_s"]), "n_cuda", len(v["cuda_ms"]),
          "peak_alloc_GiB", round(v["peak_allocated_bytes"] / 2**30, 2), "chunk_shape", v["chunk_shape"])
PY
cat $JD/summary.txt 2>/dev/null; sort -t= -k3 -n $JD/resources.txt | tail -1 | sed 's/^/peak_line /'
find $JD -type f | sort | xargs sha256sum
exit $rc
