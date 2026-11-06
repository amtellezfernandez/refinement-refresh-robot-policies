#!/usr/bin/env bash
# Build the RTX job command for one PDR diagnostic. Usage: mkjob.sh RUN_ID DRIVER RESERVATION TIMEOUT_S [driver args after CODE CKPT OUT]
# Prints the command to stdout. Driver shipped as gzip+base64 with a SHA-256 check (the Windows SSH hop limits commands to ~8 KB).
set -euo pipefail
RID=$1; DRV=$2; RES=$3; TMO=$4; shift 4
NAME=$(basename "$DRV"); SHA=$(shasum -a 256 "$DRV" | cut -c1-64); B64=$(gzip -9c "$DRV" | base64 | tr -d '\n')
cat <<EOF
set -o pipefail
W=\$HOME/alba-cross-policy-20261101; C=\$W/code-078e166/experiments/tooling/cross_policy; JD=\$HOME/jobs/pdr-$RID; OUT=\$JD/out; mkdir -p \$JD \$OUT
exec 9>\$HOME/local-science-programme-20261008/gpu.lock; flock -n 9 || { echo GPU_LOCK_HELD; exit 3; }
( while sleep 2; do a=\$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo); if [ "\$a" -lt 6 ]; then echo "WATCHDOG_FIRED \$a \$(date -u +%T)" >> \$JD/watchdog.txt; pkill -f $NAME; fi; done ) & WD=\$!
( while sleep 3; do echo "\$(date -u +%T) memavail_gib=\$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) gpu_used_mib=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> \$JD/resources.txt; done ) & RS=\$!
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="\$W/syslibs/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false ALBA_GPU_RESERVATION=$RES PDR_RUN_ID=$RID
env | grep -E '^(XLA|JAX|CUDA|OMP|MKL|OPENBLAS|HF_|TRANSFORMERS|ALBA|LD_LIB|MUJOCO|PYOPENGL|CUBLAS|PDR)' | sort > \$JD/env.txt
sha256sum \$C/server_openpi.py \$C/common.py \$C/launch_guard.py \$C/provenance.py > \$JD/code_sha256.txt; (cd \$W/src/openpi && git rev-parse HEAD && git status --porcelain | wc -l) > \$JD/openpi_rev.txt
EOF
if [ -n "${STAGED:-}" ]; then echo "cp \$HOME/jobs/pdr-staged/$NAME \$JD/$NAME"; else echo "echo $B64 | base64 -d | gunzip > \$JD/$NAME"; fi
cat <<EOF
[ "\$(sha256sum \$JD/$NAME | cut -c1-64)" = "$SHA" ] || { echo DRIVER_HASH_MISMATCH; exit 7; }; echo "driver_sha256 $SHA"
taskset -c 0-2 timeout $TMO \$W/envs/openpi/bin/python \$JD/$NAME \$C \$W/ckpt/checkpoints/pi05_libero \$OUT/result.json $* > \$JD/python.log 2>&1
rc=\$?; echo "python_rc=\$rc" > \$JD/rc.txt; cat \$JD/rc.txt
kill \$WD \$RS 2>/dev/null
grep -E "STATUS|MARK|RESOURCE_EXHAUSTED|Error" \$JD/python.log | tail -40 | cut -c1-300
sort -t= -k3 -n \$JD/resources.txt | tail -1 | sed 's/^/peak_line /'
echo BEGIN_RESULT; gzip -9c \$OUT/result.json 2>/dev/null | base64 -w0; echo; echo END_RESULT
find \$JD -type f | sort | xargs sha256sum
exit \$rc
EOF
