#!/usr/bin/env bash
# Mandatory preflight for both models. NOT YET EXECUTED, and refused until Alba's decisions are recorded.
#   ALBA_GPU_RESERVATION=<rid> run_full_preflight.sh <workspace> <new output dir>
# One model process at a time. Any failure stops the run; partial evidence stays.
set -euo pipefail
W=$1
OUT=$2
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
TOOLS="$REPO/experiments/tooling/cross_policy"
LOCK="$HOME/local-science-programme-20261008/gpu.lock"
: "${ALBA_GPU_RESERVATION:?G0 REFUSED: no reservation}"
# R9 asks for the parity criterion before parity data exists, and R6 decides the renderer.
DECISIONS="$REPO/review/releases/full_preflight.release.json"
grep -q '"approved_by": *"Alba"' "$DECISIONS" 2>/dev/null || { echo "REFUSED: no release from Alba at $DECISIONS"; exit 6; }
[ ! -e "$OUT" ] || { echo "REFUSED: $OUT exists; a rerun needs a new directory"; exit 2; }

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH="$W/libero-config"
export LD_LIBRARY_PATH="$W/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec 9>"$LOCK"
flock -n 9 || { echo "G2 REFUSED: gpu.lock is held"; exit 3; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -ge 3072 ]; then
  echo "G1 REFUSED: ${used} MiB already in use"
  exit 4
fi
mkdir -p "$OUT"
{
  echo "reservation=$ALBA_GPU_RESERVATION"
  echo "commit=$(cat "$REPO/BUNDLE_COMMIT" 2>/dev/null || git -C "$REPO" rev-parse HEAD)"
  echo "decisions_sha256=$(sha256sum "$DECISIONS" | cut -d' ' -f1)"
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "$OUT/start.txt"

"$W/envs/smolvla/bin/python" "$TOOLS/runner_smolvla.py" conditions --checkpoint "$W/ckpt/smolvla_libero" --output "$OUT/smolvla_conditions"
"$W/envs/smolvla/bin/python" "$TOOLS/runner_smolvla.py" preflight --checkpoint "$W/ckpt/smolvla_libero" --output "$OUT/smolvla"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$W/envs/openpi/bin/python" "$TOOLS/server_openpi.py" profile --checkpoint "$W/ckpt/checkpoints/pi05_libero" --output "$OUT/pi05_profile"
"$W/envs/openpi/bin/python" "$TOOLS/server_openpi.py" serve --checkpoint "$W/ckpt/checkpoints/pi05_libero" --port 8000 > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && break
  kill -0 "$SERVER" || { echo "server exited; see server.log"; exit 5; }
  sleep 5
done
export PYTHONPATH="$W/src/openpi/third_party/libero"
"$W/envs/client/bin/python" "$TOOLS/client_openpi.py" conditions --output "$OUT/pi05_conditions" --port 8000
"$W/envs/client/bin/python" "$TOOLS/client_openpi.py" parity --output "$OUT/pi05_parity" --port 8000
echo FULL_PREFLIGHT_DONE | tee "$OUT/done.txt"
