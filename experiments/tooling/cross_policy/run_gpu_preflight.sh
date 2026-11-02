#!/usr/bin/env bash
# Bounded GPU preflight, frozen in review/GPU_PREFLIGHT_PREREG.md. Runs no confirmation identity.
#   ALBA_GPU_RESERVATION=<rid> run_gpu_preflight.sh <workspace> <new output dir>
# Any failure stops the run; the partial evidence stays.
set -euo pipefail
W=$1
OUT=$2
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
TOOLS="$REPO/experiments/tooling/cross_policy"
LOCK="$HOME/local-science-programme-20261008/gpu.lock"
: "${ALBA_GPU_RESERVATION:?G0 REFUSED: no reservation}"
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
  echo "commit=$(cat "$REPO/BUNDLE_COMMIT" 2>/dev/null || echo unknown)"
  echo "gpu_used_mib_before=$used"
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
  echo "compute_apps_seen_by_nvidia_smi: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
} > "$OUT/start.txt"

timeout 3600 "$W/envs/smolvla/bin/python" "$TOOLS/preflight_smolvla_small.py" \
  --checkpoint "$W/ckpt/smolvla_libero" \
  --reference-inputs "$REPO/experiments/runs/development_gate_001/inputs.pt" \
  --reference-manifest "$REPO/experiments/runs/development_gate_001/manifest.json" \
  --output "$OUT/smolvla"

timeout 1800 "$W/envs/openpi/bin/python" "$TOOLS/preflight_pi05_small.py" \
  --checkpoint "$W/ckpt/checkpoints/pi05_libero" --output "$OUT/pi05"

XLA_PYTHON_CLIENT_PREALLOCATE=false "$W/envs/openpi/bin/python" "$TOOLS/server_openpi.py" serve \
  --checkpoint "$W/ckpt/checkpoints/pi05_libero" --port 8000 > "$OUT/pi05/server.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && break
  kill -0 "$SERVER" || { echo "server exited; see server.log"; exit 5; }
  sleep 5
done
# envs/client does not install LIBERO; OpenPI's example takes it from this submodule.
PYTHONPATH="$W/src/openpi/third_party/libero" \
  timeout 1800 "$W/envs/client/bin/python" "$TOOLS/preflight_pi05_client_small.py" --output "$OUT/pi05" --port 8000
echo PREFLIGHT_SCRIPT_DONE | tee "$OUT/done.txt"
