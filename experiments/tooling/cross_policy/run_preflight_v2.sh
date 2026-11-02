#!/usr/bin/env bash
# Versioned preflight cross_policy_preflight_v2. Run ID cross_policy_preflight_v2.run001.
# NOT YET EXECUTED. Refused until Alba's release records the unresolved decisions.
#   ALBA_GPU_RESERVATION=<rid> run_preflight_v2.sh <workspace> <new output dir>
# Runs no confirmation identity. One model process at a time. Any failure stops the run; evidence stays.
set -euo pipefail
W=$1
OUT=$2
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
TOOLS="$REPO/experiments/tooling/cross_policy"
LOCK="$HOME/local-science-programme-20261008/gpu.lock"
RUN_ID=cross_policy_preflight_v2.run001
PROTOCOL="$REPO/experiments/protocols/cross_policy_preflight_v2.json"
RELEASE="$REPO/review/releases/cross_policy_preflight_v2.release.json"
GUARD_PYTHON=${PYTHON_FOR_GUARD:-$W/envs/tools/bin/python}
: "${ALBA_GPU_RESERVATION:?G0 REFUSED: no reservation}"
"$GUARD_PYTHON" "$TOOLS/launch_guard.py" preflight_v2 || { echo "G0 REFUSED: see above"; exit 6; }
[ ! -e "$OUT" ] || { echo "REFUSED: $OUT exists; a counted run is never overwritten"; exit 2; }

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH="$W/libero-config"
export LD_LIBRARY_PATH="$W/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8

exec 9>"$LOCK"
flock -n 9 || { echo "G2 REFUSED: gpu.lock is held"; exit 3; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -ge 3072 ]; then
  echo "G1 REFUSED: ${used} MiB already in use"
  exit 4
fi
mkdir -p "$OUT"
cat > "$OUT/start.json" <<EOF
{"run_id": "$RUN_ID", "reservation": "$ALBA_GPU_RESERVATION",
 "commit": "$(cat "$REPO/BUNDLE_COMMIT" 2>/dev/null || git -C "$REPO" rev-parse HEAD)",
 "protocol_sha256": "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)",
 "release_sha256": "$(sha256sum "$RELEASE" | cut -d' ' -f1)",
 "gpu": "$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"}
EOF

# CPU. Identity is the saved state and its hash: the generator's seed does not reproduce every state.
"$W/envs/smolvla/bin/python" "$TOOLS/generate_states.py" --cohort matched_r22_r25_confirmation_v1 --output "$OUT/states"

"$W/envs/smolvla/bin/python" "$TOOLS/runner_smolvla.py" conditions --checkpoint "$W/ckpt/smolvla_libero" --output "$OUT/smolvla_conditions"
"$W/envs/smolvla/bin/python" "$TOOLS/runner_smolvla.py" preflight --checkpoint "$W/ckpt/smolvla_libero" --output "$OUT/smolvla"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$W/envs/openpi/bin/python" "$TOOLS/server_openpi.py" profile --checkpoint "$W/ckpt/checkpoints/pi05_libero" --output "$OUT/pi05_profile"
"$W/envs/openpi/bin/python" "$TOOLS/server_openpi.py" serve --checkpoint "$W/ckpt/checkpoints/pi05_libero" --output "$OUT/pi05_server" --port 8000 > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && break
  kill -0 "$SERVER" || { echo "server exited; see server.log"; exit 5; }
  sleep 5
done
# envs/client does not install LIBERO; OpenPI's example takes it from this submodule.
export PYTHONPATH="$W/src/openpi/third_party/libero"
"$W/envs/client/bin/python" "$TOOLS/client_openpi.py" conditions --output "$OUT/pi05_conditions" --port 8000
"$W/envs/client/bin/python" "$TOOLS/client_openpi.py" parity --output "$OUT/pi05_parity" --port 8000
unset PYTHONPATH
kill "$SERVER" 2>/dev/null || true

"$W/envs/smolvla/bin/python" "$TOOLS/inspect_videos.py" "$OUT/smolvla/videos"
"$W/envs/smolvla/bin/python" "$TOOLS/inspect_videos.py" "$OUT/pi05_parity/videos"

"$GUARD_PYTHON" "$TOOLS/preflight_verdict.py" "$OUT" > "$OUT/verdict.json" || true
(cd "$OUT" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS)
echo RUN_FINISHED | tee "$OUT/done.txt"
