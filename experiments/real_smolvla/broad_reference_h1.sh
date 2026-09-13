#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl
export HF_HOME=/mnt/NVME1/projectx_robopad/caches/huggingface
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$SCRIPT_DIR:/mnt/NVME1/projectx_robopad/venvs/lerobot312/coverage_stub"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_ROOT=$(mktemp -d /mnt/NVME1/projectx_robopad/logs/reference_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX)
V=/mnt/NVME1/projectx_robopad/venvs/lerobot312/bin
for T in libero_spatial libero_object libero_goal libero_10; do
  O="$RUN_ROOT/${T}_h1"
  printf 'N\n' | "$V/lerobot-eval" \
    --policy.path=/mnt/NVME1/projectx_robopad/checkpoints/smolvla_libero \
    --policy.device=cuda --policy.num_steps=10 --policy.n_action_steps=1 \
    --env.type=libero --env.task="$T" --env.task_ids='[0,1,2,3,4]' \
    --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false \
    --output_dir="$O" --job_name="broad_${T}_h1" > "$O.log" 2>&1
done
