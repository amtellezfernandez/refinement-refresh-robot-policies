#!/usr/bin/env bash
set -euo pipefail
study_root=/mnt/NVME1/projectx-robopad-proposal2
export PYTHONPATH="$study_root/runtime/overlay:$study_root/runtime/lerobot/src"
export HF_HOME=/mnt/NVME1/projectx_robopad/caches/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
cd "$study_root"
exec /mnt/NVME1/projectx_robopad/venvs/lerobot312/bin/python \
  paper/proposal_paper2/experiments/execution/last_campaign.py \
  --root "$study_root" --checkpoint "$study_root/runtime/checkpoint" \
  --output "$study_root/results/refinement_refresh_final_001"
