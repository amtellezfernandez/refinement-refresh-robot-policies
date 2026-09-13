#!/usr/bin/env bash
# Run only inside the isolated RoboPAD-Experiments WSL distribution.
set -euo pipefail
export PATH=/usr/lib/wsl/lib:/opt/robopad/venv/bin:$PATH
export HF_HOME=/opt/robopad/caches/huggingface
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
exec /opt/robopad/venv/bin/python "$@"
