#!/usr/bin/env bash
set -euo pipefail

study_root=/mnt/NVME1/projectx-robopad-proposal2
campaign="$study_root/results/refinement_refresh_final_001"
scripts="$study_root/paper/proposal_paper2/experiments/tooling"
python_bin=/mnt/NVME1/projectx_robopad/venvs/lerobot312/bin/python
log="$campaign/deadline_postflight.log"

exec >>"$log" 2>&1
echo "deadline postflight started $(date --iso-8601=seconds)"
while [[ ! -f "$campaign/primary_deadline/summary.json" ]]; do
    if [[ -f "$campaign/primary_deadline/failure.json" ]]; then
        echo "primary deadline run failed"
        exit 1
    fi
    sleep 15
done

while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
    sleep 5
done

export PYTHONPATH="$study_root/runtime/overlay:$study_root/runtime/lerobot/src:$scripts"
export HF_HOME=/mnt/NVME1/projectx_robopad/caches/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$python_bin" "$scripts/profile_language_shapes.py" \
    --checkpoint "$study_root/runtime/checkpoint" \
    --inputs "$study_root/results/development_gate_concurrent_002/inputs.pt" \
    --shape-audit "$campaign/language_shape_audit.json" \
    --output "$campaign/language_shape_flops.json"

"$python_bin" "$scripts/analyze_primary_deadline.py" \
    --campaign "$campaign" \
    --output "$campaign/primary_deadline_analysis.json"

echo "deadline postflight completed $(date --iso-8601=seconds)"
