#!/usr/bin/env bash
set -euo pipefail

campaign_root=/mnt/NVME1/projectx-robopad-proposal2/results/refinement_refresh_final_001
runtime_root=/mnt/NVME1/projectx-robopad-proposal2
python_bin=/mnt/NVME1/projectx_robopad/venvs/lerobot312/bin/python
script_dir="$runtime_root/paper/proposal_paper2/experiments/execution"
output="$campaign_root/language_shape_flops.json"
log="$campaign_root/language_shape_postflight.log"

exec >>"$log" 2>&1
echo "postflight watcher started $(date --iso-8601=seconds)"

while [[ ! -f "$campaign_root/completed.json" ]]; do
    if [[ -f "$campaign_root/failure.json" ]]; then
        echo "campaign failure marker observed; refusing postflight profile"
        exit 1
    fi
    sleep 30
done

if [[ -f "$output" ]]; then
    echo "shape profile already exists; refusing overwrite"
    exit 1
fi

while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
    sleep 30
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$python_bin" "$script_dir/profile_language_shapes.py" \
    --checkpoint "$runtime_root/runtime/checkpoint" \
    --inputs "$runtime_root/results/development_gate_concurrent_002/inputs.pt" \
    --shape-audit "$campaign_root/language_shape_audit.json" \
    --output "$output"

echo "postflight profile completed $(date --iso-8601=seconds)"
