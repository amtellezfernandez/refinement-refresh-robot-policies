#!/usr/bin/env bash
# Isolated RoboPAD-Experiments WSL distribution only. No shared host packages.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
ROOT=/opt/robopad
SOURCE=/mnt/c/Users/rayan/OneDrive/Desktop/RoboPAD/paper/proposal_paper2/build/runtime/lerobot-b6ec0060779550c0a157ae34feb89e0cf86012a8
mkdir -p "$ROOT/logs" "$ROOT/src" "$ROOT/caches"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
apt-get update -qq
apt-get install -y -qq python3-venv git curl aria2 build-essential ffmpeg libegl1 libgl1 libosmesa6 libglib2.0-0
python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --upgrade pip
python3 "$(dirname "$0")/prepare_torch.py"
if [ ! -f "$ROOT/src/lerobot/pyproject.toml" ]; then
  cp -a "$SOURCE" "$ROOT/src/lerobot"
fi
"$ROOT/venv/bin/python" -m pip install -e "$ROOT/src/lerobot[smolvla,libero,evaluation]" torchcodec==0.7.0 matplotlib jsonschema
"$ROOT/venv/bin/python" -m pip freeze > "$ROOT/logs/requirements.lock.txt"
"$ROOT/venv/bin/python" - <<'PY'
import json, torch
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'available': torch.cuda.is_available(),
                  'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))
assert torch.cuda.is_available()
x = torch.randn(256, 256, device='cuda')
assert torch.isfinite(x @ x).all()
PY
