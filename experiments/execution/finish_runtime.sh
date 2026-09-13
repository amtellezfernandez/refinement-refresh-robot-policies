#!/usr/bin/env bash
# Continue after prepare_torch.py completes, without repeating large downloads.
set -euo pipefail
ROOT=/opt/robopad
SOURCE=/mnt/c/Users/rayan/OneDrive/Desktop/RoboPAD/paper/proposal_paper2/build/runtime/lerobot-b6ec0060779550c0a157ae34feb89e0cf86012a8
export PIP_CACHE_DIR="$ROOT/caches/pip"
mkdir -p "$ROOT/src" "$ROOT/logs" "$PIP_CACHE_DIR"
if [ ! -f "$ROOT/src/lerobot/pyproject.toml" ]; then
  cp -a "$SOURCE" "$ROOT/src/lerobot"
fi
"$ROOT/venv/bin/python" -m pip install --find-links "$ROOT/extra_wheels" --find-links "$ROOT/wheels" \
  -e "$ROOT/src/lerobot[smolvla,libero,evaluation]" \
  torch==2.8.0 torchvision==0.23.0 torchcodec==0.7.0 matplotlib jsonschema
"$ROOT/venv/bin/python" -m pip freeze > "$ROOT/logs/requirements.lock.txt"
"$ROOT/venv/bin/python" - <<'PY'
import json, torch
assert torch.cuda.is_available()
x = torch.randn(256, 256, device='cuda')
assert torch.isfinite(x @ x).all()
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'available': torch.cuda.is_available(),
                  'device': torch.cuda.get_device_name(0)}))
PY
