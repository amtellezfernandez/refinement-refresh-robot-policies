#!/usr/bin/env python3
"""Resolve and prefetch non-Torch runtime dependencies; do not install them."""
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote, urlparse

ROOT = Path('/opt/robopad')
SOURCE = Path('/mnt/c/Users/rayan/OneDrive/Desktop/RoboPAD/paper/proposal_paper2/build/runtime/lerobot-b6ec0060779550c0a157ae34feb89e0cf86012a8')
TARGET = ROOT / 'src/lerobot'
if not (TARGET / 'pyproject.toml').exists():
    shutil.copytree(SOURCE, TARGET, dirs_exist_ok=True)
REPORT = ROOT / 'logs/full_runtime_resolution.json'
subprocess.run([str(ROOT / 'venv/bin/python'), '-m', 'pip', 'install', '--dry-run',
                '--ignore-installed', '--report', str(REPORT), '-e',
                str(TARGET) + '[smolvla,libero,evaluation]', 'torch==2.8.0',
                'torchvision==0.23.0', 'torchcodec==0.7.0', 'matplotlib', 'jsonschema'], check=True)
torch_hashes = {item['download_info']['archive_info']['hashes']['sha256']
                for item in json.loads((ROOT / 'logs/torch_resolution.json').read_text())['install']}
directory = ROOT / 'extra_wheels'
directory.mkdir(exist_ok=True)
lines = []
for item in json.loads(REPORT.read_text())['install']:
    info = item['download_info']
    if 'archive_info' not in info:
        continue  # locally built editable LeRobot source
    digest = info['archive_info']['hashes']['sha256']
    if digest in torch_hashes:
        continue  # already being persistently downloaded by prepare_torch.py
    name = unquote(Path(urlparse(info['url']).path).name)
    lines.extend([info['url'], '  out=' + name, '  checksum=sha-256=' + digest])
queue = ROOT / 'logs/runtime_downloads.txt'
queue.write_text('\n'.join(lines) + '\n')
subprocess.run(['aria2c', '--input-file=' + str(queue), '--dir=' + str(directory),
                '--max-concurrent-downloads=3', '--max-connection-per-server=8',
                '--split=8', '--min-split-size=1M', '--file-allocation=none',
                '--continue=true', '--max-tries=10', '--retry-wait=3',
                '--summary-interval=60', '--console-log-level=warn',
                '--show-console-readout=false'], check=True)
print('Runtime dependencies prefetched; no package installation was performed.')
