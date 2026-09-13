#!/usr/bin/env python3
"""Resolve, persist, verify, then install the isolated CUDA runtime wheels.

Downloads live outside /tmp so an interrupted WSL session can resume them.
This is environment preparation, not a measured experiment.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote, urlparse

ROOT = Path('/opt/robopad')
PYTHON = ROOT / 'venv/bin/python'
WHEELS = ROOT / 'wheels'
WHEELS.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / 'logs/torch_resolution.json'
subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--dry-run', '--ignore-installed',
                '--report', str(REPORT), 'torch==2.8.0', 'torchvision==0.23.0',
                '--index-url', 'https://pypi.org/simple'], check=True)
items = json.loads(REPORT.read_text())['install']
expected = {}
for item in items:
    info = item['download_info']
    name = unquote(Path(urlparse(info['url']).path).name)
    digest = info['archive_info']['hashes']['sha256']
    expected[digest] = (name, info['url'])

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

for cached in Path('/root/.cache/pip').rglob('*.body'):
    if cached.stat().st_size > 1024 * 1024:
        digest = sha(cached)
        if digest in expected and not (WHEELS / expected[digest][0]).exists():
            shutil.copyfile(cached, WHEELS / expected[digest][0])
            print('Reused verified cached wheel:', expected[digest][0], flush=True)
lines = []
for digest, (name, url) in expected.items():
    target = WHEELS / name
    if target.exists() and not Path(str(target) + '.aria2').exists() and sha(target) == digest:
        continue
    lines.extend([url, f'  out={name}', f'  checksum=sha-256={digest}'])
queue = ROOT / 'logs/torch_downloads.txt'
queue.write_text('\n'.join(lines) + '\n')
if lines:
    subprocess.run(['aria2c', '--input-file=' + str(queue), '--dir=' + str(WHEELS),
                    '--max-concurrent-downloads=4', '--max-connection-per-server=8',
                    '--split=8', '--min-split-size=1M', '--file-allocation=none',
                    '--continue=true', '--max-tries=10', '--retry-wait=3',
                    '--summary-interval=60', '--console-log-level=warn',
                    '--show-console-readout=false'], check=True)
for digest, (name, _) in expected.items():
    if sha(WHEELS / name) != digest:
        raise RuntimeError('Wheel digest mismatch: ' + name)
subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--no-index',
                '--find-links', str(WHEELS), 'torch==2.8.0', 'torchvision==0.23.0'], check=True)
