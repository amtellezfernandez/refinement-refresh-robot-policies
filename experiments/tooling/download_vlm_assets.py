#!/usr/bin/env python3
"""Cache the pinned public tokenizer/configuration, without VLM weights.

The trained SmolVLA checkpoint supplies all weights via strict loading. Cached
configuration/tokenizer contents resolve offline in subsequent experiment runs.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = 'HuggingFaceTB/SmolVLM2-500M-Instruct'
REVISION = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
FILES = ('added_tokens.json', 'chat_template.json', 'config.json',
         'generation_config.json', 'merges.txt', 'preprocessor_config.json',
         'processor_config.json', 'special_tokens_map.json', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.json')
ROOT = Path('/opt/robopad/caches/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct')
SNAPSHOT = ROOT / 'snapshots' / REVISION
SNAPSHOT.mkdir(parents=True, exist_ok=True)

def download(name):
    target = SNAPSHOT / name
    if not target.exists():
        for attempt in range(8):
            try:
                with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', timeout=60) as r:
                    data = r.read()
                if name.endswith('.json'):
                    json.loads(data)
                target.write_bytes(data)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2)
    return name, hashlib.sha256(target.read_bytes()).hexdigest()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    hashes = dict(pool.map(download, FILES))
(ROOT / 'refs').mkdir(exist_ok=True)
(ROOT / 'refs/main').write_text(REVISION)
manifest = {'repository': REPO, 'revision': REVISION, 'files_sha256': hashes,
            'weights_downloaded': False, 'scope': 'Tokenizer and configuration only'}
Path('/opt/robopad/logs/base_vlm_assets.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
