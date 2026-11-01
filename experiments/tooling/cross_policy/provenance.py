"""Run provenance. A value that cannot be read is recorded as UNKNOWN, never guessed or omitted."""
import hashlib
import importlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from common import file_sha, sha

ROOT = Path(__file__).resolve().parents[3]
ENV_KEYS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBERO_CONFIG_PATH", "LP_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_FLAGS", "JAX_PLATFORMS", "ALBA_GPU_RESERVATION")


def _read(command):
    try:
        return subprocess.check_output(command, universal_newlines=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return "UNKNOWN: {!r}".format(error)


def _version(module):
    try:
        return getattr(importlib.import_module(module), "__version__", "UNKNOWN: no __version__")
    except Exception as error:  # Provenance must survive a broken optional import.
        return "UNKNOWN: {!r}".format(error)


def code_hashes():
    directory = Path(__file__).resolve().parent
    return {path.name: file_sha(path) for path in sorted(directory.glob("*.py"))}


def tree_hash(path):
    """Content hash over relative paths and file hashes, for a checkpoint directory."""
    root = Path(path)
    if not root.is_dir():
        return "UNKNOWN: not a directory"
    entries = [[str(p.relative_to(root)), file_sha(p)] for p in sorted(root.rglob("*")) if p.is_file()]
    return {"files": len(entries), "sha256": hashlib.sha256(repr(entries).encode()).hexdigest()}


def capture(model, args, extra):
    arguments = {k: str(v) for k, v in vars(args).items()}
    record = {"schema_version": 1, "model": model, "captured_unix": time.time(),
              "git_commit": _read(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
              "git_status_short": _read(["git", "-C", str(ROOT), "status", "--short"]),
              "code_sha256": code_hashes(), "argv": sys.argv, "arguments": arguments,
              "python": sys.version, "platform": platform.platform(), "hostname": platform.node(),
              "packages": _read([sys.executable, "-m", "pip", "freeze"]),
              "versions": {m: _version(m) for m in ("numpy", "torch", "jax", "mujoco", "robosuite", "lerobot", "openpi")},
              "gpu": _read(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"]),
              "cuda_driver_header": _read(["nvidia-smi", "--version"]),
              "environment": {k: os.environ.get(k) for k in ENV_KEYS}}
    record.update(extra)
    checkpoint = extra.get("checkpoint_dir")
    if checkpoint:
        record["checkpoint_tree"] = tree_hash(checkpoint)
    record["sha256"] = sha({k: v for k, v in record.items() if k != "sha256"})
    return record
