"""Shared helpers for the cross-policy campaign. Standard library only; Python 3.8+.

The OpenPI LIBERO client runs on Python 3.8, so nothing here may use newer syntax.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK_IDS = tuple(range(10))
STATE_SLOTS = (0, 1, 2, 3, 4)
CONFIRMATION_SEEDS = (810001, 810002)  # Alba's specification
# Alba specified "fresh identities" for this cohort without naming seeds. PROPOSED; see audit R2.
OBSERVATION_AGE_SEEDS = (820001, 820002)
SCHEDULE_SEED = 20261101
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20261102
QUALITY_MARGIN_POINTS = -5.0
SMOLVLA_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
LEROBOT_REVISION = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
OPENPI_NORM_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyed_int(*key):
    """Stable 32-bit integer from a key. No process state enters this value."""
    return int.from_bytes(hashlib.sha256(canonical(list(key))).digest()[:4], "little")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_write_json(path, value):
    """Write through fsync and rename, so a crash leaves the old record or the new one."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-" + path.name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def identity(suite, task_id, state_slot, policy_seed):
    return {"suite": suite, "task_id": task_id, "state_slot": state_slot, "policy_seed": policy_seed}


def identity_id(row):
    return "{suite}.task{task_id:03d}.slot{state_slot}.seed{policy_seed}".format(**row)
