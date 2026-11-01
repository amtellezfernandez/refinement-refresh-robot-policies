"""The gate every GPU entry point calls before importing torch or jax.

- preflight: needs a named reservation ID in ALBA_GPU_RESERVATION, and refuses unrelated
  GPU compute processes so that one-model-per-GPU holds.
- confirmation, observation_age and asynchronous: need a release file whose SHA-256 of the
  protocol matches. Only Alba's review creates that file. It is not committed on this branch,
  and the asynchronous protocol is also refused while its status says blocked.
"""
import json
import os
import subprocess
from pathlib import Path

from common import file_sha

ROOT = Path(__file__).resolve().parents[3]
PROTOCOLS = {"confirmation": "cross_policy_confirmation_v1", "observation_age": "observation_age_confirmation_v1",
             "asynchronous": "asynchronous_confirmation_v1"}


class LaunchRefused(SystemExit):
    pass


def compute_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader"], universal_newlines=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def check(phase, root=ROOT, environ=None, processes=compute_processes):
    environ = os.environ if environ is None else environ
    root = Path(root)
    if phase == "preflight":
        reservation = environ.get("ALBA_GPU_RESERVATION")
        if not reservation:
            raise LaunchRefused("REFUSED: preflight needs ALBA_GPU_RESERVATION; the RTX GPU is not released")
    elif phase in PROTOCOLS:
        name = PROTOCOLS[phase]
        protocol_path = root / "experiments/protocols" / (name + ".json")
        protocol = json.loads(protocol_path.read_text("utf-8"))
        if str(protocol.get("status", "")).startswith("BLOCKED"):
            raise LaunchRefused("REFUSED: {} is {}".format(name, protocol["status"]))
        release_path = root / "review/releases" / (name + ".release.json")
        if not release_path.is_file():
            raise LaunchRefused("REFUSED: {} has no review release; rollouts are not authorised".format(name))
        release = json.loads(release_path.read_text("utf-8"))
        if release.get("approved_by") != "Alba" or release.get("protocol_sha256") != file_sha(protocol_path):
            raise LaunchRefused("REFUSED: release does not match the protocol hash")
        reservation = environ.get("ALBA_GPU_RESERVATION")
        if not reservation:
            raise LaunchRefused("REFUSED: rollouts need ALBA_GPU_RESERVATION")
    else:
        raise LaunchRefused("REFUSED: unknown phase {}".format(phase))
    running = processes()
    if running:
        raise LaunchRefused("REFUSED: GPU compute processes present: {}".format(running))
    return {"phase": phase, "reservation": reservation}
