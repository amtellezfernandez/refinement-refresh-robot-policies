"""The gate every GPU entry point calls before importing torch or jax.

- preflight: needs a named reservation ID in ALBA_GPU_RESERVATION, and refuses unrelated
  GPU compute processes so that one-model-per-GPU holds.
- preflight_v2: the versioned preflight's entry check. Needs Alba's release, matching the protocol
  hash and recording every unresolved decision.
- matched_r22_r25: the only confirmation design Alba permits after a successful preflight. Needs a
  release whose SHA-256 of the protocol matches and which lists the exact schedule file. Only Alba's
  review creates that file. It is not committed on this branch.
- confirmation, observation_age and asynchronous: deferred by Alba. Refused, release or not.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import file_sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PROTOCOLS = {"matched_r22_r25": "matched_r22_r25_confirmation_v1"}
DEFERRED = {"confirmation": "cross_policy_confirmation_v1", "observation_age": "observation_age_confirmation_v1",
            "asynchronous": "asynchronous_confirmation_v1"}
PREFLIGHT_V2 = "cross_policy_preflight_v2"
REQUIRED_PREFLIGHT_DECISIONS = ("R4", "R6", "R7", "R9", "R12", "N1", "N2")
IMPLEMENTED_RENDERER = "OSMesa"


class LaunchRefused(SystemExit):
    pass


def compute_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader"], universal_newlines=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def read_release(root, name):
    path = root / "review/releases" / (name + ".release.json")
    if not path.is_file():
        raise LaunchRefused("REFUSED: {} has no review release".format(name))
    release = json.loads(path.read_text("utf-8"))
    protocol = root / "experiments/protocols" / (name + ".json")
    status = str(json.loads(protocol.read_text("utf-8")).get("status", ""))
    if status.startswith("BLOCKED"):
        raise LaunchRefused("REFUSED: {} is {}".format(name, status))
    if release.get("approved_by") != "Alba" or release.get("protocol_sha256") != file_sha(protocol):
        raise LaunchRefused("REFUSED: release does not match the protocol hash")
    return release


def need_reservation(environ):
    reservation = environ.get("ALBA_GPU_RESERVATION")
    if not reservation:
        raise LaunchRefused("REFUSED: needs ALBA_GPU_RESERVATION; the RTX GPU is not released")
    return reservation


def check(phase, root=ROOT, environ=None, processes=compute_processes, schedule=None):
    environ = os.environ if environ is None else environ
    root = Path(root)
    if phase in DEFERRED:
        raise LaunchRefused("REFUSED: {} is deferred by Alba's decision".format(DEFERRED[phase]))
    if phase == "preflight":
        reservation = need_reservation(environ)
    elif phase == "preflight_v2":
        decisions = read_release(root, PREFLIGHT_V2).get("decisions", {})
        missing = [key for key in REQUIRED_PREFLIGHT_DECISIONS if not decisions.get(key)]
        if missing:
            raise LaunchRefused("REFUSED: release lacks decisions {}".format(missing))
        if decisions["R6"] != IMPLEMENTED_RENDERER:
            raise LaunchRefused("REFUSED: the harness implements only {} rendering".format(IMPLEMENTED_RENDERER))
        reservation = need_reservation(environ)
    elif phase in PROTOCOLS:
        release = read_release(root, PROTOCOLS[phase])
        if schedule is None or file_sha(schedule) not in release.get("schedule_sha256s", []):
            raise LaunchRefused("REFUSED: release does not cover this schedule file")
        reservation = need_reservation(environ)
    else:
        raise LaunchRefused("REFUSED: unknown phase {}".format(phase))
    running = processes()
    if running:
        raise LaunchRefused("REFUSED: GPU compute processes present: {}".format(running))
    return {"phase": phase, "reservation": reservation}


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1])))
