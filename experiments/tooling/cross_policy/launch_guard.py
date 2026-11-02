"""The gate every GPU entry point calls before importing torch or jax.

- preflight: needs a named reservation ID in ALBA_GPU_RESERVATION, and refuses unrelated
  GPU compute processes so that one-model-per-GPU holds.
- preflight_v2: the versioned preflight's entry check. Needs Alba's release, matching the protocol
  hash and recording every unresolved decision.
- confirmation: Alba's primary R22/R25 only. Needs her release with PRIMARY_R22_R25_ONLY scope, the
  protocol hash and the per-model schedule row and file hashes (verify_released_schedule, from her PR).
  Only Alba's review creates that file. It is not committed on this branch.
- observation_age and asynchronous: deferred by Alba. Refused, release or not.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIRMATION_SEEDS, STATE_SLOTS, SUITES, TASK_IDS, file_sha, sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PROTOCOLS = {"confirmation": "cross_policy_confirmation_v1"}
DEFERRED = {"observation_age": "observation_age_confirmation_v1", "asynchronous": "asynchronous_confirmation_v1"}
PREFLIGHT_V2 = "cross_policy_preflight_v2"
REQUIRED_PREFLIGHT_DECISIONS = ("R4", "R6", "R9", "N1")
IMPLEMENTED_RENDERER = "OSMesa"


class LaunchRefused(SystemExit):
    pass


def compute_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader"], universal_newlines=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def verify_released_schedule(root, phase, name, release, schedule_path, model):
    """Fail closed before any rollout when the on-disk schedule is not the reviewed one.

    The release must bind the byte-level schedule file as well as its canonical
    row digest. For the first confirmation, R22 and R25 alone are authorised.
    """
    if model not in ("smolvla", "pi05") or schedule_path is None:
        raise LaunchRefused("REFUSED: release needs a model and a schedule path")
    path = Path(schedule_path).resolve()
    expected_dir = (Path(root) / "experiments/runs" / name / model).resolve()
    if expected_dir not in path.parents:
        raise LaunchRefused("REFUSED: schedule is outside its model/cohort directory")
    try:
        document = json.loads(path.read_text("utf-8"))
        rows = document["schedule"]
    except (OSError, ValueError, KeyError) as error:
        raise LaunchRefused("REFUSED: cannot load schedule: {}".format(error))
    if not isinstance(rows, list) or not rows:
        raise LaunchRefused("REFUSED: empty/invalid schedule")
    if document.get("cohort") != name or document.get("model") != model:
        raise LaunchRefused("REFUSED: model/cohort does not match the schedule")
    if document.get("state_binding") != "BOUND" or not all(r.get("state_sha256") for r in rows):
        raise LaunchRefused("REFUSED: schedule contains unbound reset states")
    digest = sha(rows)
    if document.get("schedule_sha256") != digest:
        raise LaunchRefused("REFUSED: schedule row hash mismatch")
    if release.get("schedule_sha256_by_model", {}).get(model) != digest:
        raise LaunchRefused("REFUSED: schedule rows were not approved")
    if release.get("schedule_file_sha256_by_model", {}).get(model) != file_sha(path):
        raise LaunchRefused("REFUSED: schedule file bytes were not approved")
    if document.get("attempt_count") != len(rows):
        raise LaunchRefused("REFUSED: schedule attempt count mismatch")
    blocks = {}
    for row in rows:
        if row.get("cohort") != name or row.get("model") != model:
            raise LaunchRefused("REFUSED: mixed model/cohort within schedule")
        block = row.get("block_id")
        ident = row.get("episode_identity")
        if not block or not isinstance(ident, dict):
            raise LaunchRefused("REFUSED: invalid identity row")
        if ident.get("suite") not in SUITES or ident.get("task_id") not in TASK_IDS:
            raise LaunchRefused("REFUSED: suite/task outside protocol")
        if ident.get("state_slot") not in STATE_SLOTS:
            raise LaunchRefused("REFUSED: state slot outside protocol")
        if row.get("planned_run_id") is None:
            raise LaunchRefused("REFUSED: planned attempt lacks an ID")
        blocks.setdefault(block, []).append(row)
    if document.get("identity_count") != len(blocks):
        raise LaunchRefused("REFUSED: identity count mismatch")
    if len({r["planned_run_id"] for r in rows}) != len(rows):
        raise LaunchRefused("REFUSED: repeated planned attempt ID")
    if phase == "confirmation":
        if release.get("approval_scope") != "PRIMARY_R22_R25_ONLY":
            raise LaunchRefused("REFUSED: no primary-only approval scope")
        if release.get("approved_conditions") != ["R22", "R25"]:
            raise LaunchRefused("REFUSED: confirmation must be restricted to R22/R25")
        if release.get("approved_attempts_per_model") != 800 or len(rows) != 800 or len(blocks) != 400:
            raise LaunchRefused("REFUSED: primary confirmation must have 800 attempts/400 identities per model")
        expected_ident = {(suite, task, slot, seed) for suite in SUITES
                          for task in TASK_IDS for slot in STATE_SLOTS for seed in CONFIRMATION_SEEDS}
        actual_ident = set()
        for group in blocks.values():
            ident = group[0]["episode_identity"]
            key = (ident["suite"], ident["task_id"], ident["state_slot"], ident["policy_seed"])
            if key in actual_ident or any(r["episode_identity"] != ident for r in group):
                raise LaunchRefused("REFUSED: repeated/mixed confirmation identity")
            actual_ident.add(key)
            if {r["condition_id"] for r in group} != {"R22", "R25"} or len(group) != 2:
                raise LaunchRefused("REFUSED: non-primary conditions or incomplete pair")
            if len({r["state_sha256"] for r in group}) != 1:
                raise LaunchRefused("REFUSED: paired conditions have different state hashes")
            for r in group:
                expected_d = 2 if r["condition_id"] == "R22" else 5
                if r.get("h") != 2 or r.get("d") != expected_d:
                    raise LaunchRefused("REFUSED: condition does not match its frozen h/d")
        if actual_ident != expected_ident:
            raise LaunchRefused("REFUSED: confirmation identities do not match frozen population")
    return {"schedule_sha256": digest, "schedule_file_sha256": file_sha(path), "model": model}


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


def check(phase, root=ROOT, environ=None, processes=compute_processes, schedule_path=None, model=None):
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
        verify_released_schedule(root, phase, PROTOCOLS[phase], release, schedule_path, model)
        reservation = need_reservation(environ)
    else:
        raise LaunchRefused("REFUSED: unknown phase {}".format(phase))
    running = processes()
    if running:
        raise LaunchRefused("REFUSED: GPU compute processes present: {}".format(running))
    return {"phase": phase, "reservation": reservation}


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1])))
