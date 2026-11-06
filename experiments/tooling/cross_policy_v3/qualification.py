"""Machine-readable qualification records (01 A3a; D4, D5, D7, D8) and the PRE gate. Python 3.8+, stdlib only.

Records live in experiments/protocols/v3_qualification/<model>.json. Every item ships UNVERIFIED. An item can
read PASSED or FAILED only with evidence: a run identifier, the authorisation that permitted the check, and an
evidence file inside the repository whose sha256 matches. This module validates records and refuses; it has no
function that marks anything PASSED.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3common import MODELS, REPO_ROOT, file_sha  # noqa: E402

RECORD_DIR = REPO_ROOT / "experiments/protocols/v3_qualification"
SCHEMA = "cross_policy_v3_qualification_v1"
STATUSES = ("UNVERIFIED", "PASSED", "FAILED", "NOT_APPLICABLE")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# Which items apply to which model (01 A3a). NOT_APPLICABLE is legal only where this says False.
ITEMS = {
    "host_renderer": {"pi05": True, "smolvla": True},
    "driver_package_locks": {"pi05": True, "smolvla": True},
    "asset_hashes": {"pi05": True, "smolvla": True},
    "controller_mapping": {"pi05": True, "smolvla": True},
    "action_normalization_clipping": {"pi05": True, "smolvla": True},
    "determinism_config": {"pi05": True, "smolvla": True},
    "count_gate": {"pi05": True, "smolvla": True},
    "parity": {"pi05": True, "smolvla": False},
    "evaluator": {"pi05": True, "smolvla": True},
    "smolvla_horizons_termination": {"pi05": False, "smolvla": True},
    "protocol_a_preflight": {"pi05": True, "smolvla": True},
}


class Refused(SystemExit):
    pass


class RecordError(ValueError):
    pass


def record_path(model, directory=RECORD_DIR):
    return Path(directory) / "{}.json".format(model)


def validate(record, model, root=REPO_ROOT):
    """Raise RecordError on any malformed record. PASSED/FAILED need verifiable evidence."""
    if record.get("schema") != SCHEMA or record.get("model") != model:
        raise RecordError("wrong schema or model")
    items = record.get("items")
    if not isinstance(items, dict) or set(items) != set(ITEMS):
        raise RecordError("items must be exactly {}".format(sorted(ITEMS)))
    for name, item in items.items():
        status = item.get("status")
        if status not in STATUSES:
            raise RecordError("{}: unknown status {!r}".format(name, status))
        if status == "NOT_APPLICABLE" and ITEMS[name][model]:
            raise RecordError("{} applies to {}; it cannot be NOT_APPLICABLE".format(name, model))
        if ITEMS[name][model] is False and status != "NOT_APPLICABLE":
            raise RecordError("{} does not apply to {}; it must be NOT_APPLICABLE".format(name, model))
        if not isinstance(item.get("bindings"), dict):
            raise RecordError("{}: bindings must be an object".format(name))
        evidence = item.get("evidence")
        if status in ("UNVERIFIED", "NOT_APPLICABLE"):
            if evidence is not None:
                raise RecordError("{}: {} carries evidence".format(name, status))
            continue
        # PASSED or FAILED: every binding filled, evidence present and checkable.
        if any(value is None for value in item["bindings"].values()):
            raise RecordError("{}: {} with an unbound value".format(name, status))
        if not isinstance(evidence, dict):
            raise RecordError("{}: {} without evidence".format(name, status))
        for field in ("run_id", "authorisation_sha256", "evidence_path", "evidence_sha256"):
            if not evidence.get(field):
                raise RecordError("{}: evidence lacks {}".format(name, field))
        if not HEX64.match(evidence["authorisation_sha256"]) or not HEX64.match(evidence["evidence_sha256"]):
            raise RecordError("{}: malformed evidence hash".format(name))
        path = (Path(root) / evidence["evidence_path"]).resolve()
        if Path(root).resolve() not in path.parents or not path.is_file():
            raise RecordError("{}: evidence file missing or outside the repository".format(name))
        if file_sha(path) != evidence["evidence_sha256"]:
            raise RecordError("{}: evidence file hash mismatch".format(name))
    return record


def load(model, directory=RECORD_DIR, root=REPO_ROOT):
    if model not in MODELS:
        raise RecordError("unknown model {!r}".format(model))
    record = json.loads(record_path(model, directory).read_text("utf-8"))
    return validate(record, model, root)


def require_qualified(model, directory=RECORD_DIR, root=REPO_ROOT):
    """PRE gate (02 B10): refuse unless every applicable item is PASSED. Returns the record's sha256."""
    try:
        record = load(model, directory, root)
    except (OSError, ValueError) as error:
        raise Refused("REFUSED: qualification record for {} unreadable or invalid: {}".format(model, error))
    pending = sorted(name for name, item in record["items"].items()
                     if ITEMS[name][model] and item["status"] != "PASSED")
    if pending:
        raise Refused("REFUSED: {} not qualified; items not PASSED: {}".format(model, ", ".join(pending)))
    return file_sha(record_path(model, directory))


def summary(model, directory=RECORD_DIR, root=REPO_ROOT):
    record = load(model, directory, root)
    return {name: item["status"] for name, item in sorted(record["items"].items())}


if __name__ == "__main__":
    for name in MODELS:
        print(name, json.dumps(summary(name), sort_keys=True))
