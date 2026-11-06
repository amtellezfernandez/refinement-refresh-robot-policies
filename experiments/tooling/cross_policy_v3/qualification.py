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
# PASSED needs these binding values, and the evidence file's JSON must carry the same value under the same key.
REQUIRED_VERDICTS = {"evaluator": "EVALUATOR_QUALIFIED", "protocol_a_preflight": "PREFLIGHT_PASS"}
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

# Binding keys of every item, as shipped. A record whose keys differ is refused.
TEMPLATE_BINDING_KEYS = {
    "pi05": {
        "action_normalization_clipping": [
            "loaded_equals_file",
            "norm_stats_sha256",
            "openpi_clipping",
            "quantile_mode",
            "robosuite_scale_action_clipping"
        ],
        "asset_hashes": [
            "asset_root",
            "bddl_manifest_sha256",
            "libero_asset_manifest_sha256"
        ],
        "controller_mapping": [
            "controller_source_sha256",
            "controller_type",
            "delta_frame",
            "input_range",
            "output_range",
            "robosuite_version"
        ],
        "count_gate": [
            "att_drill_record_sha256",
            "cnt_ind_record_sha256",
            "server_code_sha256",
            "startup_gate_record_sha256"
        ],
        "determinism_config": [
            "cnt_t_record_sha256",
            "xla_flags"
        ],
        "driver_package_locks": [
            "cuda_umd",
            "lock_file_sha256",
            "native_stack",
            "nvidia_smi_driver",
            "openpi_commit",
            "wsl_driver_library"
        ],
        "evaluator": [
            "protocol_e_record_sha256",
            "verdict"
        ],
        "host_renderer": [
            "gl_renderer",
            "gl_vendor",
            "gl_version",
            "gpu_uuid",
            "hostname",
            "kernel",
            "mujoco_gl",
            "os_release",
            "planned_host",
            "uname_m"
        ],
        "parity": [
            "noise_match_record_sha256",
            "p2a_record_sha256",
            "r9_record_sha256"
        ],
        "protocol_a_preflight": [
            "preflight_record_sha256",
            "verdict"
        ],
        "smolvla_horizons_termination": []
    },
    "smolvla": {
        "action_normalization_clipping": [
            "evaluator_boundary_mapping",
            "loaded_equals_file",
            "normalization_mode",
            "policy_input_signature",
            "postprocessor_stats_sha256",
            "preprocessor_stats_sha256",
            "robosuite_scale_action_clipping"
        ],
        "asset_hashes": [
            "asset_root",
            "bddl_manifest_sha256",
            "libero_asset_manifest_sha256"
        ],
        "controller_mapping": [
            "controller_source_sha256",
            "controller_type",
            "delta_frame",
            "input_range",
            "output_range",
            "robosuite_version"
        ],
        "count_gate": [
            "adapter_code_sha256",
            "scnt_record_sha256"
        ],
        "determinism_config": [
            "allow_tf32",
            "cublas_workspace_config",
            "cudnn_benchmark",
            "torch_use_deterministic_algorithms"
        ],
        "driver_package_locks": [
            "cuda_umd",
            "lerobot_commit",
            "lock_file_sha256",
            "native_stack",
            "nvidia_smi_driver",
            "wsl_driver_library"
        ],
        "evaluator": [
            "protocol_e_record_sha256",
            "verdict"
        ],
        "host_renderer": [
            "gl_renderer",
            "gl_vendor",
            "gl_version",
            "gpu_uuid",
            "hostname",
            "kernel",
            "mujoco_gl",
            "os_release",
            "planned_host",
            "uname_m"
        ],
        "parity": [],
        "protocol_a_preflight": [
            "preflight_record_sha256",
            "verdict"
        ],
        "smolvla_horizons_termination": [
            "horizon_by_suite",
            "success_terminates_by_suite"
        ]
    }
}


class Refused(SystemExit):
    pass


class RecordError(ValueError):
    pass


def record_path(model, directory=RECORD_DIR):
    return Path(directory) / "{}.json".format(model)


def template_keys(model):
    """Binding keys per item, fixed by the record shipped in the commit that introduced this module."""
    return {name: set(keys) for name, keys in TEMPLATE_BINDING_KEYS[model].items()}


def _checked_file(root, relative, expected_sha, what, name):
    path = (Path(root) / str(relative)).resolve()
    if Path(root).resolve() not in path.parents or not path.is_file():
        raise RecordError("{}: {} file missing or outside the repository".format(name, what))
    if not HEX64.match(str(expected_sha)) or file_sha(path) != expected_sha:
        raise RecordError("{}: {} file hash mismatch".format(name, what))
    return path


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
        if set(item["bindings"]) != template_keys(model)[name]:
            raise RecordError("{}: binding keys differ from the shipped template".format(name))
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
        for field in ("run_id", "authorisation_path", "authorisation_sha256", "evidence_path", "evidence_sha256"):
            if not evidence.get(field):
                raise RecordError("{}: evidence lacks {}".format(name, field))
        authorisation = _checked_file(root, evidence["authorisation_path"], evidence["authorisation_sha256"],
                                      "authorisation", name)
        try:
            approved = json.loads(authorisation.read_text("utf-8")).get("approved_by")
        except ValueError:
            approved = None
        if approved != "Alba":
            raise RecordError("{}: authorisation is not Alba's".format(name))
        evidence_file = _checked_file(root, evidence["evidence_path"], evidence["evidence_sha256"], "evidence", name)
        if status == "PASSED" and name in REQUIRED_VERDICTS:
            try:
                claimed = json.loads(evidence_file.read_text("utf-8")).get("verdict")
            except ValueError:
                claimed = None
            if item["bindings"].get("verdict") != REQUIRED_VERDICTS[name] or claimed != REQUIRED_VERDICTS[name]:
                raise RecordError("{}: PASSED needs verdict {} in the binding and in the evidence".format(
                    name, REQUIRED_VERDICTS[name]))
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


def binding(model, item, key, directory=RECORD_DIR, root=REPO_ROOT):
    """A PASSED item's bound value. Refuses for any item that is not PASSED."""
    record = load(model, directory, root)
    if record["items"][item]["status"] != "PASSED":
        raise Refused("REFUSED: {}.{} is not PASSED".format(model, item))
    return record["items"][item]["bindings"][key]


def summary(model, directory=RECORD_DIR, root=REPO_ROOT):
    record = load(model, directory, root)
    return {name: item["status"] for name, item in sorted(record["items"].items())}


if __name__ == "__main__":
    for name in MODELS:
        print(name, json.dumps(summary(name), sort_keys=True))
