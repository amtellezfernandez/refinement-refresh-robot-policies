"""Read-only operator handoff for R22/R25. Python 3.8+, no native imports.

Reads existing artifacts only. Does not generate identities, qualify a model,
create authorizations, or execute any suggested command.
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qualification as Q
from v3common import MODELS, file_sha

SCHEMA = "refinement_campaign_preparation_v1"


def prepare(workspace, model, python=sys.executable):
    if model not in MODELS:
        raise ValueError("unknown model")
    root = Path(workspace).resolve() / model
    paths = {name: root / filename for name, filename in (
        ("manifest", "manifest.json"), ("schedule", "schedule.json"),
        ("states", "initial_states.npz"), ("fixtures", "fixtures.json"),
        ("startup_gate", "startup_gate.json"))}
    issues, artifacts = [], {}
    def issue(stage, message, action):
        issues.append({"stage": stage, "message": message, "next_action": action})
    try:
        statuses = Q.summary(model)
        for name, status in statuses.items():
            if Q.ITEMS[name][model] and status != "PASSED":
                issue("native_qualification", name + ": " + status,
                      "Supply retained evidence for this item from its separately authorized native check.")
    except (OSError, ValueError) as error:
        statuses = {}
        issue("native_qualification", str(error), "Repair the qualification record and its evidence references.")
    documents = {}
    for name in ("manifest", "schedule", "states", "fixtures"):
        path = paths[name]
        if not path.is_file():
            issue("inputs", "Missing " + str(path), "Place the existing model-native " + name + " at this path.")
            continue
        artifacts[name] = {"path": str(path), "sha256": file_sha(path)}
        if path.suffix == ".json":
            try:
                documents[name] = json.loads(path.read_text("utf-8"))
            except ValueError as error:
                issue("inputs", name + ": " + str(error), "Repair the malformed JSON artifact.")
    # Reuse the scientific harness's validators, rather than another definition
    # of pairing, completeness, population or qualification.
    if "manifest" in documents:
        try:
            from make_schedule_v2 import validate_manifest
            validate_manifest(documents["manifest"], model)
            if "states" in artifacts and documents["manifest"].get("initial_states_npz_sha256") != artifacts["states"]["sha256"]:
                raise ValueError("state archive hash differs from the native manifest")
        except (ValueError, KeyError, TypeError) as error:
            issue("inputs", "Manifest: " + str(error), "Reconcile the native manifest and state archive; do not regenerate reserved states here.")
    if "schedule" in documents:
        try:
            from analysis_v2 import pair_table
            from classify import MISSING
            doc = documents["schedule"]
            finals = {row["planned_run_id"]: MISSING for row in doc["schedule"]}
            pair_table(doc, finals)
            if doc.get("model") != model:
                raise ValueError("schedule names a different model")
            if "manifest" in artifacts and doc.get("state_manifest_file_sha256") != artifacts["manifest"]["sha256"]:
                raise ValueError("schedule is not bound to this manifest file")
        except (ValueError, KeyError, TypeError) as error:
            issue("inputs", "Schedule: " + str(error), "Correct the existing schedule binding and 400-pair/800-attempt accounting.")
    if not (HERE / "launch_v3.py").is_file():
        issue("integration", "launch_v3.py is absent", "Integrate the reviewed launcher from draft PR #6/#8.")
    native_args = []
    if model == "pi05":
        if not paths["startup_gate"].is_file():
            issue("native_qualification", "Missing " + str(paths["startup_gate"]), "Retain the attesting server startup gate after its authorized native validation.")
        native_args = ["--startup-gate", str(paths["startup_gate"])]
    else:
        native_args = ["--checkpoint", str(root / "checkpoint")]
        if not (root / "checkpoint").is_dir():
            issue("inputs", "Missing checkpoint directory", "Place the pinned SmolVLA checkpoint at " + str(root / "checkpoint"))
    manifests = [Path(workspace).resolve() / name / "manifest.json" for name in MODELS]
    if "fixtures" in documents and all(path.is_file() for path in manifests):
        try:
            from evaluator_validation import protocol_e_inputs
            inputs, _, _, _, _ = protocol_e_inputs(model, paths["fixtures"], manifests)
            if not inputs["ready_for_review"]:
                issue("protocol_e", "Missing fixture cases: " + repr(inputs["missing_cases"]),
                      "Complete the frozen Protocol E fixture set; E0 alone is insufficient.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            issue("protocol_e", str(error), "Repair Protocol E input bindings using evaluator_validation.py --check-inputs.")
    elif "fixtures" in documents:
        issue("protocol_e", "Both native manifests are required for the fixture collision check",
              "Supply each model's manifest before Protocol E input review.")
    commands = {
        "protocol_e_inputs_only": [python, str(HERE / "evaluator_validation.py"), "--stack", model,
                                   "--fixtures", str(paths["fixtures"]), "--check-inputs",
                                   "--confirmation-manifest", str(Path(workspace).resolve() / "pi05/manifest.json"),
                                   "--confirmation-manifest", str(Path(workspace).resolve() / "smolvla/manifest.json")],
        "confirmation_after_separate_release": [python, str(HERE / "launch_v3.py"), "--model", model,
            "--schedule", str(paths["schedule"]), "--states", str(paths["states"]),
            "--output", str(root / "run"), *native_args],
        "analysis_after_execution": [python, str(HERE / "analysis_v2.py"), "--schedule", str(paths["schedule"]),
            "--summary", str(root / "run/run_summary.json"), "--journal", str(root / "run/attempts"),
            "--output", str(root / "analysis.json")],
    }
    authorized, authorization_error = False, "No validated release"
    if paths["schedule"].is_file():
        try:
            import release
            release.check_release(paths["schedule"], model)
            authorized, authorization_error = True, None
        except (SystemExit, OSError, ValueError, KeyError, TypeError) as error:
            authorization_error = str(error)
    return {"schema": SCHEMA, "model": model, "read_only": True,
            "planned_pairs": 400, "planned_attempts": 800,
            "offline_inputs_complete": not issues,
            "readiness_note": "This preparation report does not independently verify native execution or release the campaign.",
            "execution_authorized": authorized, "authorization_error": authorization_error,
            "qualification": statuses, "artifacts": artifacts, "issues": issues,
            "commands": commands, "shell_commands": {k: " ".join(shlex.quote(a) for a in v) for k, v in commands.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
