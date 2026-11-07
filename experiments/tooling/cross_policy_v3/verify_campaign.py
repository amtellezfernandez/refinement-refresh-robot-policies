"""Read-only campaign input checks in the selected native interpreter. No native imports."""
import argparse
import json
from pathlib import Path

from v3common import file_sha
import qualification
import release
from make_schedule_v2 import validate_manifest
from analysis_v2 import pair_table
from classify import MISSING


def verify(workspace, model):
    root = Path(workspace).resolve() / model
    schedule = root / "schedule.json"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    validate_manifest(manifest, model)
    if file_sha(root / "initial_states.npz") != manifest["initial_states_npz_sha256"]:
        raise ValueError("state archive does not match its native manifest")
    doc = json.loads(schedule.read_text("utf-8"))
    pair_table(doc, {r["planned_run_id"]: MISSING for r in doc["schedule"]})
    if doc["model"] != model or doc.get("state_manifest_file_sha256") != file_sha(manifest_path):
        raise ValueError("schedule is not bound to this model's manifest")
    qualification.require_qualified(model)
    release.check_release(schedule, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", choices=("pi05", "smolvla"), required=True)
    args = parser.parse_args()
    verify(args.workspace, args.model)
    print("{}: existing release, qualification, schedule and state hashes validated".format(args.model))


if __name__ == "__main__":
    main()
