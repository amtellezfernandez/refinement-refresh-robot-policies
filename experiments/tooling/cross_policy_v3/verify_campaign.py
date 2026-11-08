"""Read-only campaign input checks in the selected native interpreter. No native imports."""
import argparse
import json
import sys
from pathlib import Path

from v3common import file_sha
import qualification
import release
from make_schedule_v2 import validate_manifest
from analysis_v2 import pair_table
from classify import MISSING


def check_runtime_role(role, version=None):
    version = sys.version_info if version is None else version
    if role == "server" and tuple(version[:2]) < (3, 11):
        raise ValueError("Pinned OpenPI server requires Python >=3.11; select its server environment, not the LIBERO client")


def verify(workspace, model, role="client"):
    check_runtime_role(role)
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
    parser.add_argument("--role", choices=("client", "server"), default="client")
    args = parser.parse_args()
    if args.role == "server" and args.model != "pi05":
        parser.error("server role applies only to pi05")
    verify(args.workspace, args.model, args.role)
    print("Interpreter: {} {} (role {})".format(sys.executable, sys.version.split()[0], args.role))
    print("{}: existing release, qualification, schedule and state hashes validated".format(args.model))


if __name__ == "__main__":
    main()
