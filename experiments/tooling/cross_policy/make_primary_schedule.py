#!/usr/bin/env python3
"""Derive the primary R22/R25 schedules without changing the frozen five-arm schedules.

Run only after initial states have been bound to their SHA-256 hashes:
  python experiments/tooling/cross_policy/make_primary_schedule.py \
    --full-schedule experiments/runs/cross_policy_confirmation_v1/smolvla/schedule.json \
    --output experiments/runs/cross_policy_confirmation_v1/smolvla/primary_r22_r25_schedule.json

Do the same for pi05. The output file is exclusive-create and needs a separate,
reviewed release binding its full byte SHA-256 and its row schedule_sha256.
"""
import argparse
import json
import os
from pathlib import Path

from common import CONFIRMATION_SEEDS, STATE_SLOTS, SUITES, TASK_IDS, sha

EXPECTED = {(suite, task, slot, seed) for suite in SUITES
            for task in TASK_IDS for slot in STATE_SLOTS for seed in CONFIRMATION_SEEDS}
PRIMARY = {"R22": (2, 2), "R25": (2, 5)}


def derive(document):
    if document.get("cohort") != "cross_policy_confirmation_v1":
        raise ValueError("wrong cohort")
    if document.get("model") not in ("smolvla", "pi05"):
        raise ValueError("unknown model")
    source = document.get("schedule")
    if not isinstance(source, list) or len(source) != 2000:
        raise ValueError("not the frozen five-condition schedule")
    if document.get("schedule_sha256") != sha(source):
        raise ValueError("source schedule hash mismatch")
    if document.get("state_binding") != "BOUND" or not all(r.get("state_sha256") for r in source):
        raise ValueError("state hashes have not been bound")
    if len({r["planned_run_id"] for r in source}) != 2000:
        raise ValueError("source has duplicate attempt identities")

    rows = [dict(row) for row in source if row["condition_id"] in PRIMARY]
    if len(rows) != 800:
        raise ValueError("expected 800 primary attempts")
    blocks = {}
    for row in rows:
        key = row["block_id"]
        blocks.setdefault(key, []).append(row)
    actual = set()
    for block in blocks.values():
        if len(block) != 2 or {r["condition_id"] for r in block} != set(PRIMARY):
            raise ValueError("incomplete or duplicated primary pair")
        ident = block[0]["episode_identity"]
        key = (ident["suite"], ident["task_id"], ident["state_slot"], ident["policy_seed"])
        if key in actual or any(r["episode_identity"] != ident for r in block):
            raise ValueError("repeated/mixed identity")
        actual.add(key)
        if len({r["state_sha256"] for r in block}) != 1:
            raise ValueError("different initial states within a pair")
        for row in block:
            if (row["h"], row["d"]) != PRIMARY[row["condition_id"]]:
                raise ValueError("unexpected h/d")
    if actual != EXPECTED:
        raise ValueError("incorrect primary population")

    # Preserve the original randomized order. Re-index *copies* for this subset.
    for index, row in enumerate(rows):
        row["schedule_index"] = index
    for block in blocks.values():
        for index, row in enumerate(block):
            row["condition_index"] = index
    return {
        "schema_version": 1,
        "cohort": "cross_policy_confirmation_v1",
        "model": document["model"],
        "stage": "PRIMARY_R22_R25_ONLY",
        "attempt_count": 800,
        "identity_count": 400,
        "state_binding": "BOUND",
        "state_manifest_sha256": document["state_manifest_sha256"],
        "source_schedule_sha256": document["schedule_sha256"],
        "schedule_sha256": sha(rows),
        "schedule": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.full_schedule.read_text("utf-8"))
    result = derive(source)
    if args.output.exists():
        raise SystemExit("REFUSED: output exists; do not replace reviewed schedule")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create. This creates no attempt journal and never launches inference.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(str(args.output.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"output": str(args.output), "model": result["model"],
                      "attempt_count": result["attempt_count"],
                      "schedule_sha256": result["schedule_sha256"]}))


if __name__ == "__main__":
    main()
