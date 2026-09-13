#!/usr/bin/env python3
"""Create the focused paired deployment pilot plan.

The pilot keeps the same four conditions and keyed pairing contract as the
320-attempt design, but restricts collection to two held-out Spatial tasks,
five initial states, and two policy seeds (80 attempts). It creates no results.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from make_evaluation_plan import DEFAULT_SCHEDULING_SEED, digest, make_plan


PILOT_SUITE = "libero_spatial"
PILOT_TASK_IDS = (5, 6)
PILOT_STATE_IDS = (10, 11, 12, 13, 14)
PILOT_POLICY_SEEDS = (2000, 2001)


def make_pilot_plan(scheduling_seed: int = DEFAULT_SCHEDULING_SEED) -> dict:
    source = make_plan(scheduling_seed)
    selected = [
        row for row in source["schedule"]
        if row["episode_identity"]["suite"] == PILOT_SUITE
        and row["episode_identity"]["task_id"] in PILOT_TASK_IDS
        and row["episode_identity"]["initial_state_id"] in PILOT_STATE_IDS
        and row["episode_identity"]["policy_seed"] in PILOT_POLICY_SEEDS
    ]
    identities = []
    for row in selected:
        identity = tuple(row["episode_identity"][key]
                         for key in ("suite", "task_id", "initial_state_id", "policy_seed"))
        if identity not in identities:
            identities.append(identity)
    schedule = []
    for block_index, identity in enumerate(identities):
        rows = [row for row in selected if tuple(
            row["episode_identity"][key]
            for key in ("suite", "task_id", "initial_state_id", "policy_seed")) == identity]
        for condition_index, row in enumerate(rows):
            item = copy.deepcopy(row)
            item["schedule_index"] = len(schedule)
            item["block_index"] = block_index
            item["condition_index"] = condition_index
            schedule.append(item)

    plan = copy.deepcopy(source)
    plan["design_revision"] = "2026-09-12-focused-pilot"
    plan["purpose"] = (
        "Focused paired pilot only; no runs launched and no outcomes recorded. "
        "Execution requires all runtime and provenance gates."
    )
    plan["design"] = {
        "suites": [PILOT_SUITE],
        "task_ids": list(PILOT_TASK_IDS),
        "initial_state_ids": list(PILOT_STATE_IDS),
        "policy_seeds": list(PILOT_POLICY_SEEDS),
        "conditions": plan["design"]["conditions"],
        "task_count": len(PILOT_TASK_IDS),
        "block_count": len(identities),
        "conditions_per_block": 4,
        "planned_attempt_count": len(schedule),
        "estimand": "The two declared Spatial tasks, equally weighted; descriptive pilot only.",
    }
    plan["scheduling"]["schedule_sha256"] = digest(schedule)
    plan["scheduling"]["pilot_scope"] = {
        "suite": PILOT_SUITE,
        "task_ids": list(PILOT_TASK_IDS),
        "initial_state_ids": list(PILOT_STATE_IDS),
        "policy_seeds": list(PILOT_POLICY_SEEDS),
    }
    plan["schedule"] = schedule
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("pilot_evaluation.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SCHEDULING_SEED)
    args = parser.parse_args()
    plan = make_pilot_plan(args.seed)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {len(plan['schedule'])} focused pilot attempts; no experiments launched.")
    print(f"Schedule SHA-256: {plan['scheduling']['schedule_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
