#!/usr/bin/env python3
"""Validate and analyze the completed frozen primary transfer comparison."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_execution import (
    check_initial_state,
    read_jsonl,
    reconcile_rollout_journal,
    reconcile_rollout_manifest,
)
from run_paired import file_sha, require


PRIMARY = {(2, 2), (2, 5)}


def condition(row):
    return row["h"], row["denoising_steps"]


def validate_episode(row):
    check_initial_state(row["initial_state"])
    generations = (row["steps"] + row["h"] - 1) // row["h"]
    require(row["counts"]["generations"] == generations, "generation count differs")
    require(row["counts"]["denoise_steps"] == generations * row["denoising_steps"],
            "denoising count differs")
    require(row["counts"]["prefill_calls"] == generations, "prefill count differs")
    require(row["counts"]["vision_encoder_calls"] == 2 * generations,
            "vision count differs")
    require(row["counts"]["action_expert_calls"] == generations * row["denoising_steps"],
            "action-expert count differs")


def validate_filtered_journal(root: Path, rows: list[dict]) -> None:
    included = {row["run_id"] for row in rows}
    events = [event for event in read_jsonl(root / "steps.jsonl") if event.get("run_id") in included]
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        with (temporary / "steps.jsonl").open("w") as stream:
            for event in events:
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        reconcile_rollout_journal(temporary, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = read_jsonl(args.campaign / "confirmation/episodes.jsonl")
    original_primary = [row for row in original if condition(row) in PRIMARY]
    deadline_root = args.campaign / "primary_deadline"
    deadline = read_jsonl(deadline_root / "episodes.jsonl")
    summary = json.loads((deadline_root / "summary.json").read_text())
    reconcile_rollout_manifest(deadline_root, summary, deadline)
    reconcile_rollout_journal(deadline_root, deadline)
    validate_filtered_journal(args.campaign / "confirmation", original_primary)

    rows = original_primary + deadline
    planned = json.loads((args.campaign / "confirmation_plan.json").read_text())["schedule"]
    expected = {row["planned_run_id"] for row in planned if condition(row) in PRIMARY}
    actual = [row["planned_run_id"] for row in rows]
    require(len(actual) == len(set(actual)) == 320 and set(actual) == expected,
            "primary comparison does not exactly cover the original 320 attempts")
    require(sum(condition(row) == (2, 2) for row in rows) == 160 and
            sum(condition(row) == (2, 5) for row in rows) == 160,
            "primary conditions are imbalanced")
    for row in rows:
        validate_episode(row)

    blocks = defaultdict(list)
    for row in rows:
        blocks[row["block_id"]].append(row)
    require(len(blocks) == 160, "expected 160 task-state-seed blocks")
    for block in blocks.values():
        require({condition(row) for row in block} == PRIMARY and len(block) == 2,
                "incomplete primary block")
        require(len({row["initial_state"]["sha256"] for row in block}) == 1 and
                len({row["initial_observation_sha256"] for row in block}) == 1,
                "paired initial state or observation differs")

    keyed = {}
    for row in rows:
        identity = row["episode_identity"]
        key = (identity["suite"], identity["task_id"], identity["initial_state_id"],
               identity["policy_seed"], *condition(row))
        keyed[key] = row
    tasks = sorted({(r["episode_identity"]["suite"], r["episode_identity"]["task_id"])
                    for r in rows})
    task_values = []
    for suite, task in tasks:
        task_values.append([
            np.mean([float(keyed[(suite, task, state, seed, 2, 2)]["success"])
                     - float(keyed[(suite, task, state, seed, 2, 5)]["success"])
                     for seed in (3000, 3001)])
            for state in range(20, 30)
        ])
    values = np.asarray(task_values)
    rng = np.random.default_rng(20260912)
    draws = sum(v[rng.integers(0, 10, size=(20000, 10))].mean(axis=1) / len(tasks)
                for v in values)
    costs = json.loads((args.campaign / "calibration/calibration.json").read_text())["generation_flops"]
    by_condition = []
    for h, d in sorted(PRIMARY):
        subset = [row for row in rows if condition(row) == (h, d)]
        by_condition.append({
            "h": h,
            "d": d,
            "attempts": len(subset),
            "successes": sum(row["success"] for row in subset),
            "success_rate": float(np.mean([row["success"] for row in subset])),
            "nominal_gflops_per_control_step": costs[str(d)] / h / 1e9,
            "mean_tflops_per_attempt": float(np.mean([
                row["counts"]["generations"] * costs[str(d)] / 1e12 for row in subset])),
            "mean_rollout_seconds_per_attempt": float(np.mean([
                row["timing"]["rollout_wall_s"] for row in subset])),
        })
    result = {
        "status": "audited_complete_frozen_primary_transfer_comparison",
        "comparison": {"selected": [2, 2], "comparator": [2, 5]},
        "frozen_predicted_difference": 0.04,
        "observed_paired_difference": float(values.mean()),
        "descriptive_95_percent_interval": np.quantile(draws, [.025, .975]).tolist(),
        "observed_minus_predicted": float(values.mean() - 0.04),
        "tasks": len(tasks),
        "task_state_blocks": 80,
        "task_state_seed_blocks": 160,
        "policy_seeds_per_state": 2,
        "conditions": by_condition,
        "scope": "Eight fixed transfer tasks; state bootstrap within task preserves both seeds and pairing. Descriptive interval; no non-inferiority or benchmark-wide claim.",
        "checks": ["original planned identity coverage", "condition balance", "paired initial states and observations", "generation and neural-stage counts", "complete action journals for included attempts", "deadline-run manifest and summary"],
        "sources": {
            "original_confirmation_episodes_sha256": file_sha(args.campaign / "confirmation/episodes.jsonl"),
            "deadline_episodes_sha256": file_sha(deadline_root / "episodes.jsonl"),
            "deadline_steps_sha256": file_sha(deadline_root / "steps.jsonl"),
            "deadline_schedule_amendment_sha256": file_sha(args.campaign / "deadline_schedule_amendment.json"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
