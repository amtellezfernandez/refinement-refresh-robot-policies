#!/usr/bin/env python3
"""Audit archived outcome recording without importing project analyzers."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path


PHASES = (
    "confirmation",
    "primary_deadline",
    "confirmation_continuation_partial",
    "confirmation_continuation_rest",
)
EXPECTED_LEROBOT_LIBERO_SHA256 = (
    "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(archive: tarfile.TarFile, name: str):
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"archive member missing: {name}")
    return [json.loads(line) for line in stream if line.strip()]


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def audit(path: Path) -> dict:
    totals = collections.Counter()
    reward_success = collections.Counter()
    phase_results = []
    with tarfile.open(path, "r:gz") as archive:
        for phase in PHASES:
            episodes = json_lines(archive, f"{phase}/episodes.jsonl")
            events = json_lines(archive, f"{phase}/steps.jsonl")
            manifest = json.loads(
                archive.extractfile(f"{phase}/runtime_manifest.json").read()
            )
            source_hash = manifest["verified_pinned_source_sha256"]["envs/libero.py"]
            require(
                source_hash == EXPECTED_LEROBOT_LIBERO_SHA256,
                f"{phase}: unexpected pinned LeRobot LIBERO source",
            )

            streams = collections.defaultdict(
                lambda: {"start": None, "steps": [], "completed": None}
            )
            for event in events:
                record = streams[event["run_id"]]
                if event["kind"] == "episode_start":
                    record["start"] = event
                elif event["kind"] == "step":
                    record["steps"].append(event)
                elif event["kind"] == "episode_completed":
                    record["completed"] = event
                else:
                    raise ValueError(f"{phase}: unknown event kind {event['kind']}")

            completed_ids = {episode["run_id"] for episode in episodes}
            phase_steps = phase_successes = 0
            for episode in episodes:
                run_id = episode["run_id"]
                stream = streams[run_id]
                start, steps, completed = (
                    stream["start"],
                    stream["steps"],
                    stream["completed"],
                )
                require(start is not None, f"{run_id}: missing start event")
                require(completed is not None, f"{run_id}: missing completion event")
                require(
                    len(steps)
                    == episode["steps"]
                    == episode["independent_driver"]["steps"],
                    f"{run_id}: step-count disagreement",
                )
                require(
                    all(
                        step["driver_is_success"] == step["vector_is_success"]
                        for step in steps
                    ),
                    f"{run_id}: driver/vector success disagreement",
                )
                require(
                    all(
                        step["producer_action"]["sha256"]
                        == step["environment_action_sha256"]
                        for step in steps
                    ),
                    f"{run_id}: producer/consumer action disagreement",
                )
                driver_success = any(step["driver_is_success"] for step in steps)
                vector_success = any(step["vector_is_success"] for step in steps)
                require(
                    episode["success"]
                    == driver_success
                    == vector_success
                    == episode["independent_driver"]["success"]
                    == completed["success"],
                    f"{run_id}: episode outcome disagreement",
                )
                require(
                    episode["steps"] == completed["steps"],
                    f"{run_id}: completion step disagreement",
                )
                require(
                    episode["episode_identity"] == start["episode_identity"],
                    f"{run_id}: start identity disagreement",
                )
                require(
                    episode["initial_state"]["sha256"]
                    == start["initial_state"]["sha256"],
                    f"{run_id}: initial state disagreement",
                )
                require(
                    episode["initial_observation_sha256"]
                    == start["initial_observation_sha256"],
                    f"{run_id}: initial observation disagreement",
                )
                require(
                    (episode["termination_reason"] == "success") == episode["success"],
                    f"{run_id}: termination reason disagreement",
                )
                for step in steps:
                    reward_success[(step["driver_is_success"], step["reward"])] += 1
                phase_steps += len(steps)
                phase_successes += int(episode["success"])

            extra_streams = [run_id for run_id in streams if run_id not in completed_ids]
            totals.update(
                episodes=len(episodes),
                steps=phase_steps,
                successes=phase_successes,
                extra_incomplete_streams=len(extra_streams),
            )
            phase_results.append(
                {
                    "phase": phase,
                    "episodes": len(episodes),
                    "steps": phase_steps,
                    "successes": phase_successes,
                    "extra_incomplete_streams": len(extra_streams),
                    "lerobot_libero_source_sha256": source_hash,
                }
            )

    require(
        set(reward_success) == {(False, 0.0), (True, 1.0)},
        f"unexpected reward/success combinations: {dict(reward_success)}",
    )
    return {
        "status": "passed",
        "scope": (
            "Internal outcome-path consistency; success fields and sparse reward "
            "share LIBERO's task predicate and are not independent outcome oracles. "
            "This is not an independent simulator replay."
        ),
        "archive": str(path),
        "archive_sha256": file_sha256(path),
        "phases": phase_results,
        "totals": dict(totals),
        "reward_success_crosscheck": [
            {"success": success, "reward": reward, "steps": count}
            for (success, reward), count in sorted(reward_success.items())
        ],
        "checks": {
            "start_and_completion_events": "all completed episodes",
            "step_counts": "episode, driver, and stream agree",
            "success": (
                "episode, driver, vector evaluator, and completion recordings "
                "of the same upstream predicate agree"
            ),
            "actions": "producer and environment-consumer hashes agree at every step",
            "pairing": "start identity, initial state, and initial observation agree",
            "reward": (
                "0 for every nonsuccess step and 1 for every success step; "
                "consistency only because sparse reward uses the same predicate"
            ),
            "pinned_lerobot_libero_source": EXPECTED_LEROBOT_LIBERO_SHA256,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
