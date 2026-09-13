#!/usr/bin/env python3
"""Audit the completed confirmation snapshot and regenerate manuscript evidence.

No model execution, network access, selector refitting, or new comparisons.
Original tests use seed 20260912; amended descriptive tests use 20260913.
The archived continuation completes the original design, not a replication.
"""

import argparse
import tarfile
import tempfile
from decimal import Decimal, ROUND_HALF_UP
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_execution import (
    canonical_sha,
    check_initial_state,
    file_sha,
    read_jsonl,
    reconcile_rollout_journal,
    reconcile_rollout_manifest,
)
from analyze_primary_deadline import validate_episode, validate_filtered_journal

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "experiments/runs/refinement_refresh_final_001"


def read(path):
    return json.loads(path.read_text())


def condition(r):
    return (r["h"], r["denoising_steps"])


def identity(r):
    e = r["episode_identity"]
    return (e["suite"], e["task_id"], e["initial_state_id"], e["policy_seed"])


def analyze(root):
    phases = [
        "confirmation",
        "primary_deadline",
        "confirmation_continuation_partial",
        "confirmation_continuation_rest",
    ]
    plan = read(root / "confirmation_plan.json")
    schedule = plan["schedule"]
    assert canonical_sha(schedule) == plan["scheduling"]["schedule_sha256"]
    expected = {r["planned_run_id"]: r for r in schedule}
    assert len(expected) == len(schedule) == 1120
    amendment = read(root / "continuation_dedup_amendment.json")
    assert amendment["canonical_precedence"] == phases
    kept, source, repeats, manifests, phase_rows, audit = {}, {}, [], {}, {}, []
    seen_runs = set()
    for phase in phases:
        rows = read_jsonl(root / phase / "episodes.jsonl")
        runtime = read(root / phase / "runtime_manifest.json")
        phase_rows[phase], manifests[phase] = rows, runtime
        mh = canonical_sha(runtime)
        selected = runtime["selected_schedule"]
        assert runtime["selected_schedule_sha256"] == canonical_sha(selected)
        assert read(root / phase / "selected_schedule.json") == selected
        selected_by_id = {r["planned_run_id"]: r for r in selected}
        for r in rows:
            key = r["planned_run_id"]
            assert key in expected and key in selected_by_id
            assert r["run_id"] not in seen_runs
            seen_runs.add(r["run_id"])
            assert all(
                r[k] == expected[key][k] == selected_by_id[key][k]
                for k in ["episode_identity", "h", "denoising_steps", "block_id"]
            )
            assert r["runtime_manifest_sha256"] == mh
            assert r["status"] == "completed" and type(r["success"]) is bool
            assert 0 < r["steps"] <= r["task_horizon"]
            assert r["success"] == r["independent_driver"]["success"]
            check_initial_state(r["initial_state"])
            validate_episode(r)
            g = r["counts"]["generations"]
            for k in ["generation_host_s", "generation_cuda_ms"]:
                a = np.asarray(r["timing"][k])
                assert len(a) == g and np.isfinite(a).all() and (a >= 0).all()
            if key in kept:
                prev = kept[key]
                repeats.append(
                    dict(
                        id=key,
                        retained_phase=source[key],
                        excluded_phase=phase,
                        retained_success=prev["success"],
                        excluded_success=r["success"],
                        equal_action_stream=prev["action_stream_sha256"]
                        == r["action_stream_sha256"],
                    )
                )
            else:
                kept[key], source[key] = r, phase
        summary = root / phase / "summary.json"
        if summary.exists():
            reconcile_rollout_manifest(root / phase, read(summary), rows)
        audit.append(
            dict(phase=phase, raw_episodes=len(rows), summary_present=summary.exists())
        )
    assert set(kept) == set(expected) and len(repeats) == 2
    canonical = read_jsonl(root / "confirmation_canonical_episodes.jsonl")
    assert canonical == [kept[key] for key in expected]
    remote_audit = read(root / "confirmation_coverage_audit.json")
    assert remote_audit["sources"] == source

    cells = sorted({condition(r) for r in kept.values()})
    assert cells == [(1, 2), (1, 5), (1, 10), (2, 2), (2, 5), (5, 2), (5, 5)]
    blocks = defaultdict(dict)
    for r in kept.values():
        assert condition(r) not in blocks[identity(r)]
        blocks[identity(r)][condition(r)] = r
    assert len(blocks) == 160
    for block in blocks.values():
        assert set(block) == set(cells)
        for field in [
            "initial_observation_sha256",
            "environment_seed",
            "task_bddl_sha256",
            "initial_state_set_sha256",
            "task_horizon",
        ]:
            assert len({r[field] for r in block.values()}) == 1, field
        assert len({r["initial_state"]["sha256"] for r in block.values()}) == 1
    tasks = sorted({i[:2] for i in blocks})
    assert tasks == [
        (s, t) for s in ["libero_goal", "libero_object"] for t in range(5, 9)
    ]
    assert set(blocks) == {
        (*t, s, p) for t in tasks for s in range(20, 30) for p in [3000, 3001]
    }

    provenance = {}
    for field in [
        "python",
        "torch_version",
        "cuda_version",
        "numpy_version",
        "mujoco_version",
        "precision",
        "native_checkpoint_config",
        "effective_policy_config",
        "checkpoint_expected_revision",
        "checkpoint_files_sha256",
        "lerobot_expected_revision",
        "verified_pinned_source_sha256",
        "control",
        "noise_contract",
        "cost_boundary",
        "device_name",
    ]:
        equal = len({canonical_sha(m[field]) for m in manifests.values()}) == 1
        provenance[field] = dict(equal=equal)
        if not equal:
            provenance[field]["by_phase"] = {p: m[field] for p, m in manifests.items()}
    provenance["runner_hashes"] = {
        p: {Path(k).name: v for k, v in m["source_sha256"].items()}
        for p, m in manifests.items()
    }
    assert len({canonical_sha(v) for v in provenance["runner_hashes"].values()}) == 1
    assert all(
        m["argv"]["observation_refresh"] == 1
        and m["argv"]["warmup"] == 20
        and m["argv"]["image_size"] == 360
        for m in manifests.values()
    )
    assert all(
        provenance[f]["equal"]
        for f in [
            "precision",
            "checkpoint_expected_revision",
            "checkpoint_files_sha256",
            "effective_policy_config",
            "verified_pinned_source_sha256",
            "control",
            "noise_contract",
            "cost_boundary",
        ]
    )

    costs = read(root / "calibration/calibration.json")["generation_flops"]
    frozen = read(root / "frozen_prediction.json")
    discovery = read_jsonl(root / "discovery/episodes.jsonl")
    assert file_sha(root / "discovery/episodes.jsonl") == frozen["discovery_sha256"]
    # Each observation is a mean over the two paired policy seeds within a state.
    # States are resampled within fixed tasks, with all conditions kept together.
    values = np.zeros((8, 10, 7, 4))
    per_task = []
    for ti, t in enumerate(tasks):
        counts = {}
        for ci, c in enumerate(cells):
            rows = [blocks[(*t, s, p)][c] for s in range(20, 30) for p in [3000, 3001]]
            counts[str(c)] = sum(r["success"] for r in rows)
            for si, s in enumerate(range(20, 30)):
                pair = [blocks[(*t, s, p)][c] for p in [3000, 3001]]
                values[ti, si, ci] = np.mean(
                    [
                        [
                            r["success"],
                            r["counts"]["generations"] * costs[str(c[1])] / 1e12,
                            r["steps"],
                            r["timing"]["rollout_wall_s"],
                        ]
                        for r in pair
                    ],
                    axis=0,
                )
        per_task.append(
            dict(suite=t[0], task=t[1], successes=counts, attempts_per_cell=20)
        )
    est = values.mean(axis=(0, 1))
    resources = []
    for ci, c in enumerate(cells):
        rows = [r for r in kept.values() if condition(r) == c]
        resources.append(
            dict(
                h=c[0],
                d=c[1],
                n=len(rows),
                successes=sum(r["success"] for r in rows),
                success_rate=est[ci, 0],
                nominal_rate_gflops=costs[str(c[1])] / c[0] / 1e9,
                nominal_fraction_native=costs[str(c[1])] / c[0] / costs["10"],
                mean_reference_tflops=est[ci, 1],
                mean_steps=est[ci, 2],
                observed_mean_seconds=est[ci, 3],
                actual_reference_gflops_per_step=sum(
                    r["counts"]["generations"] for r in rows
                )
                * costs[str(c[1])]
                / sum(r["steps"] for r in rows)
                / 1e9,
            )
        )

    def contrast(a, b, rng, kind, prediction=None):
        ia, ib = cells.index(a), cells.index(b)
        diffs = values[:, :, ia] - values[:, :, ib]
        samples = sum(
            v[rng.integers(0, 10, (20000, 10))].mean(axis=1) / 8 for v in diffs
        )
        paired = [(v[a]["success"], v[b]["success"]) for v in blocks.values()]
        return dict(
            a=a,
            b=b,
            kind=kind,
            prediction=prediction,
            mean_difference=diffs.mean(axis=(0, 1)).tolist(),
            interval95=np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
            wins=sum(x and not y for x, y in paired),
            losses=sum(y and not x for x, y in paired),
            per_task_difference=diffs[:, :, 0].mean(axis=1).tolist(),
            leave_one_task_out=[
                float(np.delete(diffs[:, :, 0], i, axis=0).mean()) for i in range(8)
            ],
        )

    comparisons = []
    rng = np.random.default_rng(20260912)
    for c in frozen["comparisons"]:
        comparisons.append(
            contrast(
                tuple(c["recommendation"]),
                tuple(c["alternative"]),
                rng,
                "original primary" if c["primary"] else "original secondary",
                c["predicted_difference"],
            )
        )
    # Reproduce the exact original primary RNG sequence independently.
    primary_exact = contrast(
        (2, 2), (2, 5), np.random.default_rng(20260912), "submitted primary"
    )
    assert np.isclose(primary_exact["mean_difference"][0], 0.10625)
    assert np.allclose(primary_exact["interval95"][0], [0.04375, 0.16875])
    assert [r["successes"] for r in resources if r["h"] == 2] == [139, 122]
    rng = np.random.default_rng(20260913)
    for a, b in [
        ((2, 2), (1, 10)),
        ((2, 2), (5, 2)),
        ((1, 2), (2, 2)),
        ((1, 2), (1, 10)),
    ]:
        comparisons.append(contrast(a, b, rng, "amended descriptive secondary"))
    # Full per-action checks for completed attempts, including both excluded repeats.
    # The interrupted original file contains a trailing incomplete attempt; retain and
    # enumerate it, but only completed episodes belong in the outcome analysis.
    for phase in phases:
        rows = phase_rows[phase]
        if phase == "confirmation":
            events = read_jsonl(root / phase / "steps.jsonl")
            included = {r["run_id"] for r in rows}
            unknown = [e for e in events if e["run_id"] not in included]
            assert unknown and events[-len(unknown) :] == unknown
            assert len({e["run_id"] for e in unknown}) == 1
            assert unknown[0]["kind"] == "episode_start"
            assert all(e["kind"] == "step" for e in unknown[1:])
            audit[0]["trailing_incomplete_attempt"] = dict(
                run_id=unknown[0]["run_id"],
                step_events=len(unknown) - 1,
                identity=unknown[0]["episode_identity"],
            )
            del events, unknown
            validate_filtered_journal(root / phase, rows)
        else:
            reconcile_rollout_journal(root / phase, rows)
        audit[phases.index(phase)]["dispatched_actions"] = sum(r["steps"] for r in rows)
        audit[phases.index(phase)]["full_completed_journal_check"] = "passed"

    result = dict(
        raw_records=sum(map(len, phase_rows.values())),
        unique_attempts=len(kept),
        paired_seed_blocks=len(blocks),
        task_state_blocks=80,
        discovery_attempts=len(discovery),
        canonical_actions=sum(r["steps"] for r in kept.values()),
        repeats=repeats,
        audit=audit,
        runtime_comparison=provenance,
        resources=resources,
        comparisons=comparisons,
        exact_submitted_primary=primary_exact,
        per_task=per_task,
        metrics=[
            "success",
            "reference_tflops",
            "control_steps",
            "observed_rollout_seconds",
        ],
        bootstrap="20000 state resamples within each of eight fixed tasks; paired seeds averaged; equal task weights",
        limits=[
            "Descriptive task-conditional intervals; secondary intervals not multiplicity adjusted",
            "Reference-calibrated arithmetic; instruction shape varies by task",
            "Timing monitoring incomplete across phases; no clean elapsed-efficiency claim",
            "Continuation completes original protocol; it is not a replication cohort",
        ],
        input_sha256={
            str(p.relative_to(root)): file_sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
        analysis_sha256=file_sha(Path(__file__)),
        validator_sha256={
            name: file_sha(Path(__file__).with_name(name))
            for name in [
                "analyze_execution.py",
                "analyze_primary_deadline.py",
                "run_paired.py",
            ]
        },
    )
    return result


def extract_snapshot(snapshot, destination):
    """Reject traversal, links and special files before extracting any member."""
    with tarfile.open(snapshot, "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)
        archive.extractall(destination, members=members, filter="data")


def number(value, signed=False):
    # Remove sub-picounit floating-point noise before rounding exact half-ties.
    rounded = Decimal(str(round(value, 10))).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
    return format(rounded, "+.1f" if signed else ".1f")


def interval(comparison):
    return (
        "["
        + ", ".join(
            number(100 * value, signed=True) for value in comparison["interval95"][0]
        )
        + "]"
    )


def latex_table(columns, header, rows):
    return "\n".join(
        [
            "% Generated by experiments/tooling/analyze_confirmation.py; do not edit.",
            r"\begin{tabular}{@{}" + columns + r"@{}}",
            r"\toprule",
            header + r"\\",
            r"\midrule",
            *(" & ".join(row) + r"\\" for row in rows),
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def configuration(pair):
    return f"$({pair[0]},{pair[1]})$"


def outputs(report):
    resources = latex_table(
        "rrrrrr",
        r"$(h,d)$ & Successes & Success (\%) & Rate/native & TFLOPs/attempt & Steps/attempt",
        [
            [
                configuration((r["h"], r["d"])),
                f"{r['successes']}/{r['n']}",
                number(100 * r["success_rate"]),
                f"{r['nominal_fraction_native']:.3f}",
                number(r["mean_reference_tflops"]),
                number(r["mean_steps"]),
            ]
            for r in report["resources"]
        ],
    )
    predictions = latex_table(
        "rllrrl",
        r"Ceiling/native & Selected & Comparator & Predicted & Observed & 95\% interval",
        [
            [
                budget,
                configuration(c["a"]),
                configuration(c["b"]),
                number(100 * c["prediction"], signed=True),
                number(100 * c["mean_difference"][0], signed=True),
                interval(c),
            ]
            for budget, c in zip(
                ["0.25", "0.50 (primary)", "1.00"], report["comparisons"][:3]
            )
        ],
    )
    secondary = latex_table(
        "llrl",
        r"Configuration A & Configuration B & Success A$-$B (pp) & 95\% interval (pp)",
        [
            [
                configuration(c["a"]),
                configuration(c["b"]),
                number(100 * c["mean_difference"][0], signed=True),
                interval(c),
            ]
            for c in report["comparisons"][3:]
        ],
    )
    cells = [(r["h"], r["d"]) for r in report["resources"]]
    task_table = latex_table(
        "l" + "r" * 7,
        "Task & " + " & ".join(configuration(c) for c in cells),
        [
            [
                t["suite"].replace("libero_", "").title() + f" {t['task']}",
                *(str(t["successes"][str(c)]) for c in cells),
            ]
            for t in report["per_task"]
        ],
    )
    return {
        "confirmation_report.json": json.dumps(report, indent=2, allow_nan=False)
        + "\n",
        "confirmation_resource_table.tex": resources,
        "confirmation_prediction_table.tex": predictions,
        "confirmation_secondary_table.tex": secondary,
        "confirmation_task_table.tex": task_table,
    }


def main():
    if not __debug__:
        raise RuntimeError("Run without -O: evidence validation requires assertions.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=CAMPAIGN / "confirmation_completed.tar.gz"
    )
    parser.add_argument("--output", type=Path, default=CAMPAIGN / "paper_analysis")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Replay all checks and compare saved outputs without editing them.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="confirmation-audit-") as directory:
        extract_snapshot(args.snapshot, Path(directory))
        report = analyze(Path(directory))
    report["snapshot_sha256"] = file_sha(args.snapshot)
    expected = outputs(report)
    if args.check:
        stale = [
            name
            for name, content in expected.items()
            if not (args.output / name).is_file()
            or (args.output / name).read_text(encoding="utf-8") != content
        ]
        if stale:
            raise ValueError(
                "Missing or stale confirmation artifacts: " + ", ".join(stale)
            )
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in expected.items():
            (args.output / name).write_text(content, encoding="utf-8")
    print(
        f"CONFIRMATION VERIFIED: {report['unique_attempts']} unique attempts; "
        f"{len(report['repeats'])} excluded repeats; "
        f"{report['canonical_actions']} canonical actions; "
        f"{len(expected)} outputs {'match' if args.check else 'written'}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
