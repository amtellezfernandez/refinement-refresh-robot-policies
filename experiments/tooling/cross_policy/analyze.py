#!/usr/bin/env python3
"""Predeclared analysis from raw attempt records. Needs NumPy; never fits a selector.

Each model is analysed alone. A test is reported complete only if every planned identity has
exactly one completed attempt in both arms of its contrast; otherwise it is `incomplete`.
Missing or failed attempts are never imputed, dropped silently, or counted as failures.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, QUALITY_MARGIN_POINTS, SUITES  # noqa: E402
from journal import AttemptJournal  # noqa: E402


def index_attempts(schedule_rows, records):
    planned = {row["planned_run_id"]: row for row in schedule_rows}
    by_plan = defaultdict(list)
    for record in records:
        if record["planned_run_id"] not in planned:
            raise ValueError("record outside schedule: {}".format(record["planned_run_id"]))
        by_plan[record["planned_run_id"]].append(record)
    return planned, by_plan


def coverage(schedule_rows, records):
    planned, by_plan = index_attempts(schedule_rows, records)
    counts = defaultdict(int)
    missing, extra, not_completed = [], [], []
    for run_id in planned:
        attempts = by_plan.get(run_id, [])
        if not attempts:
            missing.append(run_id)
        if len(attempts) > 1:
            extra.append(run_id)
        for attempt in attempts:
            counts[attempt["status"]] += 1
            if attempt["status"] != "completed":
                not_completed.append({"planned_run_id": run_id, "attempt": attempt["attempt_number"],
                                      "status": attempt["status"]})
    return {"planned": len(planned), "records": len(records), "status_counts": dict(counts),
            "missing": missing, "identities_with_multiple_attempts": extra, "not_completed": not_completed}


def success_table(schedule_rows, records, condition):
    """Return {(suite, task, slot): {seed: success}}, or the identities that block it."""
    planned, by_plan = index_attempts(schedule_rows, records)
    table, blocked = defaultdict(dict), []
    for run_id, row in planned.items():
        if row["condition_id"] != condition:
            continue
        ident = row["episode_identity"]
        attempts = by_plan.get(run_id, [])
        if len(attempts) != 1 or attempts[0]["status"] != "completed":
            blocked.append(run_id)
            continue
        table[(ident["suite"], ident["task_id"], ident["state_slot"])][ident["policy_seed"]] = \
            bool(attempts[0]["outcome"]["success"])
    return table, blocked


def state_differences(table_a, table_b):
    """Mean seed-paired A-B per state, in percentage points, nested as suite -> task -> [values]."""
    nested = defaultdict(lambda: defaultdict(list))
    for key in sorted(table_a):
        seeds = sorted(table_a[key])
        if seeds != sorted(table_b[key]):
            raise ValueError("seed sets differ for {}".format(key))
        value = np.mean([float(table_a[key][s]) - float(table_b[key][s]) for s in seeds]) * 100.0
        nested[key[0]][key[1]].append(value)
    return nested


def macro(nested):
    return float(np.mean([np.mean([np.mean(v) for v in nested[s].values()]) for s in nested]))


def bootstrap(nested, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Resample states within each fixed task; weight tasks equally in a suite, suites equally."""
    rng = np.random.default_rng(seed)
    suite_means = []
    for suite in sorted(nested):
        task_means = []
        for task in sorted(nested[suite]):
            values = np.asarray(nested[suite][task])
            picks = rng.integers(0, len(values), size=(draws, len(values)))
            task_means.append(values[picks].mean(axis=1))
        suite_means.append(np.mean(task_means, axis=0))
    return np.mean(suite_means, axis=0)


def reset_state_failures(schedule_rows, records):
    """Completed attempts whose block lacks one settled-state hash. Recomputed; the runner's flag is not read.

    A missing hash counts as a failure, never as a match.
    """
    planned = {row["planned_run_id"]: row for row in schedule_rows}
    hashes, failures = defaultdict(set), set()
    for record in records:
        if record["status"] != "completed":
            continue
        run_id = record["planned_run_id"]
        settled = record.get("settled_state_sha256")
        if not settled:
            failures.add(planned[run_id]["block_id"])
        hashes[planned[run_id]["block_id"]].add(settled)
    return failures | {block for block, values in hashes.items() if len(values) > 1}


def contrast(schedule_rows, records, a, b):
    table_a, blocked_a = success_table(schedule_rows, records, a)
    table_b, blocked_b = success_table(schedule_rows, records, b)
    blocked = blocked_a + blocked_b
    reset_failures = reset_state_failures(schedule_rows, records)
    if blocked or reset_failures:
        return {"contrast": "{} minus {}".format(a, b), "status": "incomplete", "blocking_attempts": len(blocked),
                "reset_state_failures": len(reset_failures)}
    nested = state_differences(table_a, table_b)
    samples = bootstrap(nested)
    return {"contrast": "{} minus {}".format(a, b), "status": "complete", "macro_difference_points": macro(nested),
            "paired_state_differences": [{"suite": suite, "task_id": task, "state_position": index, "points": float(value)}
                                         for suite in sorted(nested) for task in sorted(nested[suite])
                                         for index, value in enumerate(nested[suite][task])],
            "ci95_two_sided": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "lower_bound_one_sided_95": float(np.percentile(samples, 5.0)),
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                          "structure": "states within fixed tasks; both seeds retained within each state"}}


def quality_decision(secondary):
    if secondary["status"] != "complete":
        return {"decision": "test_incomplete"}
    bound = secondary["lower_bound_one_sided_95"]
    if bound > QUALITY_MARGIN_POINTS:
        return {"decision": "quality_preserved_within_margin", "lower_bound": bound, "margin": QUALITY_MARGIN_POINTS,
                "wording": "Preserves quality within the stated engineering margin for this model and "
                           "evaluation population."}
    return {"decision": "quality_loss_or_inconclusive", "lower_bound": bound, "margin": QUALITY_MARGIN_POINTS}


def condition_summary(schedule_rows, records):
    conditions = sorted({row["condition_id"] for row in schedule_rows})
    result = {}
    for condition in conditions:
        table, blocked = success_table(schedule_rows, records, condition)
        per_suite = defaultdict(list)
        for key, seeds in table.items():
            per_suite[key[0]].extend(float(v) for v in seeds.values())
        result[condition] = {"successes": int(sum(sum(s.values()) for s in table.values())),
                             "completed": int(sum(len(s) for s in table.values())),
                             "blocked_attempts": len(blocked),
                             "per_suite_success_rate": {s: float(np.mean(per_suite[s])) for s in SUITES if per_suite[s]}}
        rates = result[condition]["per_suite_success_rate"]
        result[condition]["equal_suite_macro_success"] = float(np.mean(list(rates.values()))) if len(rates) == 4 else None
    return result


def timing_summary(schedule_rows, records):
    """Host, CUDA and rollout time stay separate. A quantity not recorded is UNAVAILABLE, never substituted."""
    condition = {row["planned_run_id"]: row["condition_id"] for row in schedule_rows}
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record["status"] != "completed":
            continue
        timing, target = record.get("timing") or {}, grouped[condition[record["planned_run_id"]]]
        if "rollout_wall_s" in timing:
            target["rollout_wall_s"].append(timing["rollout_wall_s"])
        target["generation_host_s"].extend(timing.get("generation_host_s", []))
        target["generation_cuda_ms"].extend(timing.get("generation_cuda_ms", []))
        for item in timing.get("generation", []):
            target["generation_host_s"].append(item["host_generation_s"])
    result = {}
    for name in sorted(set(condition.values())):
        result[name] = {key + "_median": float(np.median(grouped[name][key])) if grouped[name][key] else "UNAVAILABLE"
                        for key in ("rollout_wall_s", "generation_host_s", "generation_cuda_ms")}
    return result


ARITHMETIC_STATUS = ("NOT_IMPLEMENTED: arithmetic per chunk, control step and attempted episode awaits Alba's ruling on "
                     "R13; SmolVLA has a length-20 calibration and pi0.5 XLA cost analysis does not exist.")


def analyze_model(schedule_rows, records):
    primary = contrast(schedule_rows, records, "R22", "R25")
    secondary = contrast(schedule_rows, records, "R52", "R22")
    return {"coverage": coverage(schedule_rows, records), "conditions": condition_summary(schedule_rows, records),
            "timing": timing_summary(schedule_rows, records), "arithmetic": ARITHMETIC_STATUS,
            "reset_state_failures": sorted(reset_state_failures(schedule_rows, records)),
            "primary": primary, "secondary": secondary, "quality_preservation": quality_decision(secondary)}


def cross_model_summary(reports):
    """Descriptive only: sign agreement. Episodes are not exchangeable across models, so nothing is pooled."""
    signs = {}
    for model, report in reports.items():
        primary = report["primary"]
        if primary["status"] != "complete":
            return {"sign_agreement": "not_available", "reason": "{} incomplete".format(model)}
        signs[model] = float(np.sign(primary["macro_difference_points"]))
    return {"primary_signs": signs, "sign_agreement": "agree" if len(set(signs.values())) == 1 else "disagree",
            "scope": "Descriptive. No pooled estimate exists."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["smolvla", "pi05"])
    args = parser.parse_args()
    reports = {}
    for model in args.models:
        schedule = json.loads((args.run_root / model / "schedule.json").read_text("utf-8"))["schedule"]
        records = AttemptJournal(args.run_root / model / "attempts").records()
        reports[model] = analyze_model(schedule, records)
    print(json.dumps({"models": reports, "cross_model": cross_model_summary(reports)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
