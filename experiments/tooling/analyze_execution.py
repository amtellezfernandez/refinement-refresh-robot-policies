#!/usr/bin/env python3
"""Generate descriptive tables/figures from completed real execution records.

No archival data are pooled with new runs. Bootstrap intervals condition on the
declared task set and preserve paired conditions and policy seeds within states.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Fixed before applying this mechanism check to any completed rollout set. The
# protocol is a reproducible code-defined test, not external preregistration.
QUERY_RECOVERY_PROTOCOL = {
    "version": 1,
    "design": "Deterministic direct-record query recovery under artificial information suppression",
    "queries": [
        {"id": "success_attempts", "label": "Successes / attempts", "unit": "count / count",
         "boundary": "All completed attempted episodes in the declared condition, including failures",
         "required_fields": ["success"]},
        {"id": "generations", "label": "Actual chunk generations", "unit": "count",
         "boundary": "Measured rollout generations; excludes model loading, reset and warm-up",
         "required_fields": ["generations"]},
        {"id": "denoise_nfe", "label": "Measured denoise evaluations", "unit": "count",
         "boundary": "Actual denoise-method entries in measured rollouts; excludes warm-up",
         "required_fields": ["denoise_nfe"]},
        {"id": "rollout_wall_s", "label": "Total outer rollout wall time", "unit": "s",
         "boundary": "Sum of outer rollout intervals over all attempts; runtime manifest defines included stages",
         "required_fields": ["rollout_wall_s"]},
        {"id": "wall_per_attempt_s", "label": "Mean wall time / attempt", "unit": "s / attempt",
         "boundary": "Total outer rollout wall time / all completed attempts; raw pooled mean, not equal-task weighting",
         "required_fields": ["rollout_wall_s"]},
        {"id": "wall_per_success_s", "label": "All-attempt wall time / success", "unit": "s / success",
         "boundary": "Total outer rollout wall time, including failed attempts, / successes; undefined at zero successes",
         "required_fields": ["rollout_wall_s", "success"],
         "zero_denominator_rule": "Known zero successes establishes undefined even when wall time is suppressed"},
    ],
    "projections": [
        {"id": "outcomes_only", "short": "O", "fields": ["attempt_id", "success"]},
        {"id": "outcomes_generations", "short": "G", "fields": ["attempt_id", "success", "generations"]},
        {"id": "full_ledger", "short": "L", "fields": ["attempt_id", "success", "generations", "denoise_nfe", "rollout_wall_s"]},
    ],
    "reference": "Completed records independently reconciled against retained step journals by this analyzer",
    "information_contract": "Reduced projections omit solver configuration, source, action-step schedules and timing. Condition identifiers are opaque to the recovery function. Only direct retained fields and the declared arithmetic are queried; the check does not assert that NFE is impossible to derive when solver configuration/source are available.",
    "status_contract": "recovered = defined value obtained; undefined = enough information establishes zero denominator; unknown = required information suppressed. Recovered and undefined count as answered when they match the reference.",
    "limits": "Finite deterministic mechanism check, not a reader study, natural disclosure experiment, independent empirical benefit estimate, or evidence of improved decisions. Full-projection success is a consistency check of deliberately retained fields, not a publication-level utility claim.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_number(value, allow_zero=False):
    return type(value) in (float, int) and np.isfinite(value) and (value >= 0 if allow_zero else value > 0)


def check_initial_state(record):
    require(isinstance(record, dict) and all(k in record for k in ("sha256", "dtype", "shape", "values")),
            "Missing paired initial-state record/hash")
    require(valid_hash(record["sha256"]), "Invalid paired initial-state hash")
    try:
        array = np.asarray(record["values"], dtype=record["dtype"])
    except (TypeError, ValueError):
        raise ValueError("Invalid initial-state array") from None
    require(array.dtype.kind in "fiu" and np.isfinite(array).all() and array.size > 0,
            "Invalid initial-state values")
    require(list(array.shape) == record["shape"], "Initial-state shape mismatch")
    header = json.dumps([array.dtype.str, list(array.shape)], separators=(",", ":")).encode()
    digest = hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()
    require(digest == record["sha256"], "Initial-state content/hash mismatch")


def validate_microbenchmark(root, completion):
    """Reconcile the actual benchmark_inference.py schema before reducing data."""
    require(completion.get("status") == "complete", "Microbenchmark must be complete before analysis")
    config = completion.get("config", {})
    blocks, calls, warmup = (config.get(k) for k in ("blocks", "calls", "warmup_calls"))
    require(type(blocks) is int and blocks > 0 and type(calls) is int and calls > 0 and
            type(warmup) is int and warmup >= 0, "Missing/invalid declared microbenchmark allocation")
    solvers = config.get("num_steps")
    require(isinstance(solvers, list) and solvers and all(type(d) is int and d > 0 for d in solvers)
            and len(set(solvers)) == len(solvers), "Invalid declared solver settings")
    expected = {(block, d, phase, index) for block in range(blocks) for d in solvers
                for phase, n in (("warmup", warmup), ("measurement", calls)) for index in range(n)}
    pairs = read_jsonl(root / "pairs.jsonl")
    keyed = {}
    for row in pairs:
        key = tuple(row.get(k) for k in ("block", "num_steps", "phase", "call_index"))
        require(key in expected and key not in keyed, "Unexpected/duplicate inference pair")
        require(row.get("pair_id") == f"b{key[0]}:d{key[1]}:{key[2]}:{key[3]}", "Inference pair ID mismatch")
        require(type(row.get("allclose")) is bool and row["allclose"],
                "Instrumented prediction parity failed or missing")
        require(type(row.get("exact_equal")) is bool, "Invalid exact-parity flag")
        require(all(valid_hash(row.get(f"{mode}_action_sha256")) for mode in ("off", "on")),
                "Missing action hashes for instrumentation pair")
        require(row["exact_equal"] == (row["off_action_sha256"] == row["on_action_sha256"]),
                "Exact-parity flag disagrees with action hashes")
        require(positive_number(row.get("max_abs_diff"), True), "Invalid recorded action difference")
        require(not row["exact_equal"] or row["max_abs_diff"] == 0, "Exact outputs have nonzero difference")
        for mode in ("off", "on"):
            require(positive_number(row.get(f"{mode}_host_elapsed_ms")), "Invalid host prediction timing")
            require(positive_number(row.get(f"{mode}_cuda_elapsed_ms"), True), "Invalid CUDA prediction timing")
        for tolerance in ("rtol", "atol"):
            require(positive_number(row.get(tolerance), True) and row[tolerance] == config.get(tolerance),
                    "Pair tolerance differs from declared configuration")
        keyed[key] = row
    require(set(keyed) == expected, "Missing declared solver/block/measurement pair")
    counts = completion.get("counts", {})
    expected_counts = {"completed_actual_chunk_generations": len(expected) * 2,
                       "issued_prediction_calls": len(expected) * 2,
                       "completed_pairs": len(expected), "unpaired_calls": 0,
                       "measured_calls": blocks * len(solvers) * calls * 2,
                       "warmup_calls": blocks * len(solvers) * warmup * 2,
                       "pairs_allclose": len(expected),
                       "pairs_exact_equal": sum(row["exact_equal"] for row in pairs)}
    require(all(type(counts.get(k)) is int and counts[k] == v for k, v in expected_counts.items()),
            "Microbenchmark summary counts disagree with declared pairs")
    require(completion.get("planned_actual_chunk_generations") == len(expected) * 2,
            "Microbenchmark planned allocation disagrees")
    trial_keys = set()
    trials = read_jsonl(root / "trials.jsonl")
    observation_hash = completion.get("observation", {}).get("observation_sha256")
    require(valid_hash(observation_hash), "Missing fixed-observation hash")
    paired_trials = {}
    for row in trials:
        key = tuple(row.get(k) for k in ("block", "num_steps", "phase", "call_index"))
        mode = row.get("instrumentation")
        require(key in expected and mode in ("off", "on") and (key, mode) not in trial_keys,
                "Unexpected/duplicate instrumentation trial")
        pair = keyed[key]
        require(row.get("pair_id") == pair["pair_id"], "Trial/pair identity mismatch")
        require(row.get("observation_sha256") == observation_hash and valid_hash(row.get("noise_sha256")),
                "Missing or inconsistent paired observation/noise hash")
        require(row.get("action_sha256") == pair[f"{mode}_action_sha256"], "Trial/pair action hash mismatch")
        require(all(row.get(f"{clock}_elapsed_ms") == pair[f"{mode}_{clock}_elapsed_ms"]
                    for clock in ("host", "cuda")), "Trial/pair timing mismatch")
        noise_key = f"robopad-fixed-observation-v1:{config.get('seed')}:{key[0]}:{key[2]}:{key[3]}"
        expected_seed = int.from_bytes(hashlib.sha256(noise_key.encode()).digest()[:8], "big") % (2**63 - 1)
        require(type(config.get("seed")) is int and row.get("noise_seed") == expected_seed,
                "Trial noise seed differs from declared key")
        require(row.get("action_shape") == [1, 50, 7] and row.get("action_dtype") == "torch.float32",
                "Unexpected complete checkpoint action shape/dtype")
        require(type(row.get("action_index")) is int and row["action_index"] == len(trial_keys),
                "Action artifact indices are missing, duplicated, or out of journal order")
        hook_counts = row.get("instrumentation_counts")
        stages = row.get("stage_cuda_ms")
        if mode == "on":
            expected_hooks = {"chunk_entries": 1, "denoise_nfe": key[1],
                              "prefill_forwards": 1, "action_expert_forwards": key[1]}
            require(isinstance(hook_counts, dict) and all(type(hook_counts.get(k)) is int and hook_counts[k] == v
                    for k, v in expected_hooks.items()) and type(hook_counts.get("vision_encoder_calls")) is int
                    and hook_counts["vision_encoder_calls"] >= 1, "Measured inference work counts disagree")
            require(isinstance(stages, dict) and set(stages) == {"embed_prefix", "vision_encoder", "prefill", "action_expert"}
                    and all(positive_number(v, True) for v in stages.values()), "Invalid measured stage timings")
        else:
            require(hook_counts is None and stages is None, "Off-mode unexpectedly reports active model hooks")
        paired_trials[(key, mode)] = row
        trial_keys.add((key, mode))
    require(trial_keys == {(key, mode) for key in expected for mode in ("off", "on")},
            "Missing on/off instrumentation trial")
    require(all(paired_trials[(key, "off")]["noise_sha256"] == paired_trials[(key, "on")]["noise_sha256"]
                for key in expected), "Instrumentation pair noise differs")
    for block in range(blocks):
        for phase, n in (("warmup", warmup), ("measurement", calls)):
            for index in range(n):
                require(len({paired_trials[((block, d, phase, index), "off")]["noise_sha256"] for d in solvers}) == 1,
                        "Solver settings received different noise despite shared-noise contract")
    artifact = completion.get("actions_artifact", {})
    require(artifact.get("path") == "actions.pt" and artifact.get("count") == len(trials)
            and artifact.get("index_field") == "action_index" and valid_hash(artifact.get("sha256")),
            "Missing complete prediction tensor artifact metadata")
    require((root / "actions.pt").is_file() and file_sha(root / "actions.pt") == artifact["sha256"],
            "Prediction tensor artifact missing or digest differs")
    return [row for row in pairs if row["phase"] == "measurement"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_figure(fig, output, stem):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def microbenchmark(root, output):
    completion = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    pairs = validate_microbenchmark(root, completion)
    blocks = sorted({row["block"] for row in pairs})
    solvers = sorted({row["num_steps"] for row in pairs})
    report = {"scope": "fixed real simulator observation; prediction microbenchmark, not rollout quality",
              "source": str(root), "paired_measurements": len(pairs),
              "all_exact": all(row["exact_equal"] for row in pairs), "cells": [],
              "validation_scope": "Declared allocation, on/off and cross-solver noise keys/hashes, actual hook counts, clock records, tensor-artifact file digest. Tensor values/parity were checked by the execution runner; this NumPy-only analysis does not deserialize actions.pt.",
              "analysis_script_sha256": file_sha(Path(__file__))}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for index, d in enumerate(solvers):
        group = [row for row in pairs if row["num_steps"] == d]
        ratios, off_medians, on_medians = [], [], []
        for block in blocks:
            rows = [row for row in group if row["block"] == block]
            if not rows:
                raise ValueError("Unbalanced benchmark blocks")
            off = float(np.median([r["off_host_elapsed_ms"] for r in rows]))
            on = float(np.median([r["on_host_elapsed_ms"] for r in rows]))
            off_medians.append(off)
            on_medians.append(on)
            ratios.append(100 * (on / off - 1))
        report["cells"].append({"num_steps": d, "blocks": len(blocks), "pairs": len(group),
                                "off_block_median_ms": off_medians,
                                "on_block_median_ms": on_medians,
                                "overhead_percent_per_block": ratios,
                                "median_block_overhead_percent": float(np.median(ratios))})
        for block_index, (off, on) in enumerate(zip(off_medians, on_medians)):
            axes[0].plot([index - .13, index + .13], [off, on], color=colors[index], alpha=.55)
        axes[0].scatter([index - .13] * len(blocks), off_medians, color=colors[index], marker="o", label=f"d={d}: off")
        axes[0].scatter([index + .13] * len(blocks), on_medians, color=colors[index], marker="s", label=f"d={d}: on")
        axes[1].scatter([index] * len(blocks), ratios, color=colors[index], s=36)
        axes[1].plot([index - .18, index + .18], [np.median(ratios)] * 2, color=colors[index], lw=2)
    axes[0].set(xticks=range(len(solvers)), xticklabels=[f"d={d}" for d in solvers],
                ylabel="Median host prediction time (ms)", title="Paired instrumentation blocks")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].axhline(0, color=".5", lw=.8)
    axes[1].set(xticks=range(len(solvers)), xticklabels=[f"d={d}" for d in solvers],
                ylabel="Instrumentation overhead (%)", title="Each dot is one block")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output, "inference_overhead")
    write_json(output / "microbenchmark_analysis.json", report)
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Solver $d$ & Blocks & Pairs & Off host ms & Hook overhead (\%)\\", r"\midrule"]
    for row in report["cells"]:
        lines.append(f"{row['num_steps']} & {row['blocks']} & {row['pairs']} & "
                     f"{np.median(row['off_block_median_ms']):.2f} & {row['median_block_overhead_percent']:.2f}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "microbenchmark_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def episode_key(row):
    identity = row.get("episode_identity", {})
    require(isinstance(identity, dict) and isinstance(identity.get("suite"), str) and identity["suite"],
            "Missing/invalid episode identity")
    require(all(type(identity.get(k)) is int and identity[k] >= 0
                for k in ("task_id", "initial_state_id", "policy_seed")), "Invalid state/task/seed identity")
    return (identity["suite"], identity["task_id"], identity["initial_state_id"], identity["policy_seed"])


def reconcile_rollout_manifest(root, completion, rows):
    """Do not infer the population solely from the rows that survived collection."""
    runtime_path = root / "runtime_manifest.json"
    require(runtime_path.is_file(), "Missing declared runtime/schedule manifest")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_hash = canonical_sha(runtime)
    require(completion.get("runtime_manifest_sha256") == runtime_hash,
            "Runtime manifest digest disagrees with completed summary")
    schedule = runtime.get("selected_schedule")
    require(isinstance(schedule, list) and schedule, "Missing declared selected schedule")
    schedule_hash = canonical_sha(schedule)
    require(runtime.get("selected_schedule_sha256") == schedule_hash and
            completion.get("schedule_sha256") == schedule_hash, "Declared schedule hash mismatch")
    expected = {}
    for row in schedule:
        key = (episode_key(row), (row.get("h"), row.get("denoising_steps")))
        require(key not in expected, "Duplicate identity/condition in declared schedule")
        expected[key] = row
    actual = {}
    for row in rows:
        key = (episode_key(row), (row.get("h"), row.get("denoising_steps")))
        require(key not in actual, "Duplicate completed identity/condition")
        require(key in expected and all(row.get(k) == expected[key].get(k)
                    for k in ("planned_run_id", "block_id")), "Completed row differs from selected schedule")
        require(row.get("runtime_manifest_sha256") == runtime_hash, "Episode/runtime manifest hash mismatch")
        actual[key] = row
    require(set(actual) == set(expected), "Missing declared paired cells/blocks")
    require(all(type(completion.get(k)) is int and completion[k] == len(rows)
                for k in ("completed_attempts", "expected_attempts")), "Completed/expected attempt totals disagree")
    require(completion.get("completed_blocks") == len({row["block_id"] for row in rows}),
            "Completed block total disagrees")
    require(completion.get("episodes") == rows, "Completed summary and episode journal disagree")
    require(completion.get("aggregate_successes") == sum(row.get("success") is True for row in rows),
            "Completed aggregate success count disagrees")
    return runtime


def reconcile_rollout_journal(root, rows):
    """Recompute recorded action/outcome evidence independently of episode totals.

    This checks consistency of retained records, not an independent attestation
    of simulator execution. A coherent rewrite of all records is out of scope.
    """
    events = read_jsonl(root / "steps.jsonl")
    cursor = 0
    seen = set()
    for row in rows:
        run_id = row.get("run_id")
        require(isinstance(run_id, str) and run_id and run_id not in seen, "Missing/duplicate actual attempt ID")
        seen.add(run_id)
        require(cursor < len(events), "Missing episode journal start")
        start = events[cursor]
        require(start.get("kind") == "episode_start" and start.get("run_id") == run_id
                and start.get("episode_identity") == row["episode_identity"]
                and start.get("initial_state") == row["initial_state"]
                and start.get("initial_observation_sha256") == row["initial_observation_sha256"],
                "Episode journal start differs from completed episode")
        cursor += 1
        actions, successes = [], []
        for tick in range(row["steps"]):
            require(cursor < len(events), "Missing dispatched-action journal record")
            event = events[cursor]
            require(event.get("kind") == "step" and event.get("run_id") == run_id and event.get("tick") == tick,
                    "Action journal order/identity differs")
            generated = tick % row["h"] == 0
            require(type(event.get("new_generation")) is bool and event["new_generation"] == generated
                    and event.get("generation_origin_tick") == tick - tick % row["h"]
                    and event.get("action_age_ticks") == tick % row["h"], "Action generation schedule differs")
            seed = int.from_bytes(bytes.fromhex(canonical_sha(["policy_noise", row["episode_identity"], tick]))[:4], "little")
            require(event.get("noise_seed") == (seed if generated else None), "Dispatched-action noise key differs")
            action = event.get("producer_action")
            check_initial_state(action)
            require(action["shape"] == [7] and action["sha256"] == event.get("environment_action_sha256"),
                    "Producer/consumer action hash differs")
            flags = ("driver_terminated", "driver_truncated", "driver_is_success",
                     "vector_terminated", "vector_truncated", "vector_is_success")
            require(all(type(event.get(k)) is bool for k in flags), "Invalid action-journal outcome flag")
            require(event["driver_terminated"] == event["vector_terminated"]
                    and event["driver_is_success"] == event["vector_is_success"], "Driver/vector outcomes differ")
            require(not event["driver_truncated"] or event["vector_truncated"], "Driver truncation disappeared")
            require(type(event.get("reward")) in (float, int) and np.isfinite(event["reward"]), "Invalid step reward")
            require(tick == row["steps"] - 1 or not (event["vector_terminated"] or event["vector_truncated"]),
                    "Action journal continues after episode termination")
            actions.append(action["sha256"])
            successes.append(event["driver_is_success"])
            cursor += 1
        require(any(successes) == row["success"] and canonical_sha(actions) == row.get("action_stream_sha256"),
                "Action journal and completed outcome/stream digest disagree")
        reason = ("success" if any(successes) else "terminated" if event["vector_terminated"] else
                  "truncated" if event["vector_truncated"] else "task_horizon")
        require(row.get("termination_reason") == reason, "Termination reason differs from action journal")
        terminal = {"reward": event["reward"], "terminated": event["driver_terminated"],
                    "truncated": event["driver_truncated"], "is_success": event["driver_is_success"]}
        require(row["independent_driver"].get("terminal_record") == terminal, "Independent terminal record differs")
        require(cursor < len(events), "Missing episode journal completion")
        end = events[cursor]
        require(end == {"kind": "episode_completed", "run_id": run_id, "success": row["success"],
                        "steps": row["steps"], "action_stream_sha256": row["action_stream_sha256"]},
                "Episode journal completion disagrees")
        cursor += 1
    require(cursor == len(events), "Undeclared or partial episode events remain in journal")


def recover_direct_query(query, projected_rows):
    """Answer using only the supplied projection; no reference/config lookup."""
    require(bool(projected_rows), "Cannot recover queries from an empty condition")
    available = set.intersection(*(set(row) for row in projected_rows))
    query_id = query["id"]
    if query_id == "wall_per_success_s" and "success" in available:
        if sum(row["success"] for row in projected_rows) == 0:
            return {"status": "undefined", "value": None, "used_fields": ["success"],
                    "reason": "Known zero successes; the requested ratio is undefined"}
    missing = sorted(set(query["required_fields"]) - available)
    if missing:
        return {"status": "unknown", "value": None, "missing_fields": missing,
                "reason": "Required direct records are absent from this artificial projection"}
    if query_id == "success_attempts":
        value = {"successes": sum(row["success"] for row in projected_rows), "attempts": len(projected_rows)}
    elif query_id in ("generations", "denoise_nfe"):
        value = sum(row[query_id] for row in projected_rows)
    elif query_id == "rollout_wall_s":
        value = math.fsum(row["rollout_wall_s"] for row in projected_rows)
    elif query_id == "wall_per_attempt_s":
        value = math.fsum(row["rollout_wall_s"] for row in projected_rows) / len(projected_rows)
    elif query_id == "wall_per_success_s":
        value = math.fsum(row["rollout_wall_s"] for row in projected_rows) / sum(row["success"] for row in projected_rows)
    else:
        raise ValueError(f"Unknown frozen query: {query_id}")
    return {"status": "recovered", "value": value, "used_fields": list(query["required_fields"])}


def query_answer_check(answer, reference):
    if answer["status"] == "unknown":
        return None  # Lack of supplied information is not a wrong numerical answer.
    if answer["status"] != reference["status"]:
        return False
    if type(reference["value"]) is float:
        return type(answer["value"]) in (float, int) and math.isclose(
            answer["value"], reference["value"], rel_tol=1e-12, abs_tol=1e-12)
    return answer["value"] == reference["value"]


def query_recovery(rows, runtime, output):
    """Apply the fixed suppression protocol after all rollout checks pass."""
    cells = sorted({(row["h"], row["denoising_steps"]) for row in rows})
    require(cells == [(1, 5), (1, 10), (5, 5), (5, 10)], "Query recovery requires the locked four cells")
    report = {"schema_version": 1, "status": "complete", "protocol": QUERY_RECOVERY_PROTOCOL,
              "protocol_sha256": canonical_sha(QUERY_RECOVERY_PROTOCOL),
              "protocol_status": "Fixed code-defined queries/projections; not external preregistration",
              "runtime_manifest_sha256": canonical_sha(runtime),
              "reference_episode_records_sha256": canonical_sha(rows),
              "analysis_script_sha256": file_sha(Path(__file__)),
              "outer_rollout_boundary": runtime.get("cost_boundary", "See source runtime manifest"),
              "queries_per_cell": len(QUERY_RECOVERY_PROTOCOL["queries"]), "cells": [], "checks": [],
              "projections": []}
    for index, (h, d) in enumerate(cells):
        selected = [row for row in rows if (row["h"], row["denoising_steps"]) == (h, d)]
        # This flat view is built only after the caller has reconciled actual
        # action journals, measured counters, outcomes and declared allocation.
        records = [{"attempt_id": row["run_id"], "success": row["success"],
                    "generations": row["counts"]["generations"],
                    "denoise_nfe": row["counts"]["denoise_steps"],
                    "rollout_wall_s": row["timing"]["rollout_wall_s"]} for row in selected]
        wins = sum(row["independent_driver"]["success"] for row in selected)
        wall = math.fsum(row["timing"]["rollout_wall_s"] for row in selected)
        # Reference arithmetic is separate from the projection recovery routine.
        values = {"success_attempts": {"successes": wins, "attempts": len(selected)},
                  "generations": sum((row["steps"] + h - 1) // h for row in selected),
                  "denoise_nfe": sum(row["counts"]["denoise_steps"] for row in selected),
                  "rollout_wall_s": wall, "wall_per_attempt_s": wall / len(selected),
                  "wall_per_success_s": wall / wins if wins else None}
        references = {key: {"status": "undefined" if key == "wall_per_success_s" and wins == 0 else "recovered",
                            "value": value} for key, value in values.items()}
        report["cells"].append({"cell_id": index, "h": h, "d": d, "attempts": len(selected), "references": references})
        for projection in QUERY_RECOVERY_PROTOCOL["projections"]:
            projected = [{key: row[key] for key in projection["fields"]} for row in records]
            report["projections"].append({"cell_id": index, "projection": projection["id"],
                                          "records_sha256": canonical_sha(projected), "records": projected})
            for query in QUERY_RECOVERY_PROTOCOL["queries"]:
                answer = recover_direct_query(query, projected)
                checked = query_answer_check(answer, references[query["id"]])
                require(checked is not False, "Recovered query answer disagrees with journal-reconciled reference")
                report["checks"].append({"cell_id": index, "query_id": query["id"],
                                          "projection": projection["id"], "requested_unit": query["unit"],
                                          "requested_boundary": query["boundary"], **answer,
                                          "reference": references[query["id"]], "correct_answer": checked})
    report["unique_queries"] = len(cells) * len(QUERY_RECOVERY_PROTOCOL["queries"])
    report["projection_checks"] = len(report["checks"])
    report["summary"] = []
    for projection in QUERY_RECOVERY_PROTOCOL["projections"]:
        checks = [row for row in report["checks"] if row["projection"] == projection["id"]]
        counts = {status: sum(row["status"] == status for row in checks)
                  for status in ("recovered", "unknown", "undefined")}
        report["summary"].append({"projection": projection["id"], "queries": len(checks), **counts,
                                  "answered_correctly": sum(row["correct_answer"] is True for row in checks),
                                  "wrong_answers": sum(row["correct_answer"] is False for row in checks)})
    write_json(output / "query_recovery.json", report)
    write_json(output / "query_recovery_protocol.json", QUERY_RECOVERY_PROTOCOL)
    labels = {"success_attempts": "Successes / attempts", "generations": "Generations (count)",
              "denoise_nfe": "Denoise NFE (count)", "rollout_wall_s": "Total wall (s)",
              "wall_per_attempt_s": "Wall / attempt (s)", "wall_per_success_s": "All-attempt wall / success (s)"}
    lines = [r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Query & $(1,5)$ & $(1,10)$ & $(5,5)$ & $(5,10)$ & O & G & L\\", r"\midrule"]
    for query in QUERY_RECOVERY_PROTOCOL["queries"]:
        answers = []
        for cell in report["cells"]:
            answer = cell["references"][query["id"]]
            value = answer["value"]
            answers.append("undef." if answer["status"] == "undefined" else
                           f"{value['successes']}/{value['attempts']}" if isinstance(value, dict) else
                           f"{value:.2f}" if type(value) is float else str(value))
        answered = [sum(row["correct_answer"] is True for row in report["checks"]
                        if row["query_id"] == query["id"] and row["projection"] == projection["id"])
                    for projection in QUERY_RECOVERY_PROTOCOL["projections"]]
        lines.append(" & ".join([labels[query["id"]], *answers, *(f"{count}/4" for count in answered)]) + r"\\")
    lines.extend([r"\midrule", " & ".join(["All queries", r"\multicolumn{4}{c}{Correctly answered / requested}",
                  *(f"{row['answered_correctly']}/{row['queries']}" for row in report["summary"])]) + r"\\",
                  r"\bottomrule", r"\end{tabular}\endgroup", r"\par\smallskip",
                  r"{\footnotesize Cell headers are $(h,d)$; values are from the completed, reconciled records. O: outcomes; G: outcomes plus generation counts; L: full records. O/G/L show correct answers per four cells; known undefined ratios count as answered. Unknown fields are never set to zero. This is deterministic direct-record recovery under artificial suppression, not a reader study or evidence of better decisions. Reduced views omit solver configuration/source; the test does not rule out deriving NFE when those are available.}"])
    (output / "query_recovery_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def rollout(root, output):
    completion = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if completion.get("status") != "completed":
        raise ValueError("Rollout set must be complete; partial outcomes are not silently selected")
    rows = read_jsonl(root / "episodes.jsonl")
    if not rows or any(row.get("status") != "completed" for row in rows):
        raise ValueError("Incomplete attempt present; resolve attempt policy explicitly")
    runtime = reconcile_rollout_manifest(root, completion, rows)
    cells = sorted({(row["h"], row["denoising_steps"]) for row in rows})
    if cells != [(1, 5), (1, 10), (5, 5), (5, 10)]:
        raise ValueError("Expected the locked four-cell design")
    keyed = {}
    for row in rows:
        require(type(row.get("h")) is int and type(row.get("denoising_steps")) is int,
                "Condition settings must be integers")
        key = (episode_key(row), (row["h"], row["denoising_steps"]))
        if key in keyed or type(row["success"]) is not bool:
            raise ValueError("Duplicate identity or invalid binary outcome")
        if not positive_number(row.get("timing", {}).get("rollout_wall_s")):
            raise ValueError("Invalid measured rollout time")
        check_initial_state(row.get("initial_state"))
        require(valid_hash(row.get("initial_observation_sha256")), "Missing paired initial-observation hash")
        require(type(row.get("steps")) is int and row["steps"] > 0, "Invalid executed-step count")
        require(all(type(row.get("counts", {}).get(k)) is int and row["counts"][k] > 0
                    for k in ("generations", "denoise_steps")), "Invalid measured work counts")
        require(row["counts"]["generations"] == (row["steps"] + row["h"] - 1) // row["h"] and
                row["counts"]["denoise_steps"] == row["counts"]["generations"] * row["denoising_steps"],
                "Generation or solver-work counts disagree with schedule")
        require(row["counts"].get("prefill_calls") == row["counts"]["generations"]
                and row["counts"].get("action_expert_calls") == row["counts"]["denoise_steps"]
                and type(row["counts"].get("vision_encoder_calls")) is int
                and row["counts"]["vision_encoder_calls"] >= row["counts"]["generations"],
                "Measured rollout encoder/prefill/expert counts disagree")
        for clock in ("generation_host_s", "generation_cuda_ms"):
            values = row["timing"].get(clock)
            require(isinstance(values, list) and len(values) == row["counts"]["generations"]
                    and all(positive_number(v, clock.endswith("ms")) for v in values),
                    "Generation timing journal is incomplete/invalid")
        driver = row.get("independent_driver", {})
        require(driver.get("steps") == row["steps"] and type(driver.get("success")) is bool and
                driver["success"] == row["success"], "Independent driver outcomes/counts disagree")
        require(type(row.get("memory", {}).get("peak_allocated_bytes")) is int and
                row["memory"]["peak_allocated_bytes"] >= 0, "Invalid peak memory")
        keyed[key] = row
    reconcile_rollout_journal(root, rows)
    identities = sorted({key for key, cell in keyed})
    for identity in identities:
        if any((identity, cell) not in keyed for cell in cells):
            raise ValueError("Incomplete paired block")
        hashes = {keyed[(identity, cell)]["initial_state"]["sha256"] for cell in cells}
        if len(hashes) != 1:
            raise ValueError("Paired initial simulator states differ")
        if len({keyed[(identity, cell)]["initial_observation_sha256"] for cell in cells}) != 1:
            raise ValueError("Paired initial observations differ")
    tasks = sorted({identity[:2] for identity in identities})
    states = {task: sorted({identity[2] for identity in identities if identity[:2] == task}) for task in tasks}
    seed_sets = {(*task, state): {identity[3] for identity in identities if identity[:3] == (*task, state)}
                 for task in tasks for state in states[task]}
    require(len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) == 1,
            "Unbalanced policy-seed groups across declared task/states")
    # Per-state means retain all seed repeats, with equal weight for each task.
    state_values = {}
    for task, state_ids in states.items():
        for state in state_ids:
            ids = [identity for identity in identities if identity[:3] == (*task, state)]
            state_values[(*task, state)] = np.array([
                [np.mean([keyed[(identity, cell)]["success"] for identity in ids]),
                 np.mean([keyed[(identity, cell)]["timing"]["rollout_wall_s"] for identity in ids])]
                for cell in cells])
    estimate = np.mean([np.mean([state_values[(*task, state)] for state in states[task]], axis=0)
                        for task in tasks], axis=0)
    draws = None
    if all(len(values) >= 2 for values in states.values()):
        rng = np.random.default_rng(731)
        draws = np.mean([np.array([state_values[(*task, state)] for state in states[task]])[
            rng.integers(0, len(states[task]), size=(10000, len(states[task])))].mean(axis=1)
            for task in tasks], axis=0)
    report = {"scope": "fixed declared task set; descriptive paired state bootstrap, not equivalence",
              "tasks": len(tasks), "episodes_per_cell": len(identities), "cells": [],
              "task_state_seed_counts": [{"suite": task[0], "task_id": task[1], "states": len(states[task]),
                                           "policy_seeds_per_state": len(seed_sets[(*task, states[task][0])])}
                                          for task in tasks],
              "mean_wall_weighting": "Equal tasks, then equal states, then equal policy-seed repeats within state",
              "wall_s_per_success_weighting": "Raw total rollout wall time divided by raw successes over all attempts",
              "runtime_manifest_sha256": canonical_sha(runtime),
              "analysis_script_sha256": file_sha(Path(__file__)),
              "journal_validation_scope": "Recomputed retained producer-action hashes, consumer hashes, generation/noise schedule, initial-state records, outcomes, and episode totals. This is record consistency, not external attestation of simulator execution.",
              "bootstrap_draws": 10000 if draws is not None else 0,
              "interval_limitation": "Conditional on these tasks; repeated seeds stay within state. Few states limit precision."}
    report["per_task"] = [
        {"suite": task[0], "task_id": task[1], "cells": [
            {"h": cell[0], "d": cell[1],
             "successes": sum(keyed[(identity, cell)]["success"]
                              for identity in identities if identity[:2] == task),
             "attempts": sum(identity[:2] == task for identity in identities)}
            for cell in cells]}
        for task in tasks]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), layout="constrained")
    for index, cell in enumerate(cells):
        group = [keyed[(identity, cell)] for identity in identities]
        wins = sum(row["success"] for row in group)
        wall = sum(row["timing"]["rollout_wall_s"] for row in group)
        gens = sum(row["counts"]["generations"] for row in group)
        nfe = sum(row["counts"]["denoise_steps"] for row in group)
        low, high = (np.quantile(draws[:, index], [.025, .975], axis=0)
                     if draws is not None else (estimate[index], estimate[index]))
        record = {"h": cell[0], "d": cell[1], "successes": wins, "attempts": len(group),
                  "task_weighted_success": float(estimate[index, 0]), "mean_wall_s": float(estimate[index, 1]),
                  "pooled_wall_s_per_attempt": wall / len(group),
                  "total_wall_s": wall, "wall_s_per_success": wall / wins if wins else None,
                  "task_weighted_wall_s_per_success": float(estimate[index, 1] / estimate[index, 0])
                                                        if estimate[index, 0] > 0 else None,
                  "generations": gens, "denoise_nfe": nfe,
                  "steps": sum(row["steps"] for row in group),
                  "peak_allocated_bytes": max(row["memory"]["peak_allocated_bytes"] for row in group),
                  "success_ci": [float(low[0]), float(high[0])] if draws is not None else None,
                  "wall_ci": [float(low[1]), float(high[1])] if draws is not None else None}
        point_delta = estimate[index] - estimate[cells.index((1, 10))]
        record["paired_difference_vs_native"] = {"success": float(point_delta[0]), "wall_s": float(point_delta[1]),
                                                  "success_ci": None, "wall_ci": None}
        if draws is not None:
            delta = draws[:, index] - draws[:, cells.index((1, 10))]
            bounds = np.quantile(delta, [.025, .975], axis=0)
            record["paired_difference_vs_native"].update(
                success_ci=bounds[:, 0].tolist(), wall_ci=bounds[:, 1].tolist())
        report["cells"].append(record)
        axes[0].errorbar(estimate[index, 1], estimate[index, 0] * 100,
                         xerr=[[estimate[index, 1]-low[1]], [high[1]-estimate[index, 1]]],
                         yerr=[[(estimate[index, 0]-low[0])*100], [(high[0]-estimate[index, 0])*100]],
                         fmt="o" if cell[0] == 1 else "s", capsize=3, label=f"h={cell[0]}, d={cell[1]}")
        axes[1].bar(index - .16, gens / len(group), width=.3, color="#3474a6")
        axes[1].bar(index + .16, nfe / len(group), width=.3, color="#d78239")
    contrast_specs = [
        ("refinement_at_h1", (1, 10), (1, 5), "Q(1,10)-Q(1,5)"),
        ("refinement_at_h5", (5, 10), (5, 5), "Q(5,10)-Q(5,5)"),
        ("freshness_at_d5", (1, 5), (5, 5), "Q(1,5)-Q(5,5)"),
        ("freshness_at_d10", (1, 10), (5, 10), "Q(1,10)-Q(5,10)"),
        ("principal_joint_vs_native", (5, 5), (1, 10), "Q(5,5)-Q(1,10)"),
    ]
    report["factorial_success_contrasts"] = []
    for name, left, right, estimand in contrast_specs:
        left_index, right_index = cells.index(left), cells.index(right)
        point = float(estimate[left_index, 0] - estimate[right_index, 0])
        interval = None
        if draws is not None:
            delta = draws[:, left_index, 0] - draws[:, right_index, 0]
            interval = np.quantile(delta, [.025, .975]).tolist()
        report["factorial_success_contrasts"].append(
            {"name": name, "estimand": estimand, "difference": point,
             "bootstrap_95_interval": interval})
    i_110, i_15 = cells.index((1, 10)), cells.index((1, 5))
    i_510, i_55 = cells.index((5, 10)), cells.index((5, 5))
    interaction = ((estimate[i_510, 0] - estimate[i_110, 0]) -
                   (estimate[i_55, 0] - estimate[i_15, 0]))
    interaction_interval = None
    if draws is not None:
        interaction_draws = ((draws[:, i_510, 0] - draws[:, i_110, 0]) -
                             (draws[:, i_55, 0] - draws[:, i_15, 0]))
        interaction_interval = np.quantile(interaction_draws, [.025, .975]).tolist()
    report["factorial_interaction"] = {
        "estimand": "[Q(5,10)-Q(1,10)]-[Q(5,5)-Q(1,5)]",
        "difference": float(interaction),
        "bootstrap_95_interval": interaction_interval,
        "interpretation": "Difference between longer-prefix effects at d=10 and d=5."}
    axes[0].set(xlabel="Task-weighted mean rollout wall time (s)", ylabel="Task-weighted success (%)",
                ylim=(-5, 105), title="Observed cost and quality")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set(xticks=range(len(cells)), xticklabels=[f"h={h}\nd={d}" for h, d in cells],
                ylabel="Mean count per attempted episode", title="Generations and actual solver work")
    axes[1].legend(["Chunk generations", "Denoise evaluations"], fontsize=7, frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output, "paired_rollout_tradeoffs")
    write_json(output / "rollout_analysis.json", report)
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$h$ & $d$ & Success & Task-wtd. wall/attempt (s) & Pooled wall/success (s) & Chunks & NFE\\", r"\midrule"]
    for row in report["cells"]:
        cps = f"{row['wall_s_per_success']:.2f}" if row['wall_s_per_success'] is not None else "--"
        lines.append(f"{row['h']} & {row['d']} & {row['successes']}/{row['attempts']} & "
                     f"{row['mean_wall_s']:.2f} & {cps} & {row['generations']} & {row['denoise_nfe']}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "rollout_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    query_recovery(rows, runtime, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbenchmark", type=Path)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.microbenchmark and not args.rollouts:
        parser.error("Provide completed experiment input")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, directory, analyze in (("microbenchmark", args.microbenchmark, microbenchmark),
                                      ("rollouts", args.rollouts, rollout)):
        if directory:
            analyze(directory, args.output)
            sources[name] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in directory.glob("*.json*")}
    write_json(args.output / "analysis_inputs.json", sources)


if __name__ == "__main__":
    main()
