"""Analysis integrity tests. All constructed records/outputs live in temp dirs.

These fixtures test estimands and validation; they are not empirical run output.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import analyze_execution as analysis

CELLS = [(1, 5), (1, 10), (5, 5), (5, 10)]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def array_record(values):
    a = np.asarray(values, dtype="<f8")
    header = json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode()
    return {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(),
            "sha256": hashlib.sha256(header + a.tobytes()).hexdigest()}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def rollout_fixture(tasks=None, seeds=(100, 101), success=None, wall=None):
    tasks = tasks or {0: [0, 1]}
    success = success or (lambda task, state, seed, cell: True)
    wall = wall or (lambda task, state, seed, cell: 10.)
    rows, schedule = [], []
    for task, states in tasks.items():
        for state in states:
            for seed in seeds:
                identity = {"suite": "libero_spatial", "task_id": task,
                            "initial_state_id": state, "policy_seed": seed}
                block = f"task{task}:state{state}:seed{seed}"
                for h, d in CELLS:
                    selected = {"episode_identity": identity, "h": h, "denoising_steps": d,
                                "block_id": block, "planned_run_id": f"{block}:h{h}:d{d}"}
                    schedule.append(copy.deepcopy(selected))
                    won = bool(success(task, state, seed, (h, d)))
                    rows.append({**copy.deepcopy(selected), "status": "completed", "success": won,
                                 "run_id": f"{block}:h{h}:d{d}:attempt_fixture",
                                 "initial_state": array_record([task, state]),
                                 "initial_observation_sha256": sha(f"observation:{task}:{state}"),
                                 "steps": 10, "counts": {"generations": 10 // h, "denoise_steps": 10 // h * d,
                                                         "prefill_calls": 10 // h, "action_expert_calls": 10 // h * d,
                                                         "vision_encoder_calls": 20 // h},
                                 "timing": {"rollout_wall_s": wall(task, state, seed, (h, d)),
                                            "generation_host_s": [.01] * (10 // h),
                                            "generation_cuda_ms": [5.] * (10 // h)},
                                 "memory": {"peak_allocated_bytes": 1000},
                                 "independent_driver": {"steps": 10, "success": won}})
    return rows, schedule


def write_rollout(root, rows, schedule):
    runtime = {"selected_schedule": schedule, "selected_schedule_sha256": analysis.canonical_sha(schedule)}
    runtime_digest = analysis.canonical_sha(runtime)
    for row in rows:
        row["runtime_manifest_sha256"] = runtime_digest
    events = []
    for row in rows:
        events.append({"kind": "episode_start", "run_id": row["run_id"],
                       "episode_identity": row["episode_identity"], "initial_state": row["initial_state"],
                       "initial_observation_sha256": row["initial_observation_sha256"]})
        actions = []
        for tick in range(row["steps"]):
            action = array_record([float(tick)] * 7)
            actions.append(action["sha256"])
            generates = tick % row["h"] == 0
            noise = int.from_bytes(bytes.fromhex(analysis.canonical_sha(["policy_noise", row["episode_identity"], tick]))[:4], "little")
            success = row["success"] and tick == row["steps"] - 1
            events.append({"kind": "step", "run_id": row["run_id"], "tick": tick,
                           "new_generation": generates, "generation_origin_tick": tick - tick % row["h"],
                           "action_age_ticks": tick % row["h"], "noise_seed": noise if generates else None,
                           "producer_action": action, "environment_action_sha256": action["sha256"],
                           "reward": float(success), "driver_terminated": success, "driver_truncated": False,
                           "driver_is_success": success, "vector_terminated": success, "vector_truncated": False,
                           "vector_is_success": success})
        row["termination_reason"] = "success" if row["success"] else "task_horizon"
        row["action_stream_sha256"] = analysis.canonical_sha(actions)
        row["independent_driver"]["terminal_record"] = {
            "reward": float(row["success"]), "terminated": row["success"], "truncated": False,
            "is_success": row["success"]}
        events.append({"kind": "episode_completed", "run_id": row["run_id"], "success": row["success"],
                       "steps": row["steps"], "action_stream_sha256": row["action_stream_sha256"]})
    summary = {"status": "completed", "runtime_manifest_sha256": runtime_digest,
               "schedule_sha256": analysis.canonical_sha(schedule),
               "expected_attempts": len(schedule), "completed_attempts": len(rows),
               "completed_blocks": len({x["block_id"] for x in rows}), "episodes": rows,
               "aggregate_successes": sum(x["success"] for x in rows)}
    analysis.write_json(root / "runtime_manifest.json", runtime)
    analysis.write_json(root / "summary.json", summary)
    write_jsonl(root / "episodes.jsonl", rows)
    write_jsonl(root / "steps.jsonl", events)


def micro_fixture():
    config = {"blocks": 2, "num_steps": [5, 10], "calls": 2, "warmup_calls": 1,
              "rtol": 1e-5, "atol": 1e-6, "seed": 1000}
    pairs, trials = [], []
    for block in range(2):
        for d in (5, 10):
            for phase, count in (("warmup", 1), ("measurement", 2)):
                for index in range(count):
                    pid = f"b{block}:d{d}:{phase}:{index}"
                    common = {"pair_id": pid, "block": block, "num_steps": d,
                              "phase": phase, "call_index": index}
                    off = float(d * 10 + block + index)
                    on = off * 1.05
                    action = sha(f"action:{pid}")
                    pair = {**common, "allclose": True, "exact_equal": True, "max_abs_diff": 0.,
                            "rtol": config["rtol"], "atol": config["atol"],
                            "off_action_sha256": action, "on_action_sha256": action,
                            "off_host_elapsed_ms": off, "on_host_elapsed_ms": on,
                            "off_cuda_elapsed_ms": off / 2, "on_cuda_elapsed_ms": on / 2}
                    pairs.append(pair)
                    for mode, time in (("off", off), ("on", on)):
                        noise_key = f"robopad-fixed-observation-v1:1000:{block}:{phase}:{index}"
                        noise_seed = int.from_bytes(hashlib.sha256(noise_key.encode()).digest()[:8], "big") % (2**63 - 1)
                        trials.append({**common, "instrumentation": mode, "action_sha256": action,
                                       "noise_sha256": sha(noise_key), "noise_seed": noise_seed,
                                       "observation_sha256": sha("observation"),
                                       "action_shape": [1, 50, 7], "action_dtype": "torch.float32", "action_index": len(trials),
                                       "instrumentation_counts": {"chunk_entries": 1, "denoise_nfe": d,
                                           "vision_encoder_calls": 2, "prefill_forwards": 1, "action_expert_forwards": d}
                                           if mode == "on" else None,
                                       "stage_cuda_ms": {"embed_prefix": 10., "vision_encoder": 8., "prefill": 2., "action_expert": 3.}
                                           if mode == "on" else None,
                                       "host_elapsed_ms": time, "cuda_elapsed_ms": time / 2})
    summary = {"status": "complete", "config": config, "planned_actual_chunk_generations": 24,
               "observation": {"observation_sha256": sha("observation")},
               "counts": {"completed_actual_chunk_generations": 24, "issued_prediction_calls": 24, "completed_pairs": 12,
                          "unpaired_calls": 0, "measured_calls": 16, "warmup_calls": 8,
                          "pairs_allclose": 12, "pairs_exact_equal": 12}}
    return summary, pairs, trials


def write_micro(root, summary, pairs, trials):
    # Artifact bytes deliberately are not a tensor: offline analysis verifies the
    # file digest only; tensor decoding/parity belongs to the GPU runner.
    (root / "actions.pt").write_bytes(b"temporary analysis digest fixture")
    summary["actions_artifact"] = {"path": "actions.pt", "sha256": analysis.file_sha(root / "actions.pt"),
                                   "count": len(trials), "index_field": "action_index"}
    # allow_nan only here to test rejection of nonfinite corrupted input.
    (root / "summary.json").write_text(json.dumps(summary))
    write_jsonl(root / "pairs.jsonl", pairs)
    write_jsonl(root / "trials.jsonl", trials)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        # Tests exercise numerical analysis, not expensive vector rendering.
        self.figures = patch.object(analysis, "save_figure", side_effect=lambda fig, *_: analysis.plt.close(fig))
        self.figures.start()
        self.addCleanup(self.figures.stop)

    def test_missing_one_cell_or_entire_block_is_rejected(self):
        for remove in (1, 4):
            with self.subTest(remove=remove):
                rows, schedule = rollout_fixture()
                write_rollout(self.root, rows, schedule)
                write_jsonl(self.root / "episodes.jsonl", rows[:-remove])
                with self.assertRaisesRegex(ValueError, "Missing declared paired"):
                    analysis.rollout(self.root, self.output)

    def test_missing_forged_or_mismatched_initial_hash_rejected(self):
        for kind in ("missing", "empty", "forged", "mismatched", "observation"):
            with self.subTest(kind=kind):
                rows, schedule = rollout_fixture()
                if kind == "missing":
                    for row in rows:
                        row["initial_state"].pop("sha256")
                elif kind == "empty":
                    for row in rows:
                        row["initial_state"]["sha256"] = None
                elif kind == "forged":
                    rows[0]["initial_state"]["values"][0] = 999.
                elif kind == "mismatched":
                    rows[0]["initial_state"] = array_record([999., 0.])
                else:
                    rows[0]["initial_observation_sha256"] = sha("different")
                write_rollout(self.root, rows, schedule)
                with self.assertRaises(ValueError):
                    analysis.rollout(self.root, self.output)

    def test_equal_task_weight_is_not_raw_pooling(self):
        rows, schedule = rollout_fixture(tasks={0: [0], 1: [0, 1, 2]}, seeds=(100,),
            success=lambda task, *_: task == 0, wall=lambda task, *_: 2. if task == 0 else 10.)
        write_rollout(self.root, rows, schedule)
        report = analysis.rollout(self.root, self.output)
        for cell in report["cells"]:
            self.assertEqual(cell["successes"] / cell["attempts"], .25)
            self.assertEqual(cell["task_weighted_success"], .5)
            self.assertEqual(cell["mean_wall_s"], 6.)
            self.assertEqual(cell["pooled_wall_s_per_attempt"], 8.)
            self.assertEqual(cell["task_weighted_wall_s_per_success"], 12.)
            self.assertEqual(cell["wall_s_per_success"], 32.)

    def test_zero_success_cost_is_null(self):
        rows, schedule = rollout_fixture(success=lambda *_: False)
        write_rollout(self.root, rows, schedule)
        report = analysis.rollout(self.root, self.output)
        self.assertTrue(all(c["wall_s_per_success"] is None and c["task_weighted_wall_s_per_success"] is None
                            for c in report["cells"]))

    def test_policy_seeds_remain_grouped_within_state(self):
        rows, schedule = rollout_fixture(success=lambda task, state, seed, cell: seed == 100)
        write_rollout(self.root, rows, schedule)
        report = analysis.rollout(self.root, self.output)
        self.assertEqual(report["bootstrap_draws"], 10000)
        for cell in report["cells"]:
            self.assertEqual(cell["success_ci"], [.5, .5])
            self.assertEqual(cell["paired_difference_vs_native"]["success_ci"], [0., 0.])
        contrasts = report["factorial_success_contrasts"]
        self.assertEqual([row["name"] for row in contrasts], [
            "refinement_at_h1", "refinement_at_h5", "freshness_at_d5",
            "freshness_at_d10", "principal_joint_vs_native"])
        self.assertTrue(all(row["difference"] == 0. and
                            row["bootstrap_95_interval"] == [0., 0.]
                            for row in contrasts))
        self.assertEqual(report["factorial_interaction"]["difference"], 0.)
        self.assertEqual(report["factorial_interaction"]["bootstrap_95_interval"], [0., 0.])
        self.assertEqual(len(report["per_task"]), 1)
        self.assertTrue(all(cell["successes"] == 2 and cell["attempts"] == 4
                            for cell in report["per_task"][0]["cells"]))
        self.assertEqual(report["task_state_seed_counts"][0]["policy_seeds_per_state"], 2)

    def test_unbalanced_seed_group_rejected_even_when_declared_complete(self):
        rows, schedule = rollout_fixture()
        rows, schedule = rows[:-4], schedule[:-4]
        write_rollout(self.root, rows, schedule)
        with self.assertRaisesRegex(ValueError, "Unbalanced policy-seed"):
            analysis.rollout(self.root, self.output)

    def test_independent_outcome_and_count_disagreements_rejected(self):
        for fault in ("driver", "nfe", "summary"):
            with self.subTest(fault=fault):
                rows, schedule = rollout_fixture()
                if fault == "driver":
                    rows[0]["independent_driver"]["success"] = False
                elif fault == "nfe":
                    rows[0]["counts"]["denoise_steps"] += 1
                write_rollout(self.root, rows, schedule)
                if fault == "summary":
                    data = json.loads((self.root / "summary.json").read_text())
                    data["expected_attempts"] += 4
                    analysis.write_json(self.root / "summary.json", data)
                with self.assertRaises(ValueError):
                    analysis.rollout(self.root, self.output)

    def test_action_journal_corruptions_rejected(self):
        for fault in ("missing", "order", "action", "consumer", "noise", "generation", "outcome", "partial_tail"):
            with self.subTest(fault=fault):
                rows, schedule = rollout_fixture()
                write_rollout(self.root, rows, schedule)
                events = analysis.read_jsonl(self.root / "steps.jsonl")
                if fault == "missing":
                    events.pop(1)
                elif fault == "order":
                    events[1], events[2] = events[2], events[1]
                elif fault == "action":
                    events[1]["producer_action"]["values"][0] += 1
                elif fault == "consumer":
                    events[1]["environment_action_sha256"] = sha("not the dispatched action")
                elif fault == "noise":
                    events[1]["noise_seed"] += 1
                elif fault == "generation":
                    events[1]["generation_origin_tick"] += 1
                elif fault == "outcome":
                    events[1]["driver_is_success"] = True
                else:
                    events.append(copy.deepcopy(events[0]))
                write_jsonl(self.root / "steps.jsonl", events)
                with self.assertRaises(ValueError):
                    analysis.rollout(self.root, self.output)

    def test_generation_clock_and_hook_counts_reconciled(self):
        for fault in ("clock", "prefill", "expert", "vision"):
            with self.subTest(fault=fault):
                rows, schedule = rollout_fixture()
                if fault == "clock":
                    rows[0]["timing"]["generation_host_s"].pop()
                else:
                    rows[0]["counts"][{"prefill": "prefill_calls", "expert": "action_expert_calls",
                                        "vision": "vision_encoder_calls"}[fault]] = 0
                write_rollout(self.root, rows, schedule)
                with self.assertRaises(ValueError):
                    analysis.rollout(self.root, self.output)

    def test_query_recovery_uses_actual_all_attempt_values_and_suppressed_fields(self):
        rows, schedule = rollout_fixture(success=lambda task, state, seed, cell: state == 0,
                                          wall=lambda task, state, *_: 2. if state == 0 else 10.)
        write_rollout(self.root, rows, schedule)
        analysis.rollout(self.root, self.output)
        report = json.loads((self.output / "query_recovery.json").read_text())
        self.assertEqual((report["unique_queries"], report["projection_checks"]), (24, 72))
        self.assertEqual([row["answered_correctly"] for row in report["summary"]], [4, 8, 24])
        self.assertEqual([row["unknown"] for row in report["summary"]], [20, 16, 0])
        self.assertTrue(all(row["wrong_answers"] == 0 for row in report["summary"]))
        for cell in report["cells"]:
            values = {key: answer["value"] for key, answer in cell["references"].items()}
            self.assertEqual(values["success_attempts"], {"successes": 2, "attempts": 4})
            self.assertEqual(values["rollout_wall_s"], 24.)
            self.assertEqual(values["wall_per_attempt_s"], 6.)
            self.assertEqual(values["wall_per_success_s"], 12.)  # Includes failed attempts.
            self.assertEqual(values["generations"], 40 // cell["h"])
            self.assertEqual(values["denoise_nfe"], 40 // cell["h"] * cell["d"])
        for projection in report["projections"]:
            for record in projection["records"]:
                self.assertTrue({"h", "d", "denoising_steps", "source"}.isdisjoint(record))
                if projection["projection"] != "full_ledger":
                    self.assertNotIn("rollout_wall_s", record)
                    self.assertNotIn("denoise_nfe", record)
        table = (self.output / "query_recovery_table.tex").read_text()
        self.assertIn("24.00", table)
        self.assertIn("12.00", table)
        self.assertIn("not a reader study", table)

    def test_query_known_zero_denominator_is_answered_undefined(self):
        rows, schedule = rollout_fixture(success=lambda *_: False)
        write_rollout(self.root, rows, schedule)
        analysis.rollout(self.root, self.output)
        report = json.loads((self.output / "query_recovery.json").read_text())
        self.assertEqual([row["answered_correctly"] for row in report["summary"]], [8, 12, 24])
        self.assertEqual([row["undefined"] for row in report["summary"]], [4, 4, 4])
        ratios = [row for row in report["checks"] if row["query_id"] == "wall_per_success_s"]
        self.assertEqual(len(ratios), 12)
        self.assertTrue(all(row["status"] == "undefined" and row["value"] is None
                            and row["correct_answer"] is True for row in ratios))
        unknown = [row for row in report["checks"] if row["status"] == "unknown"]
        self.assertTrue(all(row["value"] is None and row["correct_answer"] is None for row in unknown))

    def test_query_contract_and_answer_check_are_explicit(self):
        protocol = analysis.QUERY_RECOVERY_PROTOCOL
        self.assertEqual([q["id"] for q in protocol["queries"]], [
            "success_attempts", "generations", "denoise_nfe", "rollout_wall_s",
            "wall_per_attempt_s", "wall_per_success_s"])
        self.assertEqual([p["id"] for p in protocol["projections"]], [
            "outcomes_only", "outcomes_generations", "full_ledger"])
        query = next(q for q in protocol["queries"] if q["id"] == "denoise_nfe")
        # This deliberately bounded routine performs direct field recovery. It
        # must not quietly infer NFE from an undeclared source/config mechanism.
        answer = analysis.recover_direct_query(query, [{"attempt_id": "fixture", "generations": 2, "d": 10}])
        self.assertEqual(answer["status"], "unknown")
        self.assertEqual(answer["missing_fields"], ["denoise_nfe"])
        reference = {"status": "recovered", "value": 20}
        self.assertIsNone(analysis.query_answer_check(answer, reference))
        self.assertFalse(analysis.query_answer_check({"status": "recovered", "value": 19}, reference))
        self.assertTrue(analysis.query_answer_check({"status": "recovered", "value": 20}, reference))

    def test_real_microbenchmark_schema_and_warmup_exclusion(self):
        write_micro(self.root, *micro_fixture())
        report = analysis.microbenchmark(self.root, self.output)
        self.assertEqual(report["paired_measurements"], 8)
        for cell in report["cells"]:
            self.assertEqual((cell["blocks"], cell["pairs"]), (2, 4))
            self.assertAlmostEqual(cell["median_block_overhead_percent"], 5.)

    def test_microbenchmark_missing_pair_block_or_trial_rejected(self):
        for fault in ("pair", "block", "trial", "duplicate"):
            with self.subTest(fault=fault):
                summary, pairs, trials = micro_fixture()
                if fault == "pair":
                    pairs.pop()
                elif fault == "block":
                    pairs = [x for x in pairs if x["block"] != 1]
                elif fault == "trial":
                    trials.pop()
                else:
                    pairs.append(copy.deepcopy(pairs[0]))
                write_micro(self.root, summary, pairs, trials)
                with self.assertRaises(ValueError):
                    analysis.microbenchmark(self.root, self.output)

    def test_instrumentation_parity_and_noise_are_required(self):
        for fault in ("parity", "truthy_string", "hash", "noise", "nan", "zero"):
            with self.subTest(fault=fault):
                summary, pairs, trials = micro_fixture()
                if fault == "parity":
                    pairs[0]["allclose"] = False
                elif fault == "truthy_string":
                    pairs[0]["allclose"] = "false"
                elif fault == "hash":
                    pairs[0]["on_action_sha256"] = sha("different")
                elif fault == "noise":
                    trials[0]["noise_sha256"] = sha("different")
                elif fault == "nan":
                    pairs[0]["on_host_elapsed_ms"] = float("nan")
                else:
                    pairs[0]["off_host_elapsed_ms"] = 0.
                write_micro(self.root, summary, pairs, trials)
                with self.assertRaises(ValueError):
                    analysis.microbenchmark(self.root, self.output)

    def test_microbenchmark_measured_hooks_noise_keys_and_artifact_checked(self):
        for fault in ("work", "stages", "off_hooks", "noise_key", "cross_solver_noise", "index", "tensor_bytes"):
            with self.subTest(fault=fault):
                summary, pairs, trials = micro_fixture()
                if fault == "work":
                    trials[1]["instrumentation_counts"]["denoise_nfe"] += 1
                elif fault == "stages":
                    trials[1]["stage_cuda_ms"]["vision_encoder"] = -1
                elif fault == "off_hooks":
                    trials[0]["instrumentation_counts"] = trials[1]["instrumentation_counts"]
                elif fault == "noise_key":
                    trials[0]["noise_seed"] += 1
                elif fault == "cross_solver_noise":
                    trials[0]["noise_sha256"] = trials[1]["noise_sha256"] = sha("different noise but still on/off paired")
                elif fault == "index":
                    trials[0]["action_index"] = 1
                write_micro(self.root, summary, pairs, trials)
                if fault == "tensor_bytes":
                    (self.root / "actions.pt").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    analysis.microbenchmark(self.root, self.output)


if __name__ == "__main__":
    unittest.main()
