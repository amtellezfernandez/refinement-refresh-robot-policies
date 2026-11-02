"""Offline tests: journal, schedules, analysis with known truth, launch guard. No model or GPU."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import build_protocols as bp  # noqa: E402
import launch_guard  # noqa: E402
import make_primary_schedule as primary_schedule  # noqa: E402
import server_openpi  # noqa: E402
from common import CONFIRMATION_SEEDS, OBSERVATION_AGE_SEEDS, STATE_SLOTS, SUITES, TASK_IDS, file_sha, sha  # noqa: E402
from journal import AttemptJournal, TerminalRecordError  # noqa: E402


def fake_states():
    return {"{}.task{:03d}.slot{}".format(s, t, k): {"state_sha256": sha([s, t, k])}
            for s in SUITES for t in TASK_IDS for k in STATE_SLOTS}


def conditions(model):
    return [dict(c, h=bp.resolve(c, model)[0], d=bp.resolve(c, model)[1]) for c in bp.CONDITIONS]


def schedule(model="smolvla"):
    return bp.make_schedule(bp.CONFIRMATION, model, CONFIRMATION_SEEDS, conditions(model), fake_states())


def records_from(rows, probability, rng):
    records = []
    for row in rows:
        success = rng.random() < probability[row["condition_id"]]
        records.append({"planned_run_id": row["planned_run_id"], "attempt_number": 0, "status": "completed",
                        "outcome": {"success": success}, "settled_state_matches_block": True})
    return records


class Population(unittest.TestCase):
    def test_counts_match_specification(self):
        rows = schedule()
        self.assertEqual(len(rows), 2000)
        self.assertEqual(len({r["block_id"] for r in rows}), 400)
        self.assertEqual(len({r["planned_run_id"] for r in rows}), 2000)

    def test_every_block_has_all_conditions_and_one_state(self):
        blocks = {}
        for row in schedule():
            blocks.setdefault(row["block_id"], []).append(row)
        for rows in blocks.values():
            self.assertEqual({r["condition_id"] for r in rows}, {"NATIVE", "R22", "R25", "R52", "R55"})
            self.assertEqual(len({r["state_sha256"] for r in rows}), 1)

    def test_conditions_resolve_exactly(self):
        smol = {c["id"]: (c["h"], c["d"]) for c in conditions("smolvla")}
        pi = {c["id"]: (c["h"], c["d"]) for c in conditions("pi05")}
        self.assertEqual(smol, {"NATIVE": (1, 10), "R22": (2, 2), "R25": (2, 5), "R52": (5, 2), "R55": (5, 5)})
        self.assertEqual(pi["NATIVE"], (5, 10))

    def test_schedule_is_deterministic_and_model_specific(self):
        self.assertEqual(sha(schedule()), sha(schedule()))
        self.assertNotEqual(sha(schedule("smolvla")), sha(schedule("pi05")))

    def test_seeds_are_disjoint_from_every_recorded_prior_seed(self):
        prior = {1000, 2000, 2001, 3000, 3001}
        self.assertFalse(prior & set(CONFIRMATION_SEEDS))
        self.assertFalse(prior & set(OBSERVATION_AGE_SEEDS))
        self.assertFalse(set(CONFIRMATION_SEEDS) & set(OBSERVATION_AGE_SEEDS))


class PendingStates(unittest.TestCase):
    def test_pending_binding_is_explicit_and_cannot_be_run(self):
        manifest, states = bp.load_states("cohort_that_has_no_states")
        self.assertEqual(manifest["binding"], "PENDING_STATE_GENERATION")
        rows = bp.make_schedule("c", "smolvla", CONFIRMATION_SEEDS, conditions("smolvla"), states)
        # Both runners refuse a schedule unless every row carries a state hash.
        self.assertFalse(all(row["state_sha256"] for row in rows))


class Journal(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.journal = AttemptJournal(self.temporary.name)
        self.row = {"planned_run_id": "cohort.model.identity.R22"}

    def tearDown(self):
        self.temporary.cleanup()

    def test_record_exists_before_execution(self):
        path = self.journal.begin(self.row, "prov", ["cmd"])
        self.assertEqual(json.loads(path.read_text())["status"], "planned")

    def test_terminal_record_cannot_change(self):
        path = self.journal.begin(self.row, "prov", ["cmd"])
        self.journal.update(path, "running")
        self.journal.update(path, "completed", outcome={"success": True})
        before = path.read_bytes()
        with self.assertRaises(TerminalRecordError):
            self.journal.update(path, "failed")
        self.assertEqual(before, path.read_bytes())

    def test_no_silent_retry(self):
        self.journal.begin(self.row, "prov", ["cmd"])
        with self.assertRaises(ValueError):
            self.journal.begin(self.row, "prov", ["cmd"])

    def test_retry_keeps_predecessor_and_needs_a_reason(self):
        first = self.journal.begin(self.row, "prov", ["cmd"])
        self.journal.update(first, "failed", error="x")
        second = self.journal.begin(self.row, "prov", ["cmd"], retry_reason="reviewed infrastructure fault")
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.journal.records()), 2)

    def test_crash_recovery_marks_interrupted_and_keeps_record(self):
        path = self.journal.begin(self.row, "prov", ["cmd"])
        self.journal.update(path, "running")
        self.assertEqual(self.journal.reconcile_interrupted("restart"), [path.name])
        self.assertEqual(json.loads(path.read_text())["status"], "interrupted")

    def test_protected_fields(self):
        path = self.journal.begin(self.row, "prov", ["cmd"])
        with self.assertRaises(ValueError):
            self.journal.update(path, "running", schedule_row={})


class Analysis(unittest.TestCase):
    def test_all_five_conditions_are_reported(self):
        rows = schedule()
        report = analyze.analyze_model(rows, records_from(rows, dict.fromkeys(["NATIVE", "R22", "R25", "R52", "R55"], .5),
                                                          random.Random(1)))
        self.assertEqual(set(report["conditions"]), {"NATIVE", "R22", "R25", "R52", "R55"})

    def test_planted_effect_is_recovered(self):
        rows = schedule()
        p = {"NATIVE": .7, "R22": .85, "R25": .6, "R52": .85, "R55": .6}
        report = analyze.analyze_model(rows, records_from(rows, p, random.Random(2)))
        # One draw need not cover the truth, so this checks detection and location only.
        self.assertGreater(report["primary"]["ci95_two_sided"][0], 0)
        self.assertLess(abs(report["primary"]["macro_difference_points"] - 25.0), 12.0)

    def test_null_effect_interval_covers_zero(self):
        rows = schedule()
        p = dict.fromkeys(["NATIVE", "R22", "R25", "R52", "R55"], .7)
        report = analyze.analyze_model(rows, records_from(rows, p, random.Random(3)))
        low, high = report["primary"]["ci95_two_sided"]
        self.assertLess(low, 0)
        self.assertGreater(high, 0)

    def test_bootstrap_nominal_coverage_under_known_truth(self):
        # Regression floor on the real design shape, not a claim of nominal coverage. Measured
        # over 400 synthetic trials per scenario: 90-92% two-sided, 93-94% one-sided, because five
        # states per task shrink the bootstrap variance by (n-1)/n. See audit review item R3.
        rows = schedule()
        p = {"NATIVE": .7, "R22": .75, "R25": .70, "R52": .75, "R55": .7}
        covered = 0
        for trial in range(60):
            rng = random.Random(100 + trial)
            result = analyze.contrast(rows, records_from(rows, p, rng), "R22", "R25")
            low, high = result["ci95_two_sided"]
            covered += low <= 5.0 <= high
        self.assertGreaterEqual(covered, 50)

    def test_quality_rule_passes_only_above_margin(self):
        complete = {"status": "complete"}
        self.assertEqual(analyze.quality_decision(dict(complete, lower_bound_one_sided_95=-4.99))["decision"],
                         "quality_preserved_within_margin")
        self.assertEqual(analyze.quality_decision(dict(complete, lower_bound_one_sided_95=-5.0))["decision"],
                         "quality_loss_or_inconclusive")

    def test_never_says_equivalent(self):
        complete = {"status": "complete"}
        for bound in (-1.0, -9.0):
            text = json.dumps(analyze.quality_decision(dict(complete, lower_bound_one_sided_95=bound)))
            self.assertNotIn("equivalent", text.lower())

    def test_failed_attempt_blocks_test_and_is_not_counted_as_failure(self):
        rows = schedule()
        records = records_from(rows, dict.fromkeys(["NATIVE", "R22", "R25", "R52", "R55"], .5), random.Random(4))
        target = next(r for r in records if r["planned_run_id"].endswith(".R22"))
        target["status"] = "failed"
        report = analyze.analyze_model(rows, records)
        self.assertEqual(report["primary"]["status"], "incomplete")
        self.assertEqual(report["secondary"]["status"], "incomplete")
        self.assertEqual(report["quality_preservation"]["decision"], "test_incomplete")
        self.assertEqual(report["coverage"]["not_completed"][0]["status"], "failed")

    def test_record_outside_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze.coverage(schedule(), [{"planned_run_id": "elsewhere", "status": "completed", "attempt_number": 0}])

    def test_cross_model_is_signs_only(self):
        summary = analyze.cross_model_summary({"a": {"primary": {"status": "complete", "macro_difference_points": 3.}},
                                               "b": {"primary": {"status": "complete", "macro_difference_points": -2.}}})
        self.assertEqual(summary["sign_agreement"], "disagree")
        self.assertFalse(any("pooled_estimate" in key for key in summary))


    def test_unverified_reset_never_enters_the_primary_estimate(self):
        rows = schedule()
        records = records_from(rows, dict.fromkeys(["NATIVE", "R22", "R25", "R52", "R55"], .5),
                               random.Random(23))
        record = next(r for r in records if r["planned_run_id"].endswith(".R22"))
        record["settled_state_matches_block"] = False
        report = analyze.analyze_model(rows, records)
        self.assertEqual(report["primary"]["status"], "incomplete")
        self.assertIn("reset_state_not_verified",
                      [x["status"] for x in report["coverage"]["not_completed"]])

    def test_absent_secondary_arm_is_not_a_completed_contrast(self):
        rows = [r for r in schedule() if r["condition_id"] in ("R22", "R25")]
        records = records_from(rows, {"R22": .7, "R25": .65}, random.Random(24))
        result = analyze.analyze_model(rows, records)
        self.assertEqual(result["primary"]["status"], "complete")
        self.assertEqual(result["secondary"]["status"], "not_planned")
        self.assertEqual(result["quality_preservation"]["decision"], "test_incomplete")


class PrimarySchedule(unittest.TestCase):
    def test_primary_schedule_is_exact_and_has_distinct_source_hash(self):
        rows = schedule()
        doc = bp.schedule_document(bp.CONFIRMATION, "smolvla", rows,
                                   {"states_sha256": sha("sample states"), "binding": "BOUND"})
        result = primary_schedule.derive(doc)
        self.assertEqual(len(result["schedule"]), 800)
        self.assertEqual(result["identity_count"], 400)
        self.assertEqual(set(r["condition_id"] for r in result["schedule"]), {"R22", "R25"})
        self.assertEqual(result["source_schedule_sha256"], doc["schedule_sha256"])
        self.assertNotEqual(result["schedule_sha256"], doc["schedule_sha256"])

    def test_unbound_source_is_rejected(self):
        doc = bp.schedule_document(bp.CONFIRMATION, "smolvla", schedule(),
                                   {"states_sha256": None, "binding": "PENDING_STATE_GENERATION"})
        with self.assertRaises(ValueError):
            primary_schedule.derive(doc)


class Guard(unittest.TestCase):
    """A test release is never a real GPU authorization."""
    def make_root(self, status="PREPARED"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "experiments/protocols").mkdir(parents=True)
        for name in launch_guard.PROTOCOLS.values():
            (root / "experiments/protocols" / (name + ".json")).write_text(
                json.dumps({"status": status}))
        return root

    def write_schedule(self, root, model="smolvla"):
        rows = schedule(model)
        full = bp.schedule_document(bp.CONFIRMATION, model, rows,
                                    {"states_sha256": sha("sample states"), "binding": "BOUND"})
        selected = primary_schedule.derive(full)
        path = root / "experiments/runs" / bp.CONFIRMATION / model / "primary_r22_r25_schedule.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selected, sort_keys=True))
        return path

    def write_release(self, root, schedule_path, approved_by="Alba", protocol_sha=None):
        name = bp.CONFIRMATION
        if protocol_sha is None:
            protocol_sha = file_sha(root / "experiments/protocols" / (name + ".json"))
        document = json.loads(schedule_path.read_text("utf-8"))
        record = {"approved_by": approved_by, "protocol_sha256": protocol_sha,
                  "schedule_sha256_by_model": {document["model"]: document["schedule_sha256"]},
                  "schedule_file_sha256_by_model": {document["model"]: file_sha(schedule_path)},
                  "approval_scope": "PRIMARY_R22_R25_ONLY",
                  "approved_conditions": ["R22", "R25"],
                  "approved_attempts_per_model": 800}
        path = root / "review/releases" / (name + ".release.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return path

    def test_preflight_refused_without_reservation(self):
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("preflight", environ={}, processes=lambda: [])

    def test_preflight_refused_when_gpu_is_occupied(self):
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("preflight", environ={"ALBA_GPU_RESERVATION": "R1"},
                               processes=lambda: ["1, 10 MiB"])

    def test_preflight_allowed_with_reservation(self):
        self.assertEqual(launch_guard.check("preflight", environ={"ALBA_GPU_RESERVATION": "R1"},
                                            processes=lambda: [])["reservation"], "R1")

    def test_confirmation_refused_without_release(self):
        root = self.make_root()
        path = self.write_schedule(root)
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("confirmation", root=root, environ={"ALBA_GPU_RESERVATION": "R1"},
                               processes=lambda: [], schedule_path=path, model="smolvla")

    def test_confirmation_refused_when_protocol_changed_after_release(self):
        root = self.make_root()
        path = self.write_schedule(root)
        self.write_release(root, path, protocol_sha="0" * 64)
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("confirmation", root=root, environ={"ALBA_GPU_RESERVATION": "R1"},
                               processes=lambda: [], schedule_path=path, model="smolvla")

    def test_confirmation_allowed_only_by_matching_release(self):
        root = self.make_root()
        path = self.write_schedule(root)
        self.write_release(root, path)
        self.assertEqual(launch_guard.check(
            "confirmation", root=root, environ={"ALBA_GPU_RESERVATION": "R1"},
            processes=lambda: [], schedule_path=path, model="smolvla")["phase"], "confirmation")

    def test_confirmation_refused_when_schedule_is_modified(self):
        root = self.make_root()
        path = self.write_schedule(root)
        self.write_release(root, path)
        path.write_text(path.read_text() + " ")
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("confirmation", root=root, environ={"ALBA_GPU_RESERVATION": "R1"},
                               processes=lambda: [], schedule_path=path, model="smolvla")

    def test_confirmation_refused_without_a_schedule_path(self):
        root = self.make_root()
        path = self.write_schedule(root)
        self.write_release(root, path)
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("confirmation", root=root, environ={"ALBA_GPU_RESERVATION": "R1"},
                               processes=lambda: [], model="smolvla")

    def test_confirmation_refused_if_release_approves_extra_arms(self):
        root = self.make_root()
        path = self.write_schedule(root)
        release = self.write_release(root, path)
        record = json.loads(release.read_text())
        record["approved_conditions"] = ["R22", "R25", "R52"]
        release.write_text(json.dumps(record))
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("confirmation", root=root, environ={"ALBA_GPU_RESERVATION": "R1"},
                               processes=lambda: [], schedule_path=path, model="smolvla")

    def test_blocked_protocol_refused_even_with_release(self):
        root = self.make_root(status="BLOCKED_X")
        path = self.write_schedule(root)
        self.write_release(root, path)
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("confirmation", root=root, environ={"ALBA_GPU_RESERVATION": "R1"},
                               processes=lambda: [], schedule_path=path, model="smolvla")

    def test_this_branch_has_no_release(self):
        self.assertFalse((launch_guard.ROOT / "review/releases").exists())


class OpenPIRequestTrace(unittest.TestCase):
    def test_server_echoes_the_configured_solver_steps(self):
        from unittest.mock import patch

        class FakePolicy:
            metadata = {}
            def infer(self, obs, noise):
                return {"actions": [[0.0] * 7]}

        policy = FakePolicy()
        wrapper = server_openpi.KeyedPolicy(policy)
        with patch.object(server_openpi, "noise_for", return_value=None), \
             patch.object(server_openpi, "memory_stats", return_value={}):
            result = wrapper.infer({"observation/state": [0], "__d__": 2,
                                    "__noise_key__": json.dumps(["identity", 0])})
        self.assertEqual(result["server_configured_num_steps"], 2)
        self.assertEqual(policy._sample_kwargs, {"num_steps": 2})


if __name__ == "__main__":
    unittest.main()
