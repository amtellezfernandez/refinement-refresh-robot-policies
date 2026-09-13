"""Tests for a proposed schedule, not evidence of executed robot evaluations."""
import collections
import copy
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from make_evaluation_plan import (
    CHECKPOINT_REVISION, CONDITIONS, DEFAULT_SCHEDULING_SEED, POLICY_SEEDS,
    STATE_IDS, SUITES, TASK_IDS, digest, make_plan, write_plan,
)

ROOT = Path(__file__).resolve().parent


class EvaluationPlanTests(unittest.TestCase):
    def test_expected_count_and_full_factorial(self):
        plan = make_plan()
        self.assertEqual(plan["design"]["block_count"], 80)
        self.assertEqual(plan["design"]["planned_attempt_count"], 320)
        expected = set(itertools.product(SUITES, TASK_IDS, STATE_IDS,
                                         POLICY_SEEDS, CONDITIONS))
        actual = set()
        for row in plan["schedule"]:
            identity = row["episode_identity"]
            actual.add((identity["suite"], identity["task_id"],
                        identity["initial_state_id"], identity["policy_seed"],
                        (row["h"], row["denoising_steps"])))
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(plan["schedule"]))

    def test_contiguous_complete_blocks(self):
        rows = make_plan()["schedule"]
        for start in range(0, len(rows), 4):
            block = rows[start:start + 4]
            self.assertEqual({row["block_index"] for row in block}, {start // 4})
            self.assertEqual(len({row["block_id"] for row in block}), 1)
            self.assertEqual({row["condition_index"] for row in block}, set(range(4)))
            self.assertEqual({(row["h"], row["denoising_steps"]) for row in block},
                             set(CONDITIONS))
            self.assertTrue(all(row["episode_identity"] == block[0]["episode_identity"]
                                for row in block))
            self.assertEqual(sum(row["native_reference"] for row in block), 1)

    def test_equal_task_condition_weight(self):
        counts = collections.Counter(
            (row["episode_identity"]["suite"], row["episode_identity"]["task_id"],
             row["condition_id"]) for row in make_plan()["schedule"])
        self.assertEqual(len(counts), 32)
        self.assertEqual(set(counts.values()), {10})

    def test_determinism_and_seed_changes_order_only(self):
        first = make_plan()
        self.assertEqual(first, make_plan())
        second = make_plan(DEFAULT_SCHEDULING_SEED + 1)
        self.assertNotEqual(first["schedule"], second["schedule"])
        self.assertEqual(
            {row["planned_run_id"] for row in first["schedule"]},
            {row["planned_run_id"] for row in second["schedule"]})
        first_blocks = [row["block_id"] for row in first["schedule"][::4]]
        second_blocks = [row["block_id"] for row in second["schedule"][::4]]
        self.assertNotEqual(first_blocks, second_blocks)
        orders = {tuple(row["condition_id"] for row in first["schedule"][i:i+4])
                  for i in range(0, 320, 4)}
        self.assertGreater(len(orders), 1)

    def test_stable_unique_identities_and_native_reference(self):
        rows = make_plan()["schedule"]
        self.assertEqual(len({row["planned_run_id"] for row in rows}), 320)
        self.assertEqual(len({row["block_id"] for row in rows}), 80)
        self.assertEqual([row["schedule_index"] for row in rows], list(range(320)))
        for row in rows:
            self.assertEqual(row["native_reference"],
                             (row["h"], row["denoising_steps"]) == (1, 10))
            self.assertEqual(row["planned_run_id"],
                             row["block_id"] + "." + row["condition_id"])

    def test_digest_authenticates_canonical_schedule(self):
        plan = make_plan()
        self.assertEqual(plan["scheduling"]["schedule_sha256"], digest(plan["schedule"]))
        altered = copy.deepcopy(plan["schedule"])
        altered[0]["h"] = 999
        self.assertNotEqual(digest(altered), plan["scheduling"]["schedule_sha256"])
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))

    def test_unverified_status_and_absence_of_results(self):
        plan = make_plan()
        self.assertEqual(plan["status"], "requires_runtime_and_holdout_verification")
        self.assertEqual(plan["checkpoint"]["revision"], CHECKPOINT_REVISION)
        self.assertIsNone(plan["checkpoint"]["runtime_file_hashes"])
        self.assertTrue(all(value == "unverified"
                            for value in plan["external_gates"].values()))
        self.assertEqual(plan["native_config"]["condition_id"], "h1_d10")
        for row in plan["schedule"]:
            self.assertTrue({"success", "outcome", "elapsed_s", "actual_state_hash",
                             "result", "attempt_id"}.isdisjoint(row))

    def test_seed_type_validation(self):
        for seed in (True, 1.2, "123", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_plan(seed)

    def test_no_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "plan.json"
            first, second = make_plan(), make_plan(7)
            write_plan(target, first)
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                write_plan(target, second)
            self.assertEqual(target.read_bytes(), original)
            write_plan(target, second, force=True)
            self.assertEqual(json.loads(target.read_text()), second)

    def test_cli_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "plan.json"
            command = [sys.executable, str(ROOT / "make_evaluation_plan.py"),
                       "--output", str(target)]
            created = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(created.returncode, 0, created.stderr)
            refused = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Refusing to overwrite", refused.stderr)
            replaced = subprocess.run(command + ["--force", "--seed", "7"],
                                      capture_output=True, text=True)
            self.assertEqual(replaced.returncode, 0, replaced.stderr)
            self.assertEqual(json.loads(target.read_text())["scheduling"]["seed"], 7)

    def test_checked_in_plan_matches_generator(self):
        archived = json.loads((ROOT / "planned_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(archived, make_plan())


if __name__ == "__main__":
    unittest.main()
