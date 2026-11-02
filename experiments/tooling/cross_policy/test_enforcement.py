"""Regressions for protections that were recorded but not enforced. No model, GPU or simulator; needs NumPy."""
import argparse
import hashlib
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import build_protocols as bp  # noqa: E402
import launch_guard  # noqa: E402
from common import CONFIRMATION_SEEDS, STATE_SLOTS, SUITES, TASK_IDS, file_sha, sha  # noqa: E402
from journal import AttemptJournal  # noqa: E402


def rows_for(model="smolvla"):
    states = {"{}.task{:03d}.slot{}".format(s, t, k): {"state_sha256": sha([s, t, k])}
              for s in SUITES for t in TASK_IDS for k in STATE_SLOTS}
    conditions = [dict(c, h=bp.resolve(c, model)[0], d=bp.resolve(c, model)[1]) for c in bp.CONDITIONS]
    return bp.make_schedule(bp.CONFIRMATION, model, CONFIRMATION_SEEDS, conditions, states)


def complete_records(rows, seed=1):
    rng = random.Random(seed)
    return [{"planned_run_id": r["planned_run_id"], "attempt_number": 0, "status": "completed", "schedule_row": r,
             "settled_state_sha256": "settled-" + r["block_id"], "settled_state_matches_block": True,
             "outcome": {"success": rng.random() < .7}} for r in rows]


class ResetStateInAnalysis(unittest.TestCase):
    def test_matching_blocks_complete(self):
        rows = rows_for()
        self.assertEqual(analyze.contrast(rows, complete_records(rows), "R22", "R25")["status"], "complete")

    def test_mismatch_blocks_the_contrast(self):
        rows = rows_for()
        records = complete_records(rows)
        next(r for r in records if r["planned_run_id"].endswith(".R22"))["settled_state_sha256"] = "different"
        result = analyze.contrast(rows, records, "R22", "R25")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reset_state_failures"], 1)

    def test_missing_hash_is_not_read_as_a_match(self):
        rows = rows_for()
        records = complete_records(rows)
        del next(r for r in records if r["planned_run_id"].endswith(".R25"))["settled_state_sha256"]
        self.assertEqual(analyze.contrast(rows, records, "R22", "R25")["status"], "incomplete")


class ReportedOutputs(unittest.TestCase):
    def test_paired_state_differences_are_reported(self):
        rows = rows_for()
        result = analyze.contrast(rows, complete_records(rows), "R22", "R25")
        self.assertEqual(len(result["paired_state_differences"]), 200)

    def test_timing_stays_separate_and_cuda_is_not_substituted(self):
        rows = rows_for()
        records = complete_records(rows)
        for record in records:
            record["timing"] = {"rollout_wall_s": 2.0, "generation_host_s": [.1, .3]}
        timing = analyze.analyze_model(rows, records)["timing"]["R22"]
        self.assertEqual(timing["rollout_wall_s_median"], 2.0)
        self.assertEqual(timing["generation_host_s_median"], .2)
        self.assertEqual(timing["generation_cuda_ms_median"], "UNAVAILABLE")

    def test_arithmetic_is_marked_rather_than_silently_absent(self):
        rows = rows_for()
        self.assertIn("R13", analyze.analyze_model(rows, complete_records(rows))["arithmetic"])


class RunnerStopsAndEnforces(unittest.TestCase):
    """Drives the real client_openpi.attempts with the simulator and policy replaced."""
    def setUp(self):
        import numpy as np
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        state = np.arange(4, dtype=np.float64)
        np.savez(self.root / "states.npz", key=state)
        digest = hashlib.sha256(state.tobytes()).hexdigest()
        ident = {"suite": "libero_object", "task_id": 1, "state_slot": 0, "policy_seed": 810001}
        self.rows = [{"planned_run_id": "b.{}".format(c), "block_id": "b", "episode_identity": ident, "h": 2, "d": 2,
                      "state_key": "key", "state_sha256": digest, "condition_id": c} for c in ("R22", "R25", "R52")]
        (self.root / "schedule.json").write_text(json.dumps({"schedule": self.rows}))
        self.args = argparse.Namespace(schedule=self.root / "schedule.json", states=self.root / "states.npz",
                                       output=self.root / "out", port=8000)
        (self.root / "out").mkdir()
        import client_openpi
        self.client = client_openpi
        benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_object": lambda: mock.Mock()})
        libero_inner = types.ModuleType("libero.libero")
        libero_inner.benchmark = benchmark
        patches = [mock.patch.dict(sys.modules, {"libero": types.ModuleType("libero"), "libero.libero": libero_inner}),
                   mock.patch.object(client_openpi, "provenance", lambda a: {"sha256": "0" * 64}),
                   mock.patch.object(client_openpi, "connect", lambda p: object()),
                   mock.patch.object(client_openpi, "make_env", lambda t: (mock.Mock(), "prompt"))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def records(self):
        return AttemptJournal(self.root / "out/attempts").records()

    def fake_rollout(self, settled_by_call):
        import numpy as np
        calls = iter(settled_by_call)
        def rollout(row, env, prompt, client, state, record_video=False, expected_reset=None):
            settled = next(calls)
            from attempt_loop import check_reset
            check_reset((settled, "observation"), expected_reset)
            return {"outcome": {"success": True}, "settled_state_sha256": settled, "first_observation_sha256": "observation", "queries": [],
                    "actions": np.zeros((1, 7)), "timing": {}, "memory": None}
        return rollout

    def test_failed_attempt_stops_the_run(self):
        def crashed(*args, **kwargs):
            raise ConnectionError("server gone")
        with mock.patch.object(self.client, "rollout", crashed):
            with self.assertRaises(RuntimeError):
                self.client.attempts(self.args)
        self.assertEqual([r["status"] for r in self.records()], ["failed"])

    def test_reset_mismatch_is_a_failed_attempt_and_stops(self):
        with mock.patch.object(self.client, "rollout", self.fake_rollout(["same", "other", "same"])):
            with self.assertRaises(RuntimeError):
                self.client.attempts(self.args)
        self.assertEqual(sorted(r["status"] for r in self.records()), ["completed", "failed"])

    def test_reference_survives_a_restart(self):
        with mock.patch.object(self.client, "rollout", self.fake_rollout(["same"])):
            with self.assertRaises(Exception):  # The fake has one rollout for three rows.
                self.client.attempts(self.args)
        with mock.patch.object(self.client, "rollout", self.fake_rollout(["other"])):
            with self.assertRaises(RuntimeError):
                self.client.attempts(self.args)
        statuses = [r["status"] for r in self.records()]
        self.assertEqual(statuses.count("completed"), 1)
        self.assertEqual(statuses.count("failed"), 2)

    def test_real_client_rollout_refuses_before_any_policy_query(self):
        import numpy as np
        from attempt_loop import ResetStateMismatch
        env, policy_client = mock.Mock(), mock.Mock()
        env.step.return_value = ({}, 0, False, {})
        env.sim.get_state.return_value.flatten.return_value = np.zeros(4)
        with mock.patch.object(self.client, "policy_input", lambda o, p: {"prompt": p}):
            for expected in (("other", "x"), (self.client.digest(np.zeros(4)), "other-observation")):
                with self.assertRaises(ResetStateMismatch):
                    self.client.rollout(self.rows[0], env, "prompt", policy_client, np.zeros(4), expected_reset=expected)
        policy_client.infer.assert_not_called()


class ProvenanceInABundle(unittest.TestCase):
    def test_commit_comes_from_the_bundle_marker_when_git_is_absent(self):
        import provenance
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "BUNDLE_COMMIT").write_text("abc123\n")
            self.assertEqual(provenance.git_commit(Path(directory)), "abc123")


class ServerProvenance(unittest.TestCase):
    def test_server_records_its_own_runtime_and_normalization_hash(self):
        import server_openpi
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            norm = root / "ckpt/assets/physical-intelligence/libero/norm_stats.json"
            norm.parent.mkdir(parents=True)
            norm.write_text("{}")
            args = argparse.Namespace(checkpoint=root / "ckpt", output=root / "out", mode="serve", port=8000)
            server_openpi.write_provenance(args)
            record = json.loads((root / "out/provenance.json").read_text())
            self.assertEqual(record["model"], "pi05_server")
            self.assertEqual(record["normalization_sha256"], file_sha(norm))
            self.assertEqual(record["checkpoint_tree"]["files"], 1)
            self.assertIn("jax", record["versions"])


class BindingKeepsOrder(unittest.TestCase):
    def test_binding_states_changes_hash_not_order(self):
        rows = rows_for()
        pending = {k: {"state_sha256": None} for k in {r["state_key"] for r in rows}}
        conditions = [dict(c, h=bp.resolve(c, "smolvla")[0], d=bp.resolve(c, "smolvla")[1]) for c in bp.CONDITIONS]
        unbound = bp.make_schedule(bp.CONFIRMATION, "smolvla", CONFIRMATION_SEEDS, conditions, pending)
        self.assertEqual([r["planned_run_id"] for r in unbound], [r["planned_run_id"] for r in rows])
        self.assertNotEqual(sha(unbound), sha(rows))


if __name__ == "__main__":
    unittest.main()
