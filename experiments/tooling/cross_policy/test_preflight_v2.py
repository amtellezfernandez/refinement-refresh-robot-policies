"""Alba's versioned preflight and matched R22/R25 design. No model, GPU or simulator; needs NumPy."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import attempt_loop  # noqa: E402
import build_protocols as bp  # noqa: E402
import launch_guard  # noqa: E402
import preflight_verdict as pv  # noqa: E402
from common import CONFIRMATION_SEEDS, STATE_SLOTS, SUITES, TASK_IDS, file_sha  # noqa: E402

PREFLIGHT = "cross_policy_preflight_v2"
ENV = {"ALBA_GPU_RESERVATION": "R1"}


class AttemptAccounting(unittest.TestCase):
    def test_alba_primary_design_is_800_per_model_and_1600_in_total(self):
        import make_primary_schedule as mp
        states = {"{}.task{:03d}.slot{}".format(s, t, k): {"state_sha256": "h"} for s in SUITES for t in TASK_IDS for k in STATE_SLOTS}
        total = 0
        for model in ("smolvla", "pi05"):
            conditions = [dict(c, h=bp.resolve(c, model)[0], d=bp.resolve(c, model)[1]) for c in bp.CONDITIONS]
            rows = bp.make_schedule(bp.CONFIRMATION, model, CONFIRMATION_SEEDS, conditions, states)
            primary = mp.derive(bp.schedule_document(bp.CONFIRMATION, model, rows, {"states_sha256": "m"}))
            self.assertEqual((primary["attempt_count"], primary["identity_count"]), (800, 400))
            self.assertEqual({r["condition_id"] for r in primary["schedule"]}, {"R22", "R25"})
            total += primary["attempt_count"]
        self.assertEqual(total, 1600)

    def test_no_separate_cohort_competes_with_alba_design(self):
        self.assertFalse((bp.PROTOCOLS / "matched_r22_r25_confirmation_v1.json").exists())
        self.assertFalse((bp.RUNS / "matched_r22_r25_confirmation_v1").exists())

    def test_committed_files_match_the_builder(self):
        for path, text in bp.outputs().items():
            self.assertEqual(path.read_text("utf-8"), text, str(path))

    def test_preflight_protocol_has_a_unique_run_id_and_keeps_the_failed_verdict(self):
        protocol = json.loads((bp.PROTOCOLS / (PREFLIGHT + ".json")).read_text())
        self.assertEqual(protocol["run_id"], "cross_policy_preflight_v2.run001")
        self.assertIn("PREFLIGHT_FAILED", json.dumps(protocol["prior_preflight"]))
        self.assertEqual(set(protocol["unresolved_decisions"]), set(launch_guard.REQUIRED_PREFLIGHT_DECISIONS))
        self.assertEqual(set(launch_guard.REQUIRED_PREFLIGHT_DECISIONS), {"R4", "R6", "R9", "N1"})


class Guard(unittest.TestCase):
    def root_with(self, name):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "experiments/protocols").mkdir(parents=True)
        (root / "review/releases").mkdir(parents=True)
        path = root / "experiments/protocols" / (name + ".json")
        path.write_text(json.dumps({"status": "PREPARED"}))
        return root, path

    def test_deferred_cohorts_are_refused_even_with_a_release(self):
        for phase, name in (("observation_age", "observation_age_confirmation_v1"), ("asynchronous", "asynchronous_confirmation_v1")):
            root, path = self.root_with(name)
            (root / "review/releases" / (name + ".release.json")).write_text(json.dumps(
                {"approved_by": "Alba", "protocol_sha256": file_sha(path)}))
            with self.assertRaises(launch_guard.LaunchRefused):
                launch_guard.check(phase, root=root, environ=ENV, processes=lambda: [], schedule_path=root / "s.json", model="smolvla")

    def preflight(self, root, **release):
        (root / "review/releases" / (PREFLIGHT + ".release.json")).write_text(json.dumps(release))
        return launch_guard.check("preflight_v2", root=root, environ=ENV, processes=lambda: [])

    def complete(self, path, **change):
        decisions = {k: "x" for k in launch_guard.REQUIRED_PREFLIGHT_DECISIONS}
        decisions["R6"] = "OSMesa"
        decisions.update(change)
        return {"approved_by": "Alba", "protocol_sha256": file_sha(path), "decisions": decisions}

    def test_preflight_refused_with_no_release(self):
        root, _ = self.root_with(PREFLIGHT)
        with self.assertRaises(launch_guard.LaunchRefused):
            launch_guard.check("preflight_v2", root=root, environ=ENV, processes=lambda: [])

    def test_preflight_refused_when_any_decision_is_missing(self):
        root, path = self.root_with(PREFLIGHT)
        release = self.complete(path)
        del release["decisions"]["R9"]
        with self.assertRaises(launch_guard.LaunchRefused):
            self.preflight(root, **release)

    def test_preflight_refused_for_a_renderer_the_harness_does_not_implement(self):
        root, path = self.root_with(PREFLIGHT)
        with self.assertRaises(launch_guard.LaunchRefused):
            self.preflight(root, **self.complete(path, R6="EGL"))

    def test_preflight_refused_if_protocol_changed(self):
        root, path = self.root_with(PREFLIGHT)
        release = self.complete(path)
        path.write_text("{}")
        with self.assertRaises(launch_guard.LaunchRefused):
            self.preflight(root, **release)

    def test_preflight_allowed_by_complete_release(self):
        root, path = self.root_with(PREFLIGHT)
        self.assertEqual(self.preflight(root, **self.complete(path))["phase"], "preflight_v2")


class Provenance(unittest.TestCase):
    def test_packages_are_read_without_pip(self):
        """The three model environments on the RTX have no pip module; this uv environment has none either."""
        import provenance
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory))
            record = provenance.capture("t", args, {})
        self.assertNotIn("UNKNOWN", str(record["packages"]))
        self.assertTrue(any(line.lower().startswith("numpy==") for line in record["packages"]), record["packages"])


class PolicyStub:
    metadata = {}

    def __init__(self):
        self._sample_kwargs = {}
        self._sample_actions = self.sample

    def sample(self, rng, observation, **kwargs):
        return kwargs

    def infer(self, obs, noise=None):
        self._sample_actions("rng", obs, noise=noise, **self._sample_kwargs)
        return {"actions": None}


class Receipt(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, {"jax": types.SimpleNamespace(
            local_devices=lambda: [types.SimpleNamespace(memory_stats=lambda: {})])})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_server_reports_num_steps_at_the_sampling_boundary(self):
        import server_openpi
        policy = server_openpi.KeyedPolicy(PolicyStub())
        for d in (2, 5, 10):
            self.assertEqual(policy.infer({"__d__": d, "__noise_key__": "[1]"})["received_num_steps"], d)

    def test_server_does_not_report_what_was_not_received(self):
        import server_openpi
        stub = PolicyStub()
        policy = server_openpi.KeyedPolicy(stub)
        original = stub._sample_actions
        stub.infer = lambda obs, noise=None: (original("rng", obs, noise=noise), {"actions": None})[1]
        self.assertIsNone(policy.infer({"__d__": 2, "__noise_key__": "[1]"})["received_num_steps"])

    def test_client_stops_when_the_server_did_not_receive_d(self):
        import numpy as np
        import client_openpi
        client, env = mock.Mock(), mock.Mock()
        client.infer.return_value = {"actions": np.zeros((10, 7)), "host_generation_s": 0.0, "memory": {}, "received_num_steps": 10,
                                     "server_configured_num_steps": 2}
        env.step.return_value = ({}, 0, False, {})
        env.sim.get_state.return_value.flatten.return_value = np.zeros(4)
        row = {"episode_identity": {"suite": "libero_goal"}, "h": 2, "d": 2}
        with mock.patch.object(client_openpi, "policy_input", lambda o, p: {"prompt": p, "observation/image": np.zeros((2, 2))}):
            with self.assertRaisesRegex(ValueError, "sampler received 10"):
                client_openpi.rollout(row, env, "p", client, np.zeros(4))


class NormalizationAtRuntime(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "norm.json"
        stats = {"mean": [0.], "std": [1.], "q01": [-1.], "q99": [1.]}
        self.path.write_text(json.dumps({"norm_stats": {"state": stats, "actions": stats}}))

    def policy(self, q99=1.0, quantiles=True):
        stats = types.SimpleNamespace(mean=self.np.array([0.]), std=self.np.array([1.]), q01=self.np.array([-1.]),
                                      q99=self.np.array([q99]))
        import server_openpi
        Unnormalize = type("Unnormalize", (), {})
        transform = Unnormalize()
        transform.norm_stats, transform.use_quantiles = {"state": stats, "actions": stats}, quantiles
        return server_openpi, types.SimpleNamespace(_output_transform=types.SimpleNamespace(transforms=[object(), transform]))

    def test_matching_stats_accepted(self):
        server, policy = self.policy()
        self.assertTrue(server.verify_loaded_normalization(policy, self.path)["use_quantiles"])

    def test_different_stats_refused(self):
        server, policy = self.policy(q99=2.0)
        with self.assertRaises(SystemExit):
            server.verify_loaded_normalization(policy, self.path)

    def test_non_quantile_refused(self):
        server, policy = self.policy(quantiles=False)
        with self.assertRaises(SystemExit):
            server.verify_loaded_normalization(policy, self.path)


class FreshEnvironmentPerAttempt(unittest.TestCase):
    """Probe 2026-11-02: reusing an OpenPI client environment breaks exact settling on libero_object."""
    def test_client_builds_one_environment_per_attempt(self):
        import numpy as np
        import client_openpi
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = np.arange(4, dtype=np.float64)
            np.savez(root / "states.npz", key=state)
            ident = {"suite": "libero_object", "task_id": 1, "state_slot": 0, "policy_seed": 810001}
            rows = [{"planned_run_id": "b." + c, "block_id": "b", "episode_identity": ident, "h": 2, "d": 2, "state_key": "key",
                     "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(), "condition_id": c} for c in ("R22", "R25")]
            (root / "s.json").write_text(json.dumps({"schedule": rows}))
            (root / "out").mkdir()
            args = argparse.Namespace(schedule=root / "s.json", states=root / "states.npz", output=root / "out", port=1)
            inner = types.ModuleType("libero.libero")
            inner.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_object": lambda: mock.Mock()})
            built, closed = [], []
            def make_env(task):
                env = mock.Mock()
                env.close.side_effect = lambda: closed.append(1)
                built.append(env)
                return env, "p"
            def rollout(row, env, prompt, client, state, record_video=False, expected_reset=None):
                return {"outcome": {"success": True}, "settled_state_sha256": "s", "first_observation_sha256": "o", "queries": [], "actions": np.zeros((1, 7)), "timing": {}, "memory": None}
            with mock.patch.dict(sys.modules, {"libero": types.ModuleType("libero"), "libero.libero": inner}), \
                    mock.patch.object(client_openpi, "provenance", lambda a: {"sha256": "0" * 64}), \
                    mock.patch.object(client_openpi, "connect", lambda p: object()), \
                    mock.patch.object(client_openpi, "make_env", make_env), mock.patch.object(client_openpi, "rollout", rollout):
                client_openpi.attempts(args)
            self.assertEqual(len(built), 2)
            self.assertEqual(len(closed), 2)


def condition_results(**change):
    base = {"settled_state_sha256": "s", "action_shape": [5, 7], "received_d": True, "first_action_sha256": "a"}
    results = {"NATIVE": dict(base, first_action_sha256="n"), "R22": dict(base), "R52": dict(base),
               "R25": dict(base, first_action_sha256="b"), "R55": dict(base, first_action_sha256="b")}
    for key, value in change.items():
        results[key] = dict(results[key], **value)
    return results


class ConditionCheck(unittest.TestCase):
    def test_all_properties_pass(self):
        self.assertEqual(attempt_loop.summarize_condition_check(condition_results(), negative_control_refused=True)["status"], "PASS")

    def test_matched_first_action_difference_fails(self):
        result = attempt_loop.summarize_condition_check(condition_results(R52={"first_action_sha256": "z"}), negative_control_refused=True)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["matched_first_actions"]["R22=R52"])

    def test_reset_difference_fails(self):
        self.assertEqual(attempt_loop.summarize_condition_check(condition_results(R25={"settled_state_sha256": "z"}), negative_control_refused=True)["status"], "FAIL")

    def test_d_not_received_fails(self):
        self.assertEqual(attempt_loop.summarize_condition_check(condition_results(R55={"received_d": False}), negative_control_refused=True)["status"], "FAIL")

    def test_missing_condition_fails(self):
        results = condition_results()
        del results["R55"]
        self.assertEqual(attempt_loop.summarize_condition_check(results, negative_control_refused=True)["status"], "FAIL")

    def test_reset_negative_control_must_fire(self):
        self.assertEqual(attempt_loop.summarize_condition_check(condition_results(), negative_control_refused=False)["status"], "FAIL")


class SurvivingMutants(unittest.TestCase):
    """Each test closes a gap that a mutation run exposed."""
    def test_missing_native_fails(self):
        results = condition_results()
        del results["NATIVE"]
        self.assertEqual(attempt_loop.summarize_condition_check(results, negative_control_refused=True)["status"], "FAIL")

    def test_a_block_with_no_settled_hash_at_all_is_a_failure(self):
        rows = json.loads((bp.RUNS / bp.CONFIRMATION / "smolvla" / "schedule.json").read_text())["schedule"]
        records = [{"planned_run_id": r["planned_run_id"], "attempt_number": 0, "status": "completed",
                    "settled_state_sha256": r["block_id"], "outcome": {"success": True}} for r in rows]
        block = rows[0]["block_id"]
        for record, row in zip(records, rows):
            if row["block_id"] == block:
                del record["settled_state_sha256"]
        self.assertEqual(analyze.reset_state_failures(rows, records), {block})

    def test_server_does_not_carry_receipt_over_from_the_previous_call(self):
        with mock.patch.dict(sys.modules, {"jax": types.SimpleNamespace(
                local_devices=lambda: [types.SimpleNamespace(memory_stats=lambda: {})])}):
            import server_openpi
            stub = PolicyStub()
            original = stub.sample
            policy = server_openpi.KeyedPolicy(stub)
            self.assertEqual(policy.infer({"__d__": 2, "__noise_key__": "[1]"})["received_num_steps"], 2)
            stub.infer = lambda obs, noise=None: (original("rng", obs, noise=noise), {"actions": None})[1]
            self.assertIsNone(policy.infer({"__d__": 5, "__noise_key__": "[1]"})["received_num_steps"])


class MatchedAnalysis(unittest.TestCase):
    def test_two_condition_schedule_reports_primary_only(self):
        rows = [r for r in json.loads((bp.RUNS / bp.CONFIRMATION / "smolvla" / "schedule.json").read_text())["schedule"]
                if r["condition_id"] in ("R22", "R25")]
        records = [{"planned_run_id": r["planned_run_id"], "attempt_number": 0, "status": "completed", "settled_state_matches_block": True,
                    "settled_state_sha256": r["block_id"], "outcome": {"success": i % 3 > 0}} for i, r in enumerate(rows)]
        report = analyze.analyze_model(rows, records)
        self.assertEqual(report["primary"]["status"], "complete")
        self.assertEqual(report["secondary"]["status"], "not_planned")
        self.assertEqual(report["quality_preservation"]["decision"], "test_incomplete")


class VideoCheck(unittest.TestCase):
    def test_blank_or_constant_frames_are_flagged(self):
        import numpy as np
        import inspect_videos
        moving = [np.full((8, 8, 3), i * 10, dtype=np.uint8) for i in range(5)]
        self.assertTrue(inspect_videos.frame_checks(moving)["ok"])
        self.assertFalse(inspect_videos.frame_checks([np.zeros((8, 8, 3), dtype=np.uint8)] * 5)["ok"])
        self.assertFalse(inspect_videos.frame_checks([])["ok"])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fabricated_run(root):
    """Passing shape for the verdict tool, with every value synthetic."""
    write(root / "start.json", {"run_id": "cross_policy_preflight_v2.run001", "protocol_sha256": "p", "release_sha256": "r"})
    provenance = {"packages": ["numpy==1"], "git_commit": "abc", "gpu": "GPU-1", "versions": {}, "checkpoint_tree": {"files": 1, "sha256": "c"}}
    write(root / "smolvla/provenance.json", dict(provenance))
    write(root / "pi05_server/provenance.json", dict(provenance, normalization_sha256=pv.OPENPI_NORM_SHA256, normalization_loaded={"use_quantiles": True}))
    write(root / "pi05_parity/provenance.json", {"packages": ["numpy==1"], "git_commit": "abc", "gpu": "GPU-1", "versions": {}})
    write(root / "smolvla/profile.json", {str(d): {"host_s": [.1] * 100, "cuda_ms": [1.] * 100, "chunk_shape": [1, 50, 7],
                                                   "counts": {"generations": 100, "denoise_steps": 100 * d}, "peak_reserved_bytes": 1} for d in (2, 5, 10)})
    write(root / "pi05_profile/profile.json", {"status": "completed", "denoise_loop_calls": {"2": 2, "5": 5, "10": 10}, "points": {str(d): {"host_s": [.1] * 100, "chunk_shape": [10, 7], "received_num_steps": [d] * 100,
                                               "memory": {"peak_bytes_in_use": 1}, "fixed_noise_chunk_sha256": "c" + str(d)} for d in (2, 5, 10)}})
    for model, directory in (("smolvla", "smolvla_conditions"), ("pi05", "pi05_conditions")):
        write(root / directory / "conditions.json", {"status": "PASS"})
    for model, directory in (("smolvla", "smolvla"), ("pi05", "pi05_parity")):
        for index in range(400):
            write(root / directory / "parity_attempts" / "p{}.attempt00.json".format(index),
                  {"status": "completed", "planned_run_id": "parity.{}.libero_goal.task{:03d}.ep{}".format(model, index // 10 % 10, index % 10),
                   "schedule_row": {"episode_identity": {"suite": ["libero_spatial", "libero_object", "libero_goal", "libero_10"][index // 100], "task_id": index // 10 % 10}},
                   "outcome": {"success": True}, "first_action_sha256": "a", "action_shape": [5, 7], "action_min": [-1] * 7, "action_max": [1] * 7,
                   "timing": {"rollout_wall_s": 1.0}, "video": "v.mp4"})
        write(root / directory / "videos" / "video_manifest.json", {"videos": 400, "failed": 0, "contact_sheets": 40})
    write(root / "states/state_manifest.json", {"state_count": 200, "states": [{"state_sha256": "h"}] * 200})


class Verdict(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        fabricated_run(self.root)

    def test_complete_run_still_needs_human_review(self):
        report = pv.evaluate(self.root)
        self.assertEqual(report["verdict"], "MECHANICAL_CHECKS_PASSED_AWAITING_PARITY_AND_VIDEO_REVIEW", report["checks"])

    def edit(self, relative, change):
        path = self.root / relative
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def assert_fails(self, check):
        report = pv.evaluate(self.root)
        self.assertEqual(report["verdict"], "PREFLIGHT_FAILED")
        self.assertEqual(report["checks"][check]["status"], "FAIL", report["checks"][check])

    def test_each_requirement_can_fail(self):
        cases = [
            ("smolvla/provenance.json", lambda v: v.update(packages="UNKNOWN: no pip"), "dependency_provenance"),
            ("smolvla/provenance.json", lambda v: v.pop("checkpoint_tree"), "checkpoint_provenance"),
            ("pi05_server/provenance.json", lambda v: v.update(normalization_sha256="0"), "official_normalization"),
            ("smolvla/profile.json", lambda v: v["5"]["counts"].update(denoise_steps=100), "implementation_received_d"),
            ("pi05_profile/profile.json", lambda v: v["points"]["2"].update(received_num_steps=[10] * 100), "implementation_received_d"),
            ("pi05_profile/profile.json", lambda v: v["points"]["5"].update(fixed_noise_chunk_sha256="c2"), "implementation_received_d"),
            ("smolvla/profile.json", lambda v: v["2"].update(host_s=[.1] * 30), "timing"),
            ("pi05_conditions/conditions.json", lambda v: v.update(status="FAIL"), "conditions_smoke_and_reset"),
            ("smolvla/videos/video_manifest.json", lambda v: v.update(failed=1), "video_files"),
            ("states/state_manifest.json", lambda v: v.update(states=[{"state_sha256": "h"}] * 199), "generated_state_hashes"),
            ("pi05_parity/parity_attempts/p7.attempt00.json", lambda v: v.update(status="failed"), "native_parity_complete"),
            ("pi05_parity/parity_attempts/p9.attempt00.json", lambda v: v.update(action_shape=[5, 6]), "action_units_and_shapes"),
        ]
        for relative, change, check in cases:
            with self.subTest(check=check, file=relative):
                fabricated_run(self.root)
                self.edit(relative, change)
                self.assert_fails(check)

    def test_missing_evidence_fails_rather_than_passes(self):
        (self.root / "pi05_profile/profile.json").unlink()
        self.assert_fails("implementation_received_d")

    def test_report_keeps_parity_counts_by_task(self):
        table = pv.evaluate(self.root)["parity_success_by_task"]["pi05"]
        self.assertEqual(len(table), 40)


class EntryPoint(unittest.TestCase):
    def test_refuses_with_no_release_before_any_work(self):
        script = HERE / "run_preflight_v2.sh"
        with tempfile.TemporaryDirectory() as directory:
            done = subprocess.run(["bash", str(script), directory, directory + "/out"], env={"PATH": "/usr/bin:/bin", "ALBA_GPU_RESERVATION": "R1", "HOME": directory, "PYTHON_FOR_GUARD": sys.executable},
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(done.returncode, 6, done.stdout)
            self.assertFalse((Path(directory) / "out").exists())


if __name__ == "__main__":
    unittest.main()
