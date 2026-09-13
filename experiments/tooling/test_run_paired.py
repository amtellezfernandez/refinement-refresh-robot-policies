"""Offline integrity tests for the runner; not substitutes for a GPU smoke test."""
import copy
import cProfile
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from run_paired import (
    CONDITIONS, Counters, array_record, batch_one_success, development_schedule,
    CHECKPOINT_REVISION, file_sha, json_line, keyed_seed, observation_field_shas,
    observation_sha, parity_report,
    select_schedule, sha, verify_checkpoint_marker, verify_source_tree,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_evaluation_plan import make_plan


class RunnerTests(unittest.TestCase):
    def test_full_manifest_and_balanced_upfront_caps(self):
        plan = make_plan()
        self.assertEqual(len(select_schedule(plan)), 320)
        for cap in (16, 32, 48, 64, 80):
            rows = select_schedule(plan, cap)
            self.assertEqual(len(rows), cap * 4)
            by_task = {}
            for row in rows:
                e = row["episode_identity"]
                task = (e["suite"], e["task_id"])
                by_task.setdefault(task, set()).add((e["initial_state_id"], e["policy_seed"]))
            self.assertEqual(len(by_task), 8)
            self.assertTrue(all(len(identities) == cap // 8 for identities in by_task.values()))
            for identities in by_task.values():
                for state in {state for state, seed in identities}:
                    self.assertEqual({seed for other, seed in identities if other == state},
                                     {2000, 2001})

    def test_partial_or_out_of_range_caps_refused(self):
        for cap in (-1, 0, 1, 8, 17, 81, True):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                select_schedule(make_plan(), cap)

    def test_corrupted_manifest_and_incomplete_blocks_refused(self):
        plan = make_plan()
        plan["schedule"][0]["h"] = 99
        with self.assertRaisesRegex(ValueError, "digest"):
            select_schedule(plan)
        plan["scheduling"]["schedule_sha256"] = sha(plan["schedule"])
        with self.assertRaisesRegex(ValueError, "four-condition"):
            select_schedule(plan)
        plan = make_plan()
        plan["schedule"][3], plan["schedule"][4] = plan["schedule"][4], plan["schedule"][3]
        plan["scheduling"]["schedule_sha256"] = sha(plan["schedule"])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            select_schedule(plan)

    def test_development_excludes_reserved_ids(self):
        rows = development_schedule()
        self.assertEqual(len(rows), 4)
        self.assertEqual({(row["h"], row["denoising_steps"]) for row in rows}, CONDITIONS)
        self.assertEqual({row["episode_identity"]["task_id"] for row in rows}, {0})
        self.assertEqual({row["episode_identity"]["initial_state_id"] for row in rows}, {0})

    def test_noise_keys_independent_of_call_count_and_condition(self):
        identity = development_schedule()[0]["episode_identity"]
        key = keyed_seed(identity, 10)
        self.assertEqual(key, keyed_seed(dict(reversed(list(identity.items()))), 10))
        self.assertNotEqual(key, keyed_seed(identity, 11))
        other = dict(identity, policy_seed=1001)
        self.assertNotEqual(key, keyed_seed(other, 10))
        self.assertNotEqual(key, keyed_seed(identity, 10, "warmup"))
        # h and d never enter the identity: tick-10 keys match even when one
        # condition has made ten predictions and another has made only two.
        self.assertEqual(key, keyed_seed(development_schedule()[3]["episode_identity"], 10))

    def test_array_hash_shape_dtype_and_finiteness(self):
        import numpy as np
        a = array_record(np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(a, array_record(np.array([1.0, 2.0], dtype=np.float32)))
        self.assertNotEqual(a["sha256"], array_record(np.array([[1.0, 2.0]], dtype=np.float32))["sha256"])
        self.assertNotEqual(a["sha256"], array_record(np.array([1.0, 2.0], dtype=np.float64))["sha256"])
        with self.assertRaises(ValueError):
            array_record([float("nan")])

    def test_observation_audit_hashes_are_compact_and_nested(self):
        import numpy as np
        observation = {"image": np.array([[1, 2]], dtype=np.uint8),
                       "state": {"q": np.array([0.5], dtype=np.float32)}}
        fields = observation_field_shas(observation)
        self.assertEqual(set(fields), {"image", "state"})
        self.assertEqual(set(fields["state"]), {"q"})
        self.assertEqual(len(fields["image"]), 64)
        self.assertEqual(len(fields["state"]["q"]), 64)
        self.assertEqual(len(observation_sha(observation)), 64)

    def test_vector_success_interfaces(self):
        self.assertTrue(batch_one_success({"is_success": [True]}))
        self.assertFalse(batch_one_success({"is_success": False}))
        self.assertTrue(batch_one_success({"final_info": {"is_success": [True]}}))
        self.assertTrue(batch_one_success({"final_info": [{"is_success": True}]}))
        with self.assertRaises(ValueError):
            batch_one_success({})
        with self.assertRaises(ValueError):
            batch_one_success({"is_success": [True, False]})

    def test_actual_hook_boundaries_and_restore(self):
        class FakeEvent:
            def record(self):
                pass
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return 1.25
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            synchronize=lambda: None, Event=lambda **kwargs: FakeEvent()))
        class FakeExpert:
            def embed_image(self, image):
                return image
            def forward(self, **kwargs):
                return kwargs["inputs_embeds"]
        class FakeModel:
            def __init__(self):
                self.vlm_with_expert = FakeExpert()
            def denoise_step(self):
                return self.vlm_with_expert.forward(inputs_embeds=[None, "suffix"])
        class FakePolicy:
            def __init__(self):
                self.model = FakeModel()
            def _get_action_chunk(self):
                self.model.vlm_with_expert.embed_image("left")
                self.model.vlm_with_expert.embed_image("wrist")
                self.model.vlm_with_expert.forward(inputs_embeds=["prefix", None])
                for _ in range(5):
                    self.model.denoise_step()
                return "actions"
        policy = FakePolicy()
        original = policy._get_action_chunk
        with patch.dict(sys.modules, {"torch": fake_torch}):
            counters = Counters(policy)
            self.assertEqual(policy._get_action_chunk(), "actions")
            self.assertEqual(counters.counts, {
                "generations": 1, "denoise_steps": 5, "vision_encoder_calls": 2,
                "prefill_calls": 1, "action_expert_calls": 5})
            self.assertEqual(counters.cuda_ms, [1.25])
            counters.restore()
            self.assertEqual(policy._get_action_chunk, original)
            self.assertNotIn("_get_action_chunk", policy.__dict__)
            self.assertNotIn("denoise_step", policy.model.__dict__)
            counters.restore()  # Safe after a failure path has already restored.
            policy._get_action_chunk()
            self.assertEqual(counters.counts["generations"], 1)

    def test_json_journal_contains_complete_serializable_lines(self):
        stream = io.StringIO()
        json_line(stream, {"kind": "step", "value": 1})
        json_line(stream, {"kind": "episode_completed", "success": False})
        self.assertEqual(len([json.loads(line) for line in stream.getvalue().splitlines()]), 2)
        with self.assertRaises(ValueError):
            json_line(stream, {"value": float("nan")})

    def test_checkpoint_marker_requires_exact_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_checkpoint_marker(path)
            (path / ".robopad_revision").write_text("wrong\n")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                verify_checkpoint_marker(path)
            (path / ".robopad_revision").write_text(CHECKPOINT_REVISION + "\n")
            verify_checkpoint_marker(path)

    def test_pinned_source_content_checked_not_only_revision_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "example.py").write_text("pinned source\n")
            expected = {"example.py": file_sha(path / "example.py")}
            with patch("run_paired.PINNED_SOURCE_HASHES", expected):
                self.assertEqual(verify_source_tree(path), expected)
                (path / "example.py").write_text("modified source\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_source_tree(path)

    def test_downloaded_pinned_source_matches_recorded_files(self):
        root = Path(__file__).resolve().parents[2] / "build" / "runtime" / (
            "lerobot-b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot")
        if not root.is_dir():
            self.skipTest("pinned source checkout is not local")
        self.assertTrue(verify_source_tree(root))

    def test_upstream_parity_requires_actions_starts_steps_and_outcome(self):
        import numpy as np
        actions = np.array([[1, 2], [3, 4]], dtype=np.float32)
        initial = array_record(np.array([0.1, 0.2], dtype=np.float64))
        reference = {"run_id": "synthetic", "episode_identity": {}, "h": 1,
                     "denoising_steps": 10, "initial_state": initial,
                     "initial_observation_sha256": "obs", "steps": 2, "success": True,
                     "action_stream_sha256": sha([array_record(a)["sha256"] for a in actions])}
        report = parity_report(reference, initial, "obs", actions, True)
        self.assertEqual(report["status"], "passed")
        altered = actions.copy()
        altered[0, 0] += 1
        self.assertEqual(parity_report(reference, initial, "obs", altered, True)["status"], "failed")
        self.assertEqual(parity_report(reference, initial, "other", actions, True)["status"], "failed")
        self.assertEqual(parity_report(reference, initial, "obs", actions, False)["status"], "failed")


class DevelopmentGateTests(unittest.TestCase):
    def test_original_code_profile_avoids_same_name_collision(self):
        from development_gates import profiler_counts
        class ModelA:
            def sample_actions(self):
                pass
        class ModelB:
            def sample_actions(self):
                pass
        a, b = ModelA(), ModelB()
        profiler = cProfile.Profile()
        profiler.enable()
        a.sample_actions()
        for _ in range(3):
            b.sample_actions()
        profiler.disable()
        result = profiler_counts(profiler, {"a": a.sample_actions, "b": b.sample_actions})
        self.assertEqual(result["a"]["total_calls"], 1)
        self.assertEqual(result["b"]["total_calls"], 3)

    def test_camera_controls_change_only_images_and_preserve_input(self):
        import numpy as np
        from development_gates import image_variants
        original = {"pixels": {"left": np.ones((1, 3, 3, 3), dtype=np.uint8),
                               "wrist": np.full((1, 3, 3, 3), 7, dtype=np.uint8)},
                    "robot_state": {"position": [1.0, 2.0, 3.0]}}
        variants, keys = image_variants(original)
        self.assertEqual(keys, ["left", "wrist"])
        self.assertTrue(np.all(variants["camera_swap"]["pixels"]["left"] == 7))
        self.assertTrue(np.all(variants["zero_images"]["pixels"]["wrist"] == 0))
        self.assertTrue(np.all(original["pixels"]["left"] == 1))
        for variant in variants.values():
            self.assertEqual(variant["robot_state"], original["robot_state"])
        original["pixels"]["wrist"][:] = 1
        with self.assertRaisesRegex(ValueError, "identical"):
            image_variants(original)

    def test_full_chunk_parity_catches_late_action_change(self):
        import numpy as np
        from development_gates import compare_arrays
        reference = np.zeros((1, 50, 7), dtype=np.float32)
        altered = reference.copy()
        altered[0, 49, 6] = 0.001
        self.assertFalse(compare_arrays(reference, altered)["allclose"])
        self.assertTrue(compare_arrays(reference, reference)["exact_equal"])
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_arrays(reference, reference + np.nan)


if __name__ == "__main__":
    unittest.main()
