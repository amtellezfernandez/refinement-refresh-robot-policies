"""CPU-only regression tests for post-freeze replay execution safeguards.

These use constructed data, not archived robot actions. No simulation or GPU is run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import platform
import sys
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import governance as gov
import replay as R
import run_protocol as P


class Population(unittest.TestCase):
    def make_population(self):
        # Same frozen cell counts as the primary outcome table, but NOT original
        # identities/labels; synthetic ordering deliberately differs from the archive.
        counts = {
            ("libero_goal", 5): (16, 14), ("libero_goal", 6): (18, 15),
            ("libero_goal", 7): (19, 18), ("libero_goal", 8): (18, 14),
            ("libero_object", 5): (20, 20), ("libero_object", 6): (20, 17),
            ("libero_object", 7): (10, 7), ("libero_object", 8): (18, 17),
        }
        conditions = [(1, 2), (1, 5), (1, 10), (2, 2), (2, 5), (5, 2), (5, 5)]
        bundles = []
        for (suite, task), (s2, s5) in sorted(counts.items()):
            for state in range(20, 30):
                for seed in (3000, 3001):
                    pos = (state - 20) * 2 + (seed - 3000)
                    for h, d in conditions:
                        nsuccess = s2 if (h, d) == (2, 2) else s5 if (h, d) == (2, 5) else 10
                        key = f"confirmation.{suite}.task{task:03d}.state{state:03d}.seed{seed}.h{h}_d{d}"
                        bundles.append({
                            "provenance": "archived_paper_evidence",
                            "episode_key": key, "suite": suite, "task_id": task,
                            "condition": f"h{h}_d{d}",
                            "reset_source": {"kind": "official", "initial_state_id": state},
                            "recorded_initial_state": {"sha256": hashlib.sha256(
                                f"{suite}:{task}:{state}".encode()).hexdigest()},
                            "label": {"success": pos < nsuccess},
                        })
        strata = defaultdict(list)
        for b in bundles:
            if b["condition"] in gov.PRIMARY:
                status = "success" if b["label"]["success"] else "failure"
                strata[(b["suite"], b["task_id"], b["condition"], status)].append(b["episode_key"])
        sampled = []
        table = []
        for k in sorted(strata):
            ordered = sorted(strata[k], key=lambda p: hashlib.sha256(
                (gov.SAMPLE_SALT + p).encode()).hexdigest())
            selected = ordered[:2]
            sampled.extend(selected)
            table.append({"stratum": list(k), "frame": len(strata[k]), "sampled": selected})
        self.assertEqual(len(bundles), 1120)
        self.assertEqual(len(strata), 29)
        self.assertEqual(len(sampled), 57)
        manifest = {
            "population": 1120, "primary_frame": 320, "strata_nonempty": 29,
            "stage1_sample_size": 57, "salt": gov.SAMPLE_SALT, "per_stratum": 2,
            "stage1_sample": sampled, "stage1_sample_sha256": gov.sha_json(sorted(sampled)),
            "strata": table,
        }
        return bundles, manifest

    def test_exact_population_and_frozen_selection_pass_on_constructed_data(self):
        bundles, manifest = self.make_population()
        with patch.object(gov, "SAMPLE_SELECTION_SHA256", manifest["stage1_sample_sha256"]):
            self.assertEqual(gov.verify_population(bundles, manifest)["resets"], 80)

    def test_dropped_bundle_refused(self):
        bundles, manifest = self.make_population()
        with self.assertRaisesRegex(gov.ReplayRefused, "1,120"):
            gov.verify_population(bundles[:-1], manifest)

    def test_altered_initial_state_refused(self):
        bundles, manifest = self.make_population()
        bundles[-1]["recorded_initial_state"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(gov.ReplayRefused, "paired initial states"):
            gov.verify_population(bundles, manifest)

    def test_duplicate_sample_refused(self):
        bundles, manifest = self.make_population()
        manifest["stage1_sample"][1] = manifest["stage1_sample"][0]
        with self.assertRaisesRegex(gov.ReplayRefused, "57 distinct"):
            gov.verify_population(bundles, manifest)

    def test_changed_strata_or_sample_cannot_pass_merely_by_rehashing_manifest(self):
        bundles, manifest = self.make_population()
        manifest["stage1_sample"][0], manifest["stage1_sample"][2] = (
            manifest["stage1_sample"][2], manifest["stage1_sample"][0])
        manifest["stage1_sample_sha256"] = gov.sha_json(sorted(manifest["stage1_sample"]))
        with patch.object(gov, "SAMPLE_SELECTION_SHA256", manifest["stage1_sample_sha256"]):
            with self.assertRaisesRegex(gov.ReplayRefused, "hash-stratified draw"):
                gov.verify_population(bundles, manifest)

    def test_primary_keys_exact_320(self):
        self.assertEqual(len(gov.expected_primary_keys()), 320)
        self.assertIn("confirmation.libero_goal.task005.state020.seed3000.h2_d2",
                      gov.expected_primary_keys())


class Release(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.root.mkdir(exist_ok=True)
        self.replay_dir = self.root / "replay"
        self.proto_dir = self.root / "protocol"
        self.replay_dir.mkdir()
        self.proto_dir.mkdir()
        for n in gov.EXECUTION_FILES:
            (self.replay_dir / n).write_text(n + "\n")
        for n in gov.AMENDMENT_FILES:
            (self.proto_dir / n).write_text(n + "\n")
        self.proto = self.proto_dir / "PROTOCOL.md"
        self.proto.write_text("Frozen study protocol\n")
        self.bundles = self.root / "bundles.jsonl"
        self.bundles.write_text("synthetic bundles\n")
        self.sample = self.root / "sample.json"
        self.sample.write_text("{}\n")
        self.release_path = self.root / "release.json"
        self.release = {
            "approved_by": gov.APPROVER, "decision": "APPROVED", "scope": [0],
            "protocol_sha256": gov.sha_file(self.proto),
            "bundles_sha256": gov.sha_file(self.bundles),
            "sample_manifest_sha256": gov.sha_file(self.sample),
            "sample_selection_sha256": gov.SAMPLE_SELECTION_SHA256,
            "code_sha256": {n: gov.sha_file(self.replay_dir / n)
                            for n in gov.EXECUTION_FILES},
            "amendments_sha256": {n: gov.sha_file(self.proto_dir / n)
                                  for n in gov.AMENDMENT_FILES},
            "render": True,
        }
        self.patchers = [
            patch.object(gov, "HERE", self.replay_dir),
            patch.object(gov, "ROOT", self.root),
            patch.object(gov, "PROTOCOL_SHA256", gov.sha_file(self.proto)),
            patch.object(gov, "BUNDLES_SHA256", gov.sha_file(self.bundles)),
            patch.object(platform, "system", return_value="Darwin"),
            patch.object(platform, "machine", return_value="arm64"),
        ]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)
        self.save()

    def save(self):
        self.release_path.write_text(json.dumps(self.release))

    def check(self, through=0):
        return gov._strict_release(self.release_path, self.proto,
                                   self.bundles, self.sample, through)

    def test_stage0_only_release_allows_only_stage0(self):
        self.assertEqual(self.check(0)[0]["scope"], [0])
        with self.assertRaisesRegex(gov.ReplayRefused, "approval scope"):
            self.check(1)

    def test_stage1_requires_contiguous_scope(self):
        self.release["scope"] = [0, 1]
        self.save()
        self.assertEqual(self.check(1)[0]["scope"], [0, 1])
        for scope in ([0, 2], [], ["0"], [1], [0, 1, 1]):
            self.release["scope"] = scope
            self.save()
            with self.subTest(scope=scope), self.assertRaises(gov.ReplayRefused):
                self.check(1)

    def test_exact_approver_not_merely_nonempty_string(self):
        self.release["approved_by"] = "test"
        self.save()
        with self.assertRaisesRegex(gov.ReplayRefused, "author"):
            self.check()

    def test_code_change_invalidates_release(self):
        (self.replay_dir / "run_protocol.py").write_text("changed after approval\n")
        with self.assertRaisesRegex(gov.ReplayRefused, "code has drifted"):
            self.check()

    def test_amendment_change_invalidates_release(self):
        (self.proto_dir / "PROTOCOL_AMENDMENT_02.md").write_text("changed after approval\n")
        with self.assertRaisesRegex(gov.ReplayRefused, "amendment"):
            self.check()

    def test_sample_bytes_change_invalidates_release(self):
        self.sample.write_text(" { } \n")
        with self.assertRaisesRegex(gov.ReplayRefused, "manifest"):
            self.check()

    def test_bundle_bytes_change_invalidates_release(self):
        self.bundles.write_text("altered bundles\n")
        with self.assertRaisesRegex(gov.ReplayRefused, "bundle"):
            self.check()

    def test_nonrendering_or_other_host_refused(self):
        self.release["render"] = False
        self.save()
        with self.assertRaisesRegex(gov.ReplayRefused, "render"):
            self.check()
        self.release["render"] = True
        self.save()
        with patch.object(platform, "machine", return_value="x86_64"):
            with self.assertRaisesRegex(gov.ReplayRefused, "arm64"):
                self.check()


class Orchestrator(unittest.TestCase):
    def test_direct_archived_replay_refused(self):
        b = [{"provenance": "archived_paper_evidence", "episode_key":
              "confirmation.libero_goal.task005.state020.seed3000.h2_d2"}]
        refusal = R.gate(b, None, None)
        self.assertIn("staged orchestrator", refusal)

    def test_synthetic_fixture_drill_not_treated_as_paper_evidence(self):
        b = [{"provenance": "archived_paper_evidence", "episode_key": "fixture.goal.demo0",
              "fixture_source": {"kind": "demo"}, "reset_source": {"kind": "vector"}}]
        self.assertFalse(gov.is_real_archive(b))

    def run_with_mocked_physics(self, through, s0, s1=None, s2=None):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            bundles = d / "bundles.jsonl"
            sample = d / "sample.json"
            bundles.write_text('{"provenance": "synthetic_fixture", "episode_key": "fixture.0"}\n')
            sample.write_text('{"stage1_sample": ["fixture.0"]}\n')
            argv = ["run_protocol.py", str(bundles), str(sample), str(d / "output"),
                    "--release", str(d / "release.json"), "--protocol",
                    str(d / "protocol.md"), "--through", str(through)]
            results = iter(x for x in (s1, s2) if x is not None)
            with patch.object(sys, "argv", argv), patch.object(P.R, "gate", return_value=None), \
                 patch.object(P, "stage0", return_value={"verdict": s0, "exact": 1, "n_task_states": 1}), \
                 patch.object(P, "episode_stage", side_effect=lambda *a, **k: {
                     "verdict": next(results), "counts_by_condition": {}}):
                with self.assertRaises(SystemExit) as cm:
                    P.main()
                return cm.exception.code

    def test_failed_stage0_is_nonzero_even_when_through0(self):
        self.assertEqual(self.run_with_mocked_physics(0, "FAIL_RESET"), 6)

    def test_failed_stage1_is_nonzero_even_when_through1(self):
        self.assertEqual(self.run_with_mocked_physics(1, "PASS_EXACT", "DISAGREEMENT"), 6)

    def test_failed_stage2_is_nonzero(self):
        self.assertEqual(self.run_with_mocked_physics(2, "PASS_EXACT",
                                                      "PASS_EXACT", "DISAGREEMENT"), 6)


if __name__ == "__main__":
    unittest.main()
