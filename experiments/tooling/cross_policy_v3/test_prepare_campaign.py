"""Synthetic operator preparation: no reserved identities or native execution."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_campaign import prepare


class Preparation(unittest.TestCase):
    def test_empty_workspace_is_read_only_and_reports_both_models(self):
        with tempfile.TemporaryDirectory(prefix="spaces in workspace ") as directory:
            root = Path(directory) / "absent"
            for model in ("pi05", "smolvla"):
                report = prepare(root, model)
                self.assertFalse(root.exists())
                self.assertFalse(report["offline_inputs_complete"])
                self.assertFalse(report["execution_authorized"])
                self.assertTrue(report["issues"])
                self.assertEqual(report["planned_pairs"], 400)
                self.assertEqual(report["planned_attempts"], 800)
                args = report["commands"]["confirmation_after_separate_release"]
                self.assertEqual(args[args.index("--model") + 1], model)
                self.assertIn("--journal", report["commands"]["analysis_after_execution"])
                self.assertIn("--check-inputs", report["commands"]["protocol_e_inputs_only"])
                for command in report["commands"].values():
                    import shlex
                    name = next(k for k, v in report["commands"].items() if v is command)
                    self.assertEqual(shlex.split(report["shell_commands"][name]), command)

    def test_malformed_artifacts_are_actionable_and_never_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pi05"
            root.mkdir()
            for name in ("manifest.json", "schedule.json", "fixtures.json"):
                (root / name).write_text("{broken", encoding="utf-8")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            report = prepare(directory, "pi05")
            self.assertTrue(any("malformed JSON" in i["next_action"] for i in report["issues"]))
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            json.dumps(report, allow_nan=False)

    def test_checkpoint_directory_is_required_for_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in ("pi05", "smolvla"):
                report = prepare(directory, model)
                self.assertTrue(any(i["message"] == "Missing checkpoint directory"
                                    for i in report["issues"]))

    def test_pi05_startup_gate_belongs_to_the_future_owned_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pi05"
            (root / "checkpoint").mkdir(parents=True)
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            with patch("prepare_campaign.Q.summary", return_value={"count_gate": "UNVERIFIED"}):
                report = prepare(directory, "pi05")
            command = report["commands"]["confirmation_after_separate_release"]
            self.assertEqual(command[command.index("--startup-gate") + 1],
                             str(root / "server/startup_gate.json"))
            self.assertFalse(any("Missing " + str(root / "server") in i["message"]
                                 for i in report["issues"]))
            self.assertTrue(any(i["message"] == "count_gate: UNVERIFIED"
                                for i in report["issues"]))
            self.assertFalse(report["offline_inputs_complete"])
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))

    def test_nonobject_json_is_reported_without_reading_native_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pi05"
            root.mkdir()
            for value in ([], None, True, 3, "not an object"):
                with self.subTest(value=value):
                    for name in ("manifest.json", "schedule.json", "fixtures.json"):
                        (root / name).write_text(json.dumps(value), encoding="utf-8")
                    before = {p.name: p.read_bytes() for p in root.iterdir()}
                    report = prepare(directory, "pi05")
                    self.assertEqual(sum("artifact must be a JSON object" in i["message"]
                                         for i in report["issues"]), 3)
                    self.assertFalse(report["offline_inputs_complete"])
                    self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})


    def test_python38_syntax(self):
        source = Path(__file__).with_name("prepare_campaign.py").read_text()
        ast.parse(source, feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
