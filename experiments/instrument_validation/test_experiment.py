"""Checks experiment integrity and epistemic denominators, not policy quality."""
import json
from pathlib import Path
import unittest

import run_experiment as experiment


class ExperimentTests(unittest.TestCase):
    def test_preserved_sources_and_frozen_protocol(self):
        protocol = experiment.load_protocol()
        self.assertEqual(len(protocol["sources"]), 4)
        self.assertEqual(len(protocol["case_manifest"]), 17)

    def test_unobservable_queries_are_not_fabricated_measurements(self):
        source = experiment.LEDGER_ROOT / "results/infrastructure_v2/h10.json"
        original = source.read_text()
        for case in ("preprocessing_coverage_query", "rendering_coverage_query"):
            self.assertEqual(experiment.mutate(original, case), original)

    def test_unknowns_excluded_from_detection_denominator(self):
        report = json.loads((Path(__file__).parent / "trials.json").read_text())
        summary = experiment.summarize(report)
        for layer in summary["layers"]:
            self.assertEqual(layer["observable_defects"], 52)
            self.assertEqual(layer["controls"], 4)
            self.assertEqual(layer["unknown_scenarios"], 12)
            self.assertEqual(layer["detected"] + layer["not_detected"] + layer["validator_errors"], 52)


if __name__ == "__main__":
    unittest.main()
