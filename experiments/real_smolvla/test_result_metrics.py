"""Regression tests for scientific summaries, without GPU/runtime dependencies."""
import copy
import json
import unittest
from pathlib import Path

from result_metrics import summarize


def fixture():
    return {
        "per_task": [{"task_group": "spatial", "task_id": 0,
                      "metrics": {"successes": [True, False, True]}}],
        "per_group": {"spatial": {"n_episodes": 3, "pc_success": 200 / 3}},
        "overall": {"n_episodes": 3, "pc_success": 200 / 3, "eval_s": 3.},
        "robopad_ledger": {"select_action_calls": 6, "neural_chunk_calls": 3,
                           "neural_chunk_seconds": [0.1, 0.2, 0.3]},
    }


class ResultTests(unittest.TestCase):
    def test_equal_calls_do_not_determine_host_interval_sum(self):
        first = summarize(fixture())
        second_data = fixture()
        second_data["robopad_ledger"]["neural_chunk_seconds"] = [0.4, 0.8, 1.2]
        second = summarize(second_data)
        self.assertEqual(first["neural_chunks"], second["neural_chunks"])
        self.assertAlmostEqual(first["legacy_inference_host_seconds"], 0.6)
        self.assertAlmostEqual(second["legacy_inference_host_seconds"], 2.4)

    def test_v2_does_not_supply_legacy_host_interval(self):
        path = Path(__file__).parent / "results/infrastructure_v2/h10.json"
        self.assertIsNone(summarize(json.loads(path.read_text()))["legacy_inference_host_seconds"])

    def test_all_episodes_counted(self):
        row = summarize(fixture())
        self.assertEqual((row["successes"], row["episodes"]), (2, 3))
        self.assertEqual(row["control_to_neural"], 2.)
        self.assertEqual(row["pairing_evidence"], "unverifiable")

    def test_completed_results_and_consistent_outcomes_required(self):
        faults = [
            lambda d: d.pop("overall"),
            lambda d: d["overall"].update(n_episodes=2),
            lambda d: d["overall"].update(pc_success=100.),
            lambda d: d["per_group"]["spatial"].update(n_episodes=2),
            lambda d: d["per_group"]["spatial"].update(pc_success=100.),
            lambda d: d["per_task"][0]["metrics"].update(successes=["false", False, True]),
            lambda d: d["per_task"].append(copy.deepcopy(d["per_task"][0])),
            lambda d: d["robopad_ledger"]["neural_chunk_seconds"].pop(),
            lambda d: d["robopad_ledger"]["neural_chunk_seconds"].append(float("nan")),
            lambda d: d["robopad_ledger"].update(neural_chunk_calls=0),
            lambda d: d["robopad_ledger"].update(schema_version=3),
        ]
        for fault in faults:
            with self.subTest(fault=fault):
                data = fixture()
                fault(data)
                with self.assertRaises(ValueError):
                    summarize(data)


if __name__ == "__main__":
    unittest.main()
