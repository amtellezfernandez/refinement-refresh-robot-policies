"""Synthetic contract/fault tests; these are not robot performance evidence."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validate_trace import SCHEMA_PATH, schema_check, validate, validate_graph

ROOT = Path(__file__).resolve().parent


def fixture(name="deployment_trace.json"):
    return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))


class TraceContractTests(unittest.TestCase):
    def test_hybrid_fixture_is_complete_only_for_declared_resources(self):
        report = validate_graph(fixture("hybrid_system_trace.json"))
        self.assertTrue(report["complete_for_declared_resources"])
        self.assertEqual(report["quantitative_resource_rows"], 8)

    def test_deployment_allowed_and_unknown_not_zero(self):
        graph = fixture()
        report = validate_graph(graph)
        self.assertTrue(report["structurally_valid"])
        self.assertFalse(report["complete_for_declared_resources"])
        self.assertEqual(report["incomplete_resources"], [
            {"event": "predict", "resource": "elapsed_time", "reason": "unknown"}])
        self.assertEqual(len(report["not_applicable_resources"]), 1)

    def test_measured_zero_is_complete(self):
        graph = fixture()
        graph["events"][0]["resources"][1].update(status="measured", amount=0)
        self.assertTrue(validate_graph(graph)["complete_for_declared_resources"])

    def test_missing_resource_remains_incomplete(self):
        graph = fixture()
        graph["events"][0]["resources"].pop(1)
        report = validate_graph(graph)
        self.assertEqual(report["incomplete_resources"][0]["reason"], "missing")

    def test_status_and_amount_contract(self):
        cases = [("measured", None), ("derived", None), ("unknown", 0),
                 ("not_applicable", 0), ("reported", 1), ("measured", -1),
                 ("measured", True)]
        for status, amount in cases:
            with self.subTest(status=status, amount=amount):
                graph = fixture()
                graph["events"][0]["resources"][1].update(status=status, amount=amount)
                with self.assertRaises(ValueError):
                    validate_graph(graph)
        graph = fixture()
        del graph["events"][0]["resources"][0]["derivation"]
        with self.assertRaises(ValueError):
            validate_graph(graph)

    def test_nonfinite_values_rejected_anywhere(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for target in ("amount", "metadata"):
                with self.subTest(value=value, target=target):
                    graph = fixture()
                    if target == "amount":
                        graph["events"][0]["resources"][0]["amount"] = value
                    else:
                        graph["metadata"]["number"] = value
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        validate_graph(graph)

    def test_required_units_boundaries_and_operator(self):
        for field in ("unit", "boundary", "aggregation", "name"):
            graph = fixture()
            del graph["events"][0]["resources"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_graph(graph)
        graph = fixture()
        graph["events"][0]["operator"] = "MAGIC"
        with self.assertRaises(ValueError):
            validate_graph(graph)
        graph = fixture()
        graph["events"][0]["required_resources"] = []
        with self.assertRaises(ValueError):
            validate_graph(graph)

    def test_identifiers_and_edges(self):
        for mutation in ("state", "event", "unresolved", "root"):
            graph = fixture()
            if mutation == "state":
                graph["states"].append(copy.deepcopy(graph["states"][0]))
            elif mutation == "event":
                graph["events"].append(copy.deepcopy(graph["events"][0]))
            elif mutation == "unresolved":
                graph["events"][0]["inputs"].append("missing_state")
            else:
                graph["pretrained_roots"] = ["missing_root"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_graph(graph)

    def test_multiple_producers(self):
        graph = fixture()
        event = copy.deepcopy(graph["events"][0])
        event["id"] = "second_predict"
        graph["events"].append(event)
        with self.assertRaisesRegex(ValueError, "multiple producers"):
            validate_graph(graph)

    def test_lineage_cycle(self):
        graph = fixture()
        graph["events"][0]["inputs"].append("sim_1")
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_graph(graph)

    def test_in_place_environment_update(self):
        graph = fixture()
        graph["events"][1]["outputs"] = ["sim_0"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_graph(graph)

    def test_operator_negative_fixture(self):
        with self.assertRaisesRegex(ValueError, "UPDATE"):
            validate_graph(fixture("invalid_update_trace.json"))

    def test_simulation_is_not_physical_execution(self):
        graph = fixture()
        graph["events"][1]["changes_physical_world"] = True
        with self.assertRaisesRegex(ValueError, "physical-world"):
            validate_graph(graph)

    def test_duplicate_resource(self):
        graph = fixture()
        graph["events"][0]["resources"].append(copy.deepcopy(graph["events"][0]["resources"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate resource"):
            validate_graph(graph)

    def test_omitted_event_not_detectable_without_external_counter(self):
        graph = fixture()
        graph["events"] = graph["events"][1:]
        self.assertTrue(validate_graph(graph)["structurally_valid"])
        # A producer-free action can be an inherited input. Independent execution
        # counters are needed to determine whether its producer was omitted.

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"spec_version":"0.2","spec_version":"0.2"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate(path)

    def test_cli_separates_structural_failure_and_incompleteness(self):
        for name, extra, expected in [
            ("deployment_trace.json", [], 0),
            ("deployment_trace.json", ["--require-complete"], 2),
            ("invalid_update_trace.json", [], 1),
        ]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "validate_trace.py"),
                 str(ROOT / "examples" / name), *extra],
                capture_output=True, text=True, check=False)
            with self.subTest(name=name, extra=extra):
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_schema_vocabulary_does_not_silently_expand(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        allowed = {"$schema", "$id", "$defs", "$ref", "title", "description",
                   "type", "required", "properties", "additionalProperties",
                   "const", "enum", "items", "minItems", "uniqueItems",
                   "minLength", "minimum", "oneOf"}
        def visit(node):
            self.assertFalse(set(node) - allowed)
            for keyword in ("properties", "$defs"):
                for child in node.get(keyword, {}).values():
                    visit(child)
            for keyword in ("items", "additionalProperties"):
                if isinstance(node.get(keyword), dict):
                    visit(node[keyword])
            for child in node.get("oneOf", []):
                visit(child)
        visit(schema)

    def test_jsonschema_structural_parity(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema optional; install it for independent schema parity")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        external = Draft202012Validator(schema)
        graphs = [fixture(), fixture("hybrid_system_trace.json"),
                  fixture("invalid_update_trace.json")]
        for field, values in {
            "amount": [None, -1, 0, True, "1"],
            "status": ["measured", "derived", "unknown", "not_applicable", "reported"],
            "unit": ["", "s", None],
            "boundary": ["", "policy", 0],
            "aggregation": ["sum", "max", "snapshot", "interval", "average"],
        }.items():
            for value in values:
                graph = fixture()
                graph["events"][0]["resources"][0][field] = value
                graphs.append(graph)
        for index, graph in enumerate(graphs):
            try:
                schema_check(graph, schema, schema)
                local_valid = True
            except ValueError:
                local_valid = False
            with self.subTest(index=index):
                self.assertEqual(local_valid, external.is_valid(graph))


if __name__ == "__main__":
    unittest.main()
