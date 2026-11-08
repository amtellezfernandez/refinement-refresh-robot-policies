"""Malformed decoded responses: stdlib stubs only, no policy or simulator imports."""
import unittest
from types import SimpleNamespace

import adapter_pi05 as P
import classify as C


class ResponseContract(unittest.TestCase):
    def test_non_mapping_response_is_instrument_incident_before_actions(self):
        for response in (None, [], 0, "not a response", True):
            with self.subTest(response=response):
                steps, verified = [], []
                env = SimpleNamespace(
                    reset=lambda: None, set_init_state=lambda state: {},
                    sim=SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: [])),
                    step=lambda action: steps.append(action))
                context = SimpleNamespace(
                    row={"planned_run_id": "synthetic", "episode_identity": {}},
                    attempt_number=0, expected_reset=None,
                    record_reset=lambda *args: None, first_query_started=lambda: True)
                adapter = object.__new__(P.Pi05ClientAdapter)
                adapter.np = SimpleNamespace(asarray=lambda value: value)
                adapter.co = SimpleNamespace(
                    WAIT_STEPS=0, MAX_STEPS={"synthetic": 1}, digest=lambda value: "s",
                    input_hash=lambda value: "o", policy_input=lambda *args: {},
                    check_reset=lambda *args: None)
                adapter.client = SimpleNamespace(infer=lambda request: response)
                adapter.verifier = SimpleNamespace(verify=lambda *args: verified.append(args))
                with self.assertRaises(C.InstrumentIncident) as caught:
                    adapter._rollout(context, env, "synthetic", [], {"suite": "synthetic"}, 2, 2)
                self.assertEqual(C.classify_exception(caught.exception, True), C.INSTRUMENT_INCIDENT)
                self.assertEqual(steps, [])
                self.assertEqual(verified, [])


if __name__ == "__main__":
    unittest.main()
