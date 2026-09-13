"""Synthetic fault injections test validation, not scientific performance."""
import copy
import unittest
import numpy as np
from ledger_arrays import array_record
from validate_ledger import validate, validate_completed, compare
import random
import importlib.util


def fixture():
    action = array_record(np.array([[0.1, 0.2]], dtype=np.float32))
    e = {"identity": ["spatial", 0, 0, 42],
         "seeds": {k: 42 for k in ("python", "numpy", "torch", "libero")},
         "initial_state": array_record(np.array([1., 2.])), "select_calls": 2,
         "complete": True, "max_steps": 5, "success": True,
         "producer_steps": [{"step": i, "action": copy.deepcopy(action)} for i in range(2)]}
    e["environment_steps"] = copy.deepcopy(e["producer_steps"])
    for i, row in enumerate(e["environment_steps"]):
        row.update(reward=float(i), terminated=bool(i), truncated=False, success=bool(i))
    return {"schema_version": 3, "complete": True,
            "contract": {"config": {"policy": {}}, "checkpoint": {"weights": "test"}, "software": {"torch": "test"}, "source": {"ledger": "test"}},
            "h": 1, "denoising_steps": 10, "warmup_passes": 1,
            "episodes": [e], "chunk_calls": {"warmup": 1, "measured": 2},
            "sampler_calls": {"warmup": 1, "measured": 2},
            "inferences": [{"phase": phase, "episode": e["identity"], "step": step, "cuda_ms": 1.}
                           for phase, step in [("warmup", 0), ("measured", 0), ("measured", 1)]]}


class Tests(unittest.TestCase):
    def test_valid_and_replay(self):
        validate(fixture())
        compare(fixture(), fixture(), replay=True)

    def test_fault_injections(self):
        faults = [
            lambda d: d["episodes"][0]["environment_steps"].pop(),
            lambda d: d["episodes"].append(copy.deepcopy(d["episodes"][0])),
            lambda d: d["sampler_calls"].update(measured=3),
            lambda d: d["inferences"][1].update(step=9),
            lambda d: d["inferences"][1].update(cuda_ms=float("nan")),
            lambda d: d["episodes"][0]["producer_steps"][0]["action"].update(sha256="bad"),
            lambda d: d["episodes"][0]["seeds"].update(numpy=3),
            lambda d: d.update(warmup_passes=0),
            lambda d: d.update(h=True),
            lambda d: d.update(denoising_steps=0),
            lambda d: d["inferences"][0].update(step=1),
            lambda d: d["inferences"].reverse(),
            lambda d: d.update(suite="different_suite"),
            lambda d: d["chunk_calls"].update(warmup=True),
            lambda d: d["episodes"][0].update(select_calls=2.),
            lambda d: d["episodes"][0]["producer_steps"][0].update(step=False),
            lambda d: d["inferences"][0].update(step=False),
            lambda d: d.update(complete=False),
            lambda d: d["episodes"][0].update(success=False),
            lambda d: d["episodes"][0]["environment_steps"][-1].update(terminated=False),
        ]
        for fault in faults:
            with self.subTest(fault=fault):
                d = fixture()
                fault(d)
                with self.assertRaises(ValueError):
                    validate(d)

    def test_pairing_vs_replay(self):
        d = fixture()
        for side in ("producer_steps", "environment_steps"):
            d["episodes"][0][side][0]["action"] = array_record(np.array([[8., 9.]], dtype=np.float32))
        compare(fixture(), d)  # same initial condition, different behavior allowed
        with self.assertRaises(ValueError):
            compare(fixture(), d, replay=True)
        d["episodes"][0]["initial_state"] = array_record(np.array([3., 4.]))
        with self.assertRaises(ValueError):
            compare(fixture(), d)

    def test_different_refresh_pairing(self):
        d = fixture()
        d["h"] = 5
        d["inferences"].pop()
        d["chunk_calls"]["measured"] = 1
        d["sampler_calls"]["measured"] = 1
        compare(fixture(), d)
        with self.assertRaises(ValueError):
            compare(fixture(), d, replay=True)

    def test_rng_isolation(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Torch runtime unavailable; offline ledger checks still run")
        import torch
        from robopad_ledger import isolated_rng, seed_all
        def draw():
            return random.random(), np.random.random(), torch.rand(1).item()
        seed_all(42)
        expected = draw()
        seed_all(42)
        with isolated_rng(100):
            first = draw()
        self.assertEqual(draw(), expected)
        with isolated_rng(100):
            self.assertEqual(draw(), first)

    def test_legacy_rejected(self):
        with self.assertRaises(ValueError):
            validate({"select_action_calls": 355})

    def test_completed_artifact_required(self):
        with self.assertRaisesRegex(ValueError, "completed eval_info"):
            validate_completed(fixture())

    def test_completed_outcomes_reconciled(self):
        data = {"robopad_ledger": fixture(),
                "per_task": [{"task_group": "spatial", "task_id": 0,
                              "metrics": {"successes": [True], "sum_rewards": [1.0],
                                           "max_rewards": [1.0]}}],
                "overall": {"n_episodes": 1, "pc_success": 100., "eval_s": 1.}}
        validate_completed(data)
        data["per_task"][0]["task_id"] = 1
        with self.assertRaisesRegex(ValueError, "ledger/task|reported task successes"):
            validate_completed(data)

    def test_outcome_and_contract_tampering(self):
        d = fixture()
        d["contract"]["checkpoint"]["weights"] = "different"
        with self.assertRaises(ValueError):
            compare(fixture(), d, replay=True)
        d = fixture()
        full = {"robopad_ledger": d, "overall": {"n_episodes": 1, "pc_success": 100},
                "per_task": [{"task_group": "spatial", "task_id": 0, "metrics": {"successes": [False]}}]}
        with self.assertRaises(ValueError):
            validate(full)

    def test_coordinated_prefix_truncation(self):
        d = fixture()
        e = d["episodes"][0]
        e["producer_steps"] = e["producer_steps"][:1]
        e["environment_steps"] = e["environment_steps"][:1]
        e["select_calls"] = 1
        d["inferences"].pop()
        d["chunk_calls"]["measured"] = d["sampler_calls"]["measured"] = 1
        with self.assertRaises(ValueError):
            validate(d)


if __name__ == "__main__":
    unittest.main()
