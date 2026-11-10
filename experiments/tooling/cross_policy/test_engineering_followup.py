"""Real-weight denoising count and the client environment false positive. No model, GPU or simulator."""
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preflight_verdict as pv  # noqa: E402
import test_preflight_v2 as earlier  # noqa: E402

CHECK = HERE / "check_openpi_client_environment.py"


class Layer:
    calls = 0

    def __call__(self, value):
        return value


class Model:
    """Stands in for Pi0: one output projection per Euler step, as in pinned pi0.py."""
    def __init__(self, ignores_num_steps=False):
        self.action_out_proj = Layer()
        self.other = Layer()
        self.ignores = ignores_num_steps

    def sample_actions(self, rng, observation, num_steps=10, noise=None):
        self.other(0)
        for _ in range(10 if self.ignores else num_steps):
            self.action_out_proj(0)
        return "chunk-unjitted"


class Policy:
    metadata = {}

    def __init__(self, model):
        self._model = model
        self._sample_kwargs = {}
        self._sample_actions = lambda rng, observation, **kwargs: "chunk-jitted"

    def infer(self, obs, noise=None):
        kwargs = dict(self._sample_kwargs, noise=noise)
        return {"actions": self._sample_actions("rng", obs, **kwargs)}


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LoopCount(unittest.TestCase):
    def setUp(self):
        fake = types.SimpleNamespace(disable_jit=Context,
                                     local_devices=lambda: [types.SimpleNamespace(memory_stats=lambda: {})])
        patcher = mock.patch.dict(sys.modules, {"jax": fake})
        patcher.start()
        self.addCleanup(patcher.stop)
        import server_openpi
        self.server = server_openpi

    def measure(self, model):
        policy = self.server.KeyedPolicy(Policy(model))
        return policy, self.server.count_denoise_loop(policy, {"prompt": "x"})

    def test_counts_only_the_action_projection_through_policy_infer(self):
        _, counts = self.measure(Model())
        self.assertEqual(counts, {"2": 2, "5": 5, "10": 10})

    def test_a_model_that_ignores_num_steps_is_visible(self):
        _, counts = self.measure(Model(ignores_num_steps=True))
        self.assertEqual(counts, {"2": 10, "5": 10, "10": 10})

    def test_patches_are_removed_afterwards(self):
        model = Model()
        policy, _ = self.measure(model)
        self.assertEqual(policy.infer({"__d__": 2, "__noise_key__": "[1]"})["actions"], "chunk-jitted")
        Layer.__call__(model.action_out_proj, 0)
        self.assertEqual(type(model.action_out_proj).__call__.__name__, "__call__")


class VerdictUsesTheCount(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        earlier.fabricated_run(self.root)

    def edit(self, change):
        path = self.root / "pi05_profile/profile.json"
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_measured_count_passes(self):
        self.assertEqual(pv.evaluate(self.root)["checks"]["implementation_received_d"]["status"], "PASS")

    def test_wrong_count_fails(self):
        self.edit(lambda v: v["denoise_loop_calls"].update({"5": 10}))
        self.assertEqual(pv.evaluate(self.root)["checks"]["implementation_received_d"]["status"], "FAIL")

    def test_missing_count_fails(self):
        self.edit(lambda v: v.pop("denoise_loop_calls"))
        self.assertEqual(pv.evaluate(self.root)["checks"]["implementation_received_d"]["status"], "FAIL")


class EnvironmentCheck(unittest.TestCase):
    """Runs the real check on a fake LIBERO. Reported on the RTX with the old config: READY, under python3.12."""
    def run_check(self, paths=None, root_argument=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "source"
        inner = source / "libero/libero"
        inner.mkdir(parents=True)
        own = {"assets": inner / "assets", "bddl_files": inner / "bddl_files", "init_states": inner / "init_files",
               "benchmark_root": inner, "datasets": inner / "../datasets"}
        for key in ("assets", "bddl_files", "init_states"):
            own[key].mkdir()
        chosen = {k: str(v) for k, v in own.items()}
        if paths:
            for key, relative in paths.items():
                target = root / relative
                target.mkdir(parents=True, exist_ok=True)
                chosen[key] = str(target)
        (inner / "__init__.py").write_text("PATHS = {!r}\ndef get_libero_path(key):\n    return PATHS[key]\n".format(chosen))
        (inner / "benchmark.py").write_text(
            "def get_benchmark_dict():\n    return dict.fromkeys(('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'))\n")
        client = source / "openpi_client"
        client.mkdir()
        (client / "__init__.py").write_text("")
        (client / "websocket_client_policy.py").write_text("")
        command = [sys.executable, str(CHECK)] + (["--expected-libero-root", str(source)] if root_argument else [])
        return subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(source)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True)

    def test_own_tree_passes(self):
        done = self.run_check()
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertEqual(json.loads(done.stdout)["status"], "READY_FOR_CLIENT_IMPORT_CHECK_ONLY")

    def test_other_environment_bddl_fails(self):
        done = self.run_check({"bddl_files": "env/lib/python3.12/site-packages/libero/libero/bddl_files"})
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("bddl_files", done.stdout)

    def test_every_data_path_is_checked(self):
        for key in ("assets", "init_states"):
            with self.subTest(key=key):
                done = self.run_check({key: "elsewhere/" + key})
                self.assertNotEqual(done.returncode, 0, done.stdout)
                self.assertIn(key, done.stdout)

    def test_foreign_site_packages_fail_even_without_a_pinned_root(self):
        other = 9 if sys.version_info.minor != 9 else 10
        done = self.run_check({"init_states": "env/lib/python3.{}/site-packages/init".format(other)}, root_argument=False)
        self.assertNotEqual(done.returncode, 0, done.stdout)


if __name__ == "__main__":
    unittest.main()
