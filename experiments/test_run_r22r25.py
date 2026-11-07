"""Full orchestration with fake subprocesses only; no native stacks or identities."""
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('run_r22r25', Path(__file__).with_name('run_r22r25.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Operator(unittest.TestCase):
    def fixture(self, root):
        tools = root / 'repo/experiments/tooling/cross_policy_v3'
        tools.mkdir(parents=True)
        workspace = root / 'workspace with spaces'
        for model in ('smolvla', 'pi05'):
            (workspace / model / 'checkpoint').mkdir(parents=True)
        (tools / 'verify_campaign.py').write_text('print("fake qualification checked")')
        (tools / 'launch_v3.py').write_text('''import argparse
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output'); a,_=p.parse_known_args()
Path(a.output).mkdir(); (Path(a.output)/'attempts').mkdir()
print('fake run finished')
''')
        (tools / 'analysis_v2.py').write_text('''import argparse
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output'); a,_=p.parse_known_args()
Path(a.output).write_text('fake analysis')
''')
        (tools / 'server_openpi_v3.py').write_text('''import argparse,socket,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--port',type=int); a,_=p.parse_known_args()
Path(a.output).mkdir(); (Path(a.output)/'startup_gate.json').write_text('{}')
s=socket.socket(); s.bind(('127.0.0.1',a.port)); s.listen(); time.sleep(60)
''')
        return tools.parents[2], workspace

    def test_complete_fake_round_and_service_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root, workspace = self.fixture(Path(directory))
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            plan = runner.plan(workspace, sys.executable, sys.executable, port=port, root=root)
            self.assertEqual([n for n,_ in plan][:2], ['smolvla-verify', 'pi05-verify'])
            runner.execute(plan, workspace, port=port, startup_timeout=10, root=root)
            for model in ('smolvla', 'pi05'):
                self.assertEqual((workspace / model / 'analysis.json').read_text(), 'fake analysis')
            receipt = json.loads(next((workspace / 'operator-logs').glob('launch-*/pi05-server.receipt.json')).read_text())
            self.assertIsNotNone(receipt['exit_code'])
            with self.assertRaisesRegex(ValueError, 'Existing output'):
                runner.execute(plan, workspace, port=port, root=root)

    def test_server_death_after_readiness_stops_client_and_analysis(self):
        import benchstudio as bs
        with tempfile.TemporaryDirectory() as directory:
            root, workspace = self.fixture(Path(directory))
            tools = root / 'experiments/tooling/cross_policy_v3'
            server = tools / 'server_openpi_v3.py'
            source = server.read_text().replace("time.sleep(60)",
                "\nwhile not Path('client-running').exists(): time.sleep(0.01)\n")
            server.write_text(source)
            client = tools / 'launch_v3.py'
            client.write_text(client.read_text() +
                "\nimport sys,time\nif 'pi05' in sys.argv:\n"
                "    Path('client-running').touch()\n    time.sleep(1)\n"
                "    Path('client-incorrectly-completed').touch()\n")
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            commands = runner.plan(workspace, sys.executable, sys.executable, port=port, root=root)
            with self.assertRaisesRegex(bs.ProcessFailure, 'required service'):
                runner.execute(commands, workspace, port=port, startup_timeout=10, root=root)
            self.assertFalse((root / 'client-incorrectly-completed').exists())
            self.assertFalse((workspace / 'pi05/analysis.json').exists())
            receipt = json.loads(next((workspace / 'operator-logs').glob('launch-*/pi05-campaign.receipt.json')).read_text())
            self.assertEqual(receipt['reason'], 'ProcessFailure')

    def test_stopped_native_run_is_analyzed_without_starting_other_model(self):
        import benchstudio as bs
        with tempfile.TemporaryDirectory() as directory:
            root, workspace = self.fixture(Path(directory))
            script = root / 'experiments/tooling/cross_policy_v3/launch_v3.py'
            script.write_text("import argparse; from pathlib import Path\n"
                "p=argparse.ArgumentParser(); p.add_argument('--output'); a,_=p.parse_known_args()\n"
                "Path(a.output).mkdir(); (Path(a.output)/'run_summary.json').write_text('{}')\n"
                "raise SystemExit(3)\n")
            plan = runner.plan(workspace, sys.executable, sys.executable, root=root)
            with self.assertRaises(bs.ProcessFailure, msg='must preserve the native stop'):
                runner.execute(plan, workspace, root=root)
            self.assertTrue((workspace / 'smolvla/analysis.json').exists())
            self.assertFalse((workspace / 'pi05/server').exists())

    def test_failed_second_precheck_prevents_all_native_jobs(self):
        import benchstudio as bs
        with tempfile.TemporaryDirectory() as directory:
            root, workspace = self.fixture(Path(directory))
            script = root / 'experiments/tooling/cross_policy_v3/verify_campaign.py'
            script.write_text('import sys; raise SystemExit(2 if "pi05" in sys.argv else 0)')
            plan = runner.plan(workspace, sys.executable, sys.executable, root=root)
            with self.assertRaises(bs.ProcessFailure):
                runner.execute(plan, workspace, root=root)
            self.assertFalse((workspace / 'smolvla/run').exists())
            self.assertFalse((workspace / 'pi05/server').exists())


if __name__ == '__main__':
    unittest.main()
