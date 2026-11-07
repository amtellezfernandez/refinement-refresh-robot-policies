"""Offline checks for the fresh-process launcher, using a fake runner."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LaunchV3Test(unittest.TestCase):
    def check_launch(self, statuses, expected, calls):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(Path(__file__).with_name("launch_v3.py"), root / "launch_v3.py")
            (root / "runner_v2.py").write_text(
                """import json
import sys
from pathlib import Path

EXIT_FRESH_PROCESS_RETRY, EXIT_STOPPED = 75, 3

if __name__ == "__main__":
    path = Path("calls.json")
    calls = json.loads(path.read_text()) if path.exists() else []
    calls.append(sys.argv[1:])
    path.write_text(json.dumps(calls))
    sys.exit(int(sys.argv[1].split(",")[len(calls) - 1]))
""",
                encoding="utf-8",
            )
            args = [statuses, "--output", "run directory", "; literal"]
            # Keep the retry-cap test small without modifying the launcher on disk.
            command = [
                sys.executable,
                "-c",
                "import sys, launch_v3; launch_v3.MAX_RELAUNCHES = 2; "
                "sys.exit(launch_v3.main(sys.argv[1:]))",
                *args,
            ]
            result = subprocess.run(command, cwd=root, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected, result.stderr.decode())
            self.assertEqual(json.loads((root / "calls.json").read_text()), [args] * calls)

    def test_retries_explicit_status_with_same_arguments(self):
        self.check_launch("75,75,0", 0, 3)

    def test_stops_after_non_retry_status(self):
        self.check_launch("75,3", 3, 2)

    def test_propagates_refusal_without_retry(self):
        self.check_launch("2", 2, 1)

    def test_stops_at_retry_cap(self):
        self.check_launch("75,75,75", 3, 3)


if __name__ == "__main__":
    unittest.main()
