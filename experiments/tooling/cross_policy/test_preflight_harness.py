"""Regression: run the real run_gpu_preflight.sh and the real pi0.5 client script in a fake workspace.

No model, GPU or simulator. Stubs stand in for nvidia-smi, flock, the two model environments and the
server. The client script and the wrapper are the ones that failed in alba-preflight-2, with:
    ModuleNotFoundError: No module named 'libero'
"""
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_gpu_preflight.sh"
SENTINEL = "FAKE_BENCHMARK_REACHED"


def write_exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def port_free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return probe.connect_ex(("127.0.0.1", port)) != 0


@unittest.skipUnless(port_free(8000), "the wrapper's fixed port 8000 is in use")
class ClientLiberoPath(unittest.TestCase):
    def run_wrapper(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        workspace, bin_dir, home = root / "w", root / "bin", root / "home"
        (home / "local-science-programme-20261008").mkdir(parents=True)
        write_exe(bin_dir / "nvidia-smi", "#!/bin/sh\necho 0\n")
        write_exe(bin_dir / "flock", "#!/bin/sh\nexit 0\n")
        write_exe(bin_dir / "timeout", '#!/bin/sh\nshift\nexec "$@"\n')
        write_exe(workspace / "envs/smolvla/bin/python", "#!/bin/sh\nexit 0\n")
        listener = ("import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
                    "s.bind(('127.0.0.1',8000));s.listen(8);time.sleep(60)")
        write_exe(workspace / "envs/openpi/bin/python",
                  '#!/bin/bash\ncase "$*" in\n*server_openpi.py*) exec "{}" -c "{}";;\n'
                  '*) while [ $# -gt 0 ]; do [ "$1" = --output ] && mkdir -p "$2"; shift; done;;\nesac\n'.format(sys.executable, listener))
        # The client environment has no libero of its own, exactly like envs/client on the RTX.
        write_exe(workspace / "envs/client/bin/python", '#!/bin/sh\nexec "{}" "$@"\n'.format(sys.executable))
        package = workspace / "src/openpi/third_party/libero/libero/libero"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "benchmark.py").write_text(
            "def get_benchmark_dict():\n    raise RuntimeError('{} ' + __file__)\n".format(SENTINEL))
        environment = {"PATH": "{}:{}".format(bin_dir, os.environ["PATH"]), "HOME": str(home),
                       "ALBA_GPU_RESERVATION": "TEST"}
        output = root / "out"
        done = subprocess.run(["bash", str(SCRIPT), str(workspace), str(output)], env=environment, timeout=120,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        report = json.loads((output / "pi05/pi05_client_small.json").read_text())
        return done, report, package

    def test_client_finds_openpi_libero_from_the_workspace(self):
        done, report, package = self.run_wrapper()
        self.assertNotIn("No module named 'libero'", report.get("error", ""), done.stdout)
        # Identity, not silence: the import must resolve to the workspace's OpenPI LIBERO package.
        self.assertIn(SENTINEL, report["error"])
        self.assertIn(str(package / "benchmark.py"), report["error"])


if __name__ == "__main__":
    unittest.main()
