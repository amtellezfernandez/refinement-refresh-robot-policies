"""The frozen 1 November script stays frozen; the versioned entry point carries the LIBERO path.

Runs the real scripts and the real client code in a fake workspace, with no model, GPU or simulator.
Stubs stand in for nvidia-smi, flock, the two model environments and the server. The fake LIBERO has a
namespace top level, like OpenPI's third_party/libero, where libero.__file__ is None.
"""
import hashlib
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
FROZEN = HERE / "run_gpu_preflight.sh"
VERSIONED = HERE / "run_preflight_v2.sh"
FROZEN_BLOB = "6ae6c11acdc4aad49cdfda0ea2b9fee3c8302dcc"  # At bb86691, the commit alba-preflight-2 ran.
SENTINEL = "FAKE_BENCHMARK_REACHED"


def write_exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def port_free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def fake_workspace(root):
    workspace, bin_dir, home = root / "w", root / "bin", root / "home"
    (home / "local-science-programme-20261008").mkdir(parents=True)
    write_exe(bin_dir / "nvidia-smi", "#!/bin/sh\necho 0\n")
    write_exe(bin_dir / "flock", "#!/bin/sh\nexit 0\n")
    write_exe(bin_dir / "timeout", '#!/bin/sh\nshift\nexec "$@"\n')
    write_exe(bin_dir / "sha256sum", "#!/bin/sh\necho 0\n")
    write_exe(workspace / "envs/smolvla/bin/python", "#!/bin/sh\nexit 0\n")
    write_exe(workspace / "envs/tools/bin/python", "#!/bin/sh\nexit 0\n")
    listener = ("import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
                "s.bind(('127.0.0.1',8000));s.listen(8);time.sleep(60)")
    write_exe(workspace / "envs/openpi/bin/python",
              '#!/bin/bash\ncase "$*" in\n*"server_openpi.py" *serve*) exec "{}" -c "{}";;\n'
              '*) while [ $# -gt 0 ]; do [ "$1" = --output ] && mkdir -p "$2"; shift; done;;\nesac\n'.format(sys.executable, listener))
    # The client environment has no libero of its own, exactly like envs/client on the RTX.
    write_exe(workspace / "envs/client/bin/python", '#!/bin/sh\nexec "{}" "$@"\n'.format(sys.executable))
    source = workspace / "src/openpi/third_party/libero"
    inner = source / "libero/libero"  # No __init__.py one level up: a namespace package.
    inner.mkdir(parents=True)
    (inner / "__init__.py").write_text("def get_libero_path(key):\n    return __file__.rsplit('/', 1)[0]\n")
    (inner / "benchmark.py").write_text(
        "import os\n"
        "def get_benchmark_dict():\n"
        "    raise RuntimeError('{} ' + __file__ + ' CONFIG=' + os.environ.get('LIBERO_CONFIG_PATH', ''))\n".format(SENTINEL))
    client = source / "openpi_client"
    client.mkdir()
    (client / "__init__.py").write_text("")
    (client / "websocket_client_policy.py").write_text("")
    environment = {"PATH": "{}:{}".format(bin_dir, os.environ["PATH"]), "HOME": str(home), "ALBA_GPU_RESERVATION": "TEST"}
    return workspace, environment, inner


@unittest.skipUnless(port_free(8000), "the scripts' fixed port 8000 is in use")
class Scripts(unittest.TestCase):
    def run_script(self, script):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        workspace, environment, inner = fake_workspace(root)
        output = root / "out"
        done = subprocess.run(["bash", str(script), str(workspace), str(output)], env=environment, timeout=120,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        return done, output, inner, workspace

    def test_frozen_script_still_shows_the_original_failure(self):
        done, output, _, _ = self.run_script(FROZEN)
        report = json.loads((output / "pi05/pi05_client_small.json").read_text())
        self.assertIn("No module named 'libero'", report["error"], done.stdout)

    def test_versioned_entry_point_reaches_openpi_libero(self):
        done, _, inner, workspace = self.run_script(VERSIONED)
        self.assertNotIn("No module named 'libero'", done.stdout)
        # Identity, not silence: the import resolves to the workspace's OpenPI tree.
        self.assertIn(SENTINEL, done.stdout, done.stdout)
        self.assertIn(str(inner / "benchmark.py"), done.stdout)


class FrozenIdentity(unittest.TestCase):
    def test_frozen_script_is_byte_identical_to_the_run_that_counted(self):
        data = FROZEN.read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        self.assertEqual(blob, FROZEN_BLOB)


if __name__ == "__main__":
    unittest.main()
