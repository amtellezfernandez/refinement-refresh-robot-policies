#!/usr/bin/env python3
"""CPU-only dependency gate for the OpenPI LIBERO client environment.

Run this with the exact Python interpreter that will run client_openpi.py,
before reserving the GPU or starting a versioned preflight. The previous
counted run failed with ModuleNotFoundError: libero; no historical evidence is
changed by this tool.

Example:
  envs/client/bin/python check_openpi_client_environment.py \
      --expected-libero-root /path/to/pinned/libero/source/root
"""
import argparse
import json
import sys
from pathlib import Path


def check(expected_root=None):
    try:
        import libero
        from libero.libero import benchmark, get_libero_path
        from openpi_client import websocket_client_policy
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "OpenPI LIBERO client environment incomplete: {}. Install the pinned "
            "LIBERO source into this client interpreter; do not silently import "
            "binary site-packages from a different Python environment.".format(error)
        )
    import libero.libero as _inner
    package = Path(_inner.__file__).resolve()
    bddl = Path(get_libero_path("bddl_files")).resolve()
    if expected_root is not None:
        root = Path(expected_root).resolve()
        if root not in package.parents:
            raise RuntimeError("libero package {} is not under pinned root {}".format(package, root))
    if not bddl.is_dir():
        raise RuntimeError("LIBERO BDDL path does not exist: {}".format(bddl))
    suites = tuple(benchmark.get_benchmark_dict().keys())
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if suite not in suites:
            raise RuntimeError("missing LIBERO suite: {}".format(suite))
    for entry in sys.path:
        # A Python 3.8 client must not borrow site-packages with binary extensions
        # built for Python 3.12 (and vice versa).
        if "site-packages" in entry:
            for minor in range(8, 14):
                if minor != sys.version_info.minor and "python3.{}".format(minor) in entry:
                    raise RuntimeError("cross-interpreter site-packages on sys.path: {}".format(entry))
    return {"status": "READY_FOR_CLIENT_IMPORT_CHECK_ONLY",
            "python": sys.version, "python_executable": sys.executable,
            "libero_module": str(package), "bddl_directory": str(bddl),
            "openpi_client_module": str(Path(websocket_client_policy.__file__).resolve()),
            "suites_found": list(suites)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-libero-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.expected_libero_root), indent=2))


if __name__ == "__main__":
    main()
