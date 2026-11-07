"""Launch the Protocol B runner and honor its explicit fresh-process retry signal.

Pass the same arguments as runner_v2.py. The runner retains all release and
qualification checks and decides which attempt, if any, is eligible to retry.
"""
import subprocess
import sys
from pathlib import Path

from runner_v2 import EXIT_FRESH_PROCESS_RETRY, EXIT_STOPPED

# Protocol B accepts exactly 800 planned rows for one model. Each row can
# request at most one fresh-process retry.
MAX_RELAUNCHES = 800


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    runner = Path(__file__).resolve().with_name("runner_v2.py")
    command = [sys.executable, str(runner), *args]
    for launch in range(MAX_RELAUNCHES + 1):
        result = subprocess.run(command, check=False)
        if result.returncode != EXIT_FRESH_PROCESS_RETRY:
            return result.returncode if result.returncode >= 0 else 128 - result.returncode
        if launch < MAX_RELAUNCHES:
            print("Protocol B requested a fresh process for INFRA_PRE retry.", file=sys.stderr)
    print("Protocol B exceeded the per-schedule fresh-process retry limit.", file=sys.stderr)
    return EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
