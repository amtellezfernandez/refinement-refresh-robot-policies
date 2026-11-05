# NON-COUNTING probe driver. Runs the frozen runner_smolvla.preflight() unmodified up to its profile step.
# Parity is prevented deterministically: the module-level AttemptJournal, first used at the start of parity
# (runner_smolvla.py:292, after profile.json is written and env.close()), is replaced by a function that raises.
import argparse, json, sys
from pathlib import Path
C = Path(sys.argv[1]); sys.path[:0] = [str(C), str(C.parent)]
import launch_guard, runner_smolvla
class ParityBlocked(SystemExit):
    pass
def blocked(*args, **kwargs):
    raise ParityBlocked("PARITY_BLOCKED_BY_PROBE: AttemptJournal({}) refused; no parity environment or episode created".format(args))
runner_smolvla.AttemptJournal = blocked
print("guard", json.dumps(launch_guard.check("preflight")), flush=True)
args = argparse.Namespace(mode="preflight", phase="preflight", schedule=None, states=None, checkpoint=Path(sys.argv[2]), output=Path(sys.argv[3]))
try:
    runner_smolvla.preflight(args)
    print("UNEXPECTED: preflight returned without reaching parity", flush=True); sys.exit(5)
except ParityBlocked as stop:
    print(stop, flush=True); sys.exit(0)
