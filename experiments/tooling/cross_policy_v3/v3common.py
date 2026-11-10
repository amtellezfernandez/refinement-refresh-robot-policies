"""Constants and helpers for the v3 tooling (Protocol B, Protocol E, qualification). Python 3.8+, stdlib only.

Imports the v2 helpers in ../cross_policy without changing them. Nothing here runs a simulator, a model or a GPU.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "cross_policy"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from common import (CONFIRMATION_SEEDS, STATE_SLOTS, SUITES, TASK_IDS, atomic_write_json,  # noqa: E402,F401
                    canonical, file_sha, identity_id, keyed_int, require, sha)

PROTOCOL_B = "cross_policy_confirmation_v2"
MODELS = ("smolvla", "pi05")
# condition_id -> (d, h). R22 = d=2, h=2; R25 = d=5, h=2 (02 B2).
CONDITIONS = {"R22": {"d": 2, "h": 2}, "R25": {"d": 5, "h": 2}}
ATTEMPTS_PER_MODEL = 800
IDENTITIES_PER_MODEL = 400
REPO_ROOT = HERE.parents[2]
PROTOCOL_B_DOCUMENT = REPO_ROOT / "review/preflight_v2/protocol_v3_revision_20261106/02_SCIENTIFIC_CONFIRMATION.md"
PROTOCOL_E_DOCUMENT = REPO_ROOT / "review/preflight_v2/protocol_v3_revision_20261106/13_EVALUATOR_VALIDATION.md"


class Population(object):
    """Which identities a schedule may contain. The confirmation population is reserved (02 B3)."""

    def __init__(self, cohort, suites, task_ids, state_slots, policy_seeds, reserved):
        self.cohort, self.suites, self.task_ids = cohort, tuple(suites), tuple(task_ids)
        self.state_slots, self.policy_seeds, self.reserved = tuple(state_slots), tuple(policy_seeds), bool(reserved)

    def state_keys(self):
        return ["{}.task{:03d}.slot{}".format(s, t, k) for s in self.suites for t in self.task_ids for k in self.state_slots]

    def identity_count(self):
        return len(self.suites) * len(self.task_ids) * len(self.state_slots) * len(self.policy_seeds)


CONFIRMATION_POPULATION = Population(PROTOCOL_B, SUITES, TASK_IDS, STATE_SLOTS, CONFIRMATION_SEEDS, reserved=True)


def state_key(suite, task_id, state_slot):
    return "{}.task{:03d}.slot{}".format(suite, task_id, state_slot)

