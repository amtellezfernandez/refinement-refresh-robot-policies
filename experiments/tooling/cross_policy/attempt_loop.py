"""The attempt loop shared by both runners. Standard library only; Python 3.8+.

Two rules live here, because a recorded flag does not enforce anything:
- Exact reset-state reuse: an attempt whose settled state differs from its block's stops before
  any policy query. The reference survives a process restart, because it is read from the journal.
- A failed attempt is journaled and then stops the run. Nothing is retried or skipped silently, and
  one server crash cannot mark thousands of planned attempts as failed. pi0.5's stop rule on
  unsupported_refinement_setting is enforced by the same path.
"""
import time


class ResetStateMismatch(ValueError):
    pass


class AttemptStopped(RuntimeError):
    """A failure has been journaled and the run has stopped for review."""


def check_settled(settled, expected):
    if expected is not None and settled != expected:
        raise ResetStateMismatch("settled state {} differs from block reference {}".format(settled, expected))


def block_references(journal):
    references = {}
    for record in journal.records():
        settled = record.get("settled_state_sha256")
        if settled:
            references.setdefault(record["schedule_row"]["block_id"], settled)
    return references


def execute_schedule(rows, journal, provenance_sha256, argv, run_one, failure_fields=None):
    """run_one(row, expected_settled) returns journal fields, including settled_state_sha256."""
    references = block_references(journal)
    for row in rows:
        if journal.attempts_for(row["planned_run_id"]):
            continue
        path = journal.begin(row, provenance_sha256, argv)
        journal.update(path, "running", started_unix=time.time())
        try:
            fields = run_one(row, references.get(row["block_id"]))
            references.setdefault(row["block_id"], fields["settled_state_sha256"])
            journal.update(path, "completed", finished_unix=time.time(), **fields)
        except Exception as error:  # Retained as a record; never replaced.
            extra = failure_fields() if failure_fields else {}
            journal.update(path, "failed", error=repr(error), finished_unix=time.time(), **extra)
            raise AttemptStopped("{} failed: {!r}".format(row["planned_run_id"], error)) from error


REQUIRED_CONDITIONS = {"NATIVE", "R22", "R25", "R52", "R55"}
# Same reset, noise key and d at tick 0, so the first action must be identical. h has not yet acted.
MATCHED_FIRST_ACTION = (("R22", "R52"), ("R25", "R55"))


def summarize_condition_check(results, negative_control_refused):
    """One non-confirmation identity through every condition. Outcomes are not evidence."""
    complete = set(results) == REQUIRED_CONDITIONS
    settled = {r["settled_state_sha256"] for r in results.values()}
    shapes_ok = all(len(r["action_shape"]) == 2 and r["action_shape"][1] == 7 for r in results.values())
    received = all(r["received_d"] for r in results.values())
    matched = {"{}={}".format(a, b): a in results and b in results and
               results[a]["first_action_sha256"] == results[b]["first_action_sha256"] for a, b in MATCHED_FIRST_ACTION}
    passed = complete and len(settled) == 1 and shapes_ok and received and all(matched.values()) and negative_control_refused
    return {"all_conditions_present": complete, "reset_state_reused": len(settled) == 1, "valid_action_shapes": shapes_ok,
            "implementation_received_d": received, "matched_first_actions": matched,
            "reset_negative_control_refused": negative_control_refused, "conditions": sorted(results),
            "status": "PASS" if passed else "FAIL", "scope": "One non-confirmation identity. Outcomes are not evidence."}
