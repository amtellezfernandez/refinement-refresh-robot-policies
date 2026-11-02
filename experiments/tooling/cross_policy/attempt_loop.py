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


def summarize_condition_check(results):
    """results: {condition: {"settled_state_sha256", "action_shape", "outcome"}} on one non-confirmation identity."""
    settled = {r["settled_state_sha256"] for r in results.values()}
    shapes_ok = all(len(r["action_shape"]) == 2 and r["action_shape"][1] == 7 for r in results.values())
    return {"reset_state_reused": len(settled) == 1, "valid_action_shapes": shapes_ok,
            "conditions": sorted(results), "status": "PASS" if len(settled) == 1 and shapes_ok and results else "FAIL",
            "scope": "One non-confirmation identity. Outcomes are not evidence."}
