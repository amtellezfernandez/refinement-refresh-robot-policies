#!/usr/bin/env python3
"""Protocol B runner (02 B5-B9). Python 3.8+. NEVER RUN against a simulator or a model.

Model-agnostic core plus a CLI that loads one model-native adapter (adapter_pi05.py or adapter_smolvla.py).
Rules enforced here, by records rather than by convention:
- G0-B: Alba's release binds the Protocol B document and this model's schedule (release.check_release).
- PRE: every qualification item for this model is PASSED (qualification.require_qualified).
- Write-ahead journal (v2 journal.py, imported unchanged); every attempt record exists before execution.
- Fresh environment per attempt (the adapter's contract; the runner never passes an env between attempts).
- First-policy-query marker, written durably before the first query; it alone splits INFRA_PRE from INFRA_POST.
- INFRA_PRE: one retry, only as the first attempt of a fresh process (exit 75 asks the launcher for one).
- INFRA_POST: no retry; more than 20 halts the run. INSTRUMENT_INCIDENT and PROTOCOL_STOP stop the run.
- A stopped or halted run never resumes here; resumption needs an amendment Alba approves (02 B9).
- No row is ever passed over without a final class; the summary lists every planned attempt.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify as C  # noqa: E402
from v3common import CONFIRMATION_POPULATION, MODELS, PROTOCOL_B, atomic_write_json, file_sha, sha  # noqa: E402
from attempt_loop import block_references  # noqa: E402  (frozen v2 module, imported unchanged)
from journal import AttemptJournal  # noqa: E402  (v2 module, imported unchanged)

EXIT_COMPLETED, EXIT_STOPPED, EXIT_FRESH_PROCESS_RETRY = 0, 3, 75
MAX_INFRA_POST_ATTEMPTS = 20  # 02 B9 operational halt (attempts, not pairs)
HALT_FILE = "HALTED.json"


class AttemptContext(object):
    """What an adapter receives for one attempt. The adapter must build a fresh env inside run_attempt."""

    def __init__(self, journal, path, row, attempt_number, expected_reset, run_dir):
        self.row, self.attempt_number, self.expected_reset = row, attempt_number, expected_reset
        self._path, self.run_dir, self.reset_hashes = Path(path), Path(run_dir), None
        self.arrays_dir = self.run_dir / "arrays"
        self.arrays_dir.mkdir(exist_ok=True)

    def marker_path(self):
        return self._path.with_name(self._path.name + ".first_query")

    def mark_first_query(self):
        """Durably record that the first policy query is about to be sent. Call before sending it."""
        descriptor = os.open(str(self.marker_path()), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(descriptor, json.dumps({"unix": time.time()}).encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def first_query_started(self):
        return self.marker_path().exists()

    def record_reset(self, settled_sha256, first_observation_sha256):
        """Adapters call this as soon as both reset hashes exist, so a later failure still carries them."""
        self.reset_hashes = (settled_sha256, first_observation_sha256)


def record_class(record, journal_root):
    """The class of one attempt record, including records left open by a dead process."""
    if record.get("attempt_class") in C.ALL_CLASSES:
        return record["attempt_class"]
    if record["status"] == "interrupted":
        marker = Path(journal_root) / "{}.attempt{:02d}.json.first_query".format(
            record["planned_run_id"], record["attempt_number"])
        return C.interrupted_class(marker.exists())
    raise ValueError("record {} attempt {} has status {} and no class".format(
        record["planned_run_id"], record["attempt_number"], record["status"]))


def classes_by_row(journal):
    out = {}
    for record in journal.records():
        out.setdefault(record["planned_run_id"], []).append((record["attempt_number"],
                                                             record_class(record, journal.root)))
    return {k: [c for _, c in sorted(v)] for k, v in out.items()}


def stop_state(classes, run_dir):
    """Why this run may not continue, or None."""
    halt = Path(run_dir) / HALT_FILE
    if halt.exists():
        return json.loads(halt.read_text("utf-8"))["run_status"]
    flat = [c for values in classes.values() for c in values]
    if C.PROTOCOL_STOP in flat:
        return "STOPPED_PROTOCOL"
    if C.INSTRUMENT_INCIDENT in flat:
        return "STOPPED_INSTRUMENT_INCIDENT"
    if flat.count(C.INFRA_POST) > MAX_INFRA_POST_ATTEMPTS:
        return "HALTED_INFRA_POST"
    return None


def write_halt(run_dir, run_status, reason):
    path = Path(run_dir) / HALT_FILE
    if not path.exists():
        atomic_write_json(path, {"run_status": run_status, "reason": reason, "unix": time.time(),
                                 "resumption": "only under an amendment Alba approves before any outcome is read"})


def write_incident(run_dir, row, attempt_number, error):
    directory = Path(run_dir) / "incidents"
    directory.mkdir(exist_ok=True)
    details = getattr(error, "details", {})
    atomic_write_json(directory / "{}.attempt{:02d}.json".format(row["planned_run_id"], attempt_number),
                      {"planned_run_id": row["planned_run_id"], "attempt_number": attempt_number,
                       "attempt_class": error.attempt_class, "message": str(error),
                       "details": json.loads(json.dumps(details, default=repr)),
                       "status": "UNDER_INVESTIGATION", "action_applied": False})


def run_one(journal, row, adapter, run_dir, provenance_sha256, argv, retry_reason, references):
    attempt_number = len(journal.attempts_for(row["planned_run_id"]))
    path = journal.begin(row, provenance_sha256, argv, retry_reason=retry_reason)
    journal.update(path, "running", started_unix=time.time(), stack=adapter.stack_config())
    context = AttemptContext(journal, path, row, attempt_number, references.get(row["block_id"]), run_dir)
    try:
        result = adapter.run_attempt(context)
        try:
            attempt_class = C.classify_outcome(result["outcome"])
            reset = (result["settled_state_sha256"], result["first_observation_sha256"])
            provenance = result["action_provenance"]
        except (KeyError, TypeError, ValueError) as error:
            # A malformed result is an adapter or evaluator contract fault, never a missing outcome.
            raise C.InstrumentIncident("malformed attempt result: {!r}".format(error))
        journal.update(path, "completed", finished_unix=time.time(), attempt_class=attempt_class,
                       y=C.outcome_value(attempt_class), outcome=result["outcome"],
                       settled_state_sha256=reset[0], first_observation_sha256=reset[1],
                       settled_state_matches_block=True, action_provenance=provenance,
                       action_array=result.get("action_array"), timing=result.get("timing"),
                       first_query_marker=context.first_query_started())
        references.setdefault(row["block_id"], reset)
        return attempt_class
    except Exception as error:  # every failure is journaled and classified; nothing is swallowed
        attempt_class = C.classify_exception(error, context.first_query_started())
        fields = {"finished_unix": time.time(), "attempt_class": attempt_class, "error": repr(error),
                  "first_query_marker": context.first_query_started()}
        if context.reset_hashes is not None:
            fields["settled_state_sha256"], fields["first_observation_sha256"] = context.reset_hashes
        journal.update(path, "failed", **fields)
        if attempt_class in C.STOPPING:
            write_incident(run_dir, row, attempt_number, error if isinstance(error, C.ClassifiedFailure)
                           else C.InstrumentIncident(repr(error)))
        return attempt_class


def execute(rows, journal, adapter, run_dir, provenance_sha256, argv):
    """Run every planned attempt in schedule order. Returns (run_status or None, exit code)."""
    print("interrupted:", journal.reconcile_interrupted("process restart"), flush=True)
    executed_here = 0
    while True:
        classes = classes_by_row(journal)
        stopped = stop_state(classes, run_dir)
        if stopped:
            write_halt(run_dir, stopped, "stop condition present in the journal")
            return stopped, EXIT_STOPPED
        pending = None
        for row in rows:
            final = C.final_class(classes.get(row["planned_run_id"], []))
            if final is None:
                pending = row
                break
        if pending is None:
            return "COMPLETED", EXIT_COMPLETED
        prior = classes.get(pending["planned_run_id"], [])
        retry_reason = None
        if prior:
            # Exactly one INFRA_PRE record: the single permitted retry, only in a fresh process (02 B7).
            if prior != [C.INFRA_PRE]:
                raise RuntimeError("unexpected pending state {} for {}".format(prior, pending["planned_run_id"]))
            if executed_here:
                return None, EXIT_FRESH_PROCESS_RETRY
            retry_reason = "INFRA_PRE retry in a fresh process and fresh env (02 B7)"
        references = block_references(journal)
        run_one(journal, pending, adapter, run_dir, provenance_sha256, argv, retry_reason, references)
        executed_here += 1


def summarize(rows, journal, run_dir, schedule_sha256, model, run_status):
    classes = classes_by_row(journal)
    finals, accounting = {}, {"attempt_records": 0, "by_class": {c: 0 for c in C.ALL_CLASSES},
                               "by_arm_and_suite": {}, "infra_pre_retries": 0, "not_executed": []}
    for row in rows:
        values = classes.get(row["planned_run_id"], [])
        accounting["attempt_records"] += len(values)
        cell = accounting["by_arm_and_suite"].setdefault(
            "{}|{}".format(row["condition_id"], row["episode_identity"]["suite"]),
            dict({c: 0 for c in C.ALL_CLASSES}, infra_pre_retries=0))
        for value in values:
            accounting["by_class"][value] += 1
            cell[value] += 1
        if len(values) > 1:
            cell["infra_pre_retries"] += 1
        if len(values) > 1:
            accounting["infra_pre_retries"] += 1
        final = C.final_class(values)
        if final is None and not values:
            accounting["not_executed"].append(row["planned_run_id"])
        finals[row["planned_run_id"]] = final
    if run_status == "COMPLETED" and (accounting["not_executed"] or
                                      any(v not in C.FINAL_SCIENTIFIC + (C.MISSING, C.INVALIDATED)
                                          for v in finals.values())):
        raise RuntimeError("COMPLETED with an attempt lacking a final class: refusing to summarise")
    summary = {"schema_version": 1, "protocol": PROTOCOL_B, "model": model, "schedule_sha256": schedule_sha256,
               "run_status": run_status, "final_classes": finals, "accounting": accounting,
               "journal_checksums": journal.checksums()}
    atomic_write_json(Path(run_dir) / "run_summary.json", summary)
    return summary


def record_started(run_dir, model, schedule_sha256, argv, bound):
    """Scheduler-independent `started` record (02 B10). Later launches append, never replace."""
    started = Path(run_dir) / "started"
    entry = {"unix": time.time(), "model": model, "schedule_sha256": schedule_sha256, "argv_sha256": sha(argv),
             "runner_sha256": file_sha(__file__), "pid": os.getpid(), "release": bound}
    try:
        descriptor = os.open(str(started), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        with started.with_name("launches.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def load_adapter(model, args):
    if model == "pi05":
        import adapter_pi05
        if args.startup_gate is None:
            raise SystemExit("REFUSED: pi0.5 needs --startup-gate (the server's startup_gate.json)")
        return adapter_pi05.Pi05ClientAdapter(states_path=args.states, port=args.port, startup_gate=args.startup_gate)
    import adapter_smolvla
    return adapter_smolvla.SmolVLAAdapter(states_path=args.states, checkpoint=args.checkpoint)


def main():
    import qualification
    import release
    parser = argparse.ArgumentParser(description="Protocol B runner (needs Alba's release; never run).")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000, help="pi0.5 attesting server port")
    parser.add_argument("--checkpoint", type=Path, help="SmolVLA checkpoint directory")
    parser.add_argument("--startup-gate", type=Path, help="pi0.5: the attesting server's startup_gate.json")
    args = parser.parse_args()
    # Gates first: nothing below may import a simulator or framework until both pass.
    bound = release.check_release(args.schedule, args.model)
    bound["qualification_sha256"] = qualification.require_qualified(args.model)
    if args.model == "smolvla":  # this process holds the GPU; the pi0.5 client does not (its server does)
        import launch_guard
        bound["reservation"] = launch_guard.need_reservation(os.environ)
        running = launch_guard.compute_processes()
        if running:
            raise SystemExit("REFUSED: GPU compute processes present: {}".format(running))
    document = json.loads(args.schedule.read_text("utf-8"))
    rows = document["schedule"]
    if document.get("protocol") != PROTOCOL_B or document.get("cohort") != CONFIRMATION_POPULATION.cohort:
        raise SystemExit("REFUSED: not a Protocol B schedule")
    if document.get("model") != args.model or len(rows) != 800:
        raise SystemExit("REFUSED: schedule is not this model's 800-row schedule")
    args.output.mkdir(parents=True, exist_ok=True)
    record_started(args.output, args.model, document["schedule_sha256"], sys.argv, bound)
    adapter = load_adapter(args.model, args)
    provenance_sha256 = adapter.provenance_sha256()
    journal = AttemptJournal(args.output / "attempts")
    run_status, code = execute(rows, journal, adapter, args.output, provenance_sha256, sys.argv)
    if run_status is not None:
        summarize(rows, journal, args.output, document["schedule_sha256"], args.model, run_status)
    adapter.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
