"""Write-ahead attempt journal: one JSON record per attempt, updated in place.

Rules enforced here, because a convention would not hold under a crash or a retry:
- The record exists, durably, before execution starts (exclusive create).
- A terminal record can never be changed or replaced.
- Unfinished records found later become `interrupted`; they are never deleted.
- There are no automatic retries. A further attempt needs a stated reason and keeps
  its predecessor, and `analyze.py` reports it as a deviation.
"""
import json
import os
import socket
import time
from pathlib import Path

from common import atomic_write_json, file_sha, require

OPEN = ("planned", "running")
TERMINAL = ("completed", "failed", "timeout", "interrupted")
ALLOWED = {"planned": {"running", "failed", "interrupted"},
           "running": {"completed", "failed", "timeout", "interrupted"}}


class TerminalRecordError(ValueError):
    """Raised on any attempt to alter a finished attempt."""


def now():
    return time.time()


class AttemptJournal:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, planned_run_id, attempt_number):
        require("/" not in planned_run_id and ".." not in planned_run_id, "unsafe planned_run_id")
        return self.root / "{}.attempt{:02d}.json".format(planned_run_id, attempt_number)

    def attempts_for(self, planned_run_id):
        return sorted(self.root.glob("{}.attempt[0-9][0-9].json".format(planned_run_id)))

    def begin(self, row, provenance_sha256, command_line, retry_reason=None):
        """Durably create the record before execution. Returns the record path."""
        planned = row["planned_run_id"]
        existing = self.attempts_for(planned)
        if existing:
            require(retry_reason, "attempt already exists for {}; no silent retry".format(planned))
            for path in existing:
                require(json.loads(path.read_text("utf-8"))["status"] in TERMINAL,
                        "earlier attempt still open: {}".format(path))
        number = len(existing)
        path = self.path(planned, number)
        record = {"schema_version": 1, "planned_run_id": planned, "attempt_number": number,
                  "schedule_row": row, "status": "planned", "created_unix": now(),
                  "host": socket.gethostname(), "pid": os.getpid(), "command_line": command_line,
                  "provenance_sha256": provenance_sha256, "retry_reason": retry_reason,
                  "history": [{"status": "planned", "unix": now()}]}
        # Exclusive create: two processes can never both own one attempt number.
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(descriptor)
        atomic_write_json(path, record)
        return path

    def update(self, path, status, **fields):
        record = json.loads(Path(path).read_text("utf-8"))
        current = record["status"]
        if current in TERMINAL:
            raise TerminalRecordError("{} is terminal ({})".format(path, current))
        require(status in ALLOWED[current], "illegal transition {} -> {}".format(current, status))
        protected = {"planned_run_id", "attempt_number", "schedule_row", "created_unix", "history"}
        require(not protected & set(fields), "protected fields cannot be updated")
        record.update(fields)
        record["status"] = status
        record["history"].append({"status": status, "unix": now()})
        atomic_write_json(path, record)
        return record

    def reconcile_interrupted(self, reason):
        """Mark orphaned open records `interrupted`. Run once before a process starts work."""
        changed = []
        for path in sorted(self.root.glob("*.attempt[0-9][0-9].json")):
            record = json.loads(path.read_text("utf-8"))
            if record["status"] in OPEN:
                self.update(path, "interrupted", interruption_reason=reason)
                changed.append(path.name)
        return changed

    def records(self):
        return [json.loads(path.read_text("utf-8"))
                for path in sorted(self.root.glob("*.attempt[0-9][0-9].json"))]

    def checksums(self):
        return {path.name: file_sha(path) for path in sorted(self.root.iterdir()) if path.is_file()}
