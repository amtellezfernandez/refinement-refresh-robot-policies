"""pi0.5 per-request executed-step attestation (D5; 09 C4). Python 3.8+, stdlib only.

Shared by the attesting server (which builds attestations from callback counters) and the client (which
verifies each one before the first action of a chunk is applied). Verification raises InstrumentIncident;
nothing here ever produces a passing attestation from a request's own fields.
Malformed JSON field types are instrument incidents, not generic infrastructure
exceptions: the runner must stop before any further action/attempt. Integer
fields exclude booleans and float lookalikes; sequence state changes only on pass.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import InstrumentIncident  # noqa: E402
from v3common import canonical  # noqa: E402

SCHEMA = "pi05_step_attestation_v1"
FIELDS = ("schema", "request_id", "tick", "d_requested", "executed_head_evaluations", "time_values",
          "sequence_number", "startup_gate_sha256", "xla_flags")


def request_id(planned_run_id, attempt_number, tick):
    """sha256 of the canonical JSON of [planned_run_id, attempt_number, tick] (09 C4)."""
    return hashlib.sha256(canonical([planned_run_id, int(attempt_number), int(tick)])).hexdigest()


def f32(value):
    """Round a Python float to IEEE float32 (round-to-nearest-even), returned as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_hex(value):
    return struct.pack("<f", value).hex()


def expected_time_values(d):
    """The time operand visited at each Euler step: t0 = 1, t(k+1) = t(k) + (-1/d), in float32, k = 0..d-1.

    Each float32 sum is exact in double and then rounded once, so this equals float32 arithmetic. -1/d is
    rounded once from the double quotient. The recurrence follows the G02 faithful sampler P (bitwise equal to
    stock); it has not been re-read from pinned pi0.py (09 C4, check 5).
    """
    if not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive int")
    dt = f32(-1.0 / d)
    values, t = [], f32(1.0)
    for _ in range(d):
        values.append(t)
        t = f32(t + dt)
    return values


def attestation_digest(attestation):
    body = {k: attestation[k] for k in FIELDS}
    return hashlib.sha256(canonical(body)).hexdigest()


def build(request_id_value, tick, d_requested, head_evaluations, time_values, sequence_number,
          startup_gate_sha256, xla_flags):
    """Server side. head_evaluations and time_values must come from the compiled callbacks, never the request."""
    attestation = {"schema": SCHEMA, "request_id": request_id_value, "tick": int(tick),
                   "d_requested": int(d_requested), "executed_head_evaluations": int(head_evaluations),
                   "time_values": [f32_hex(float(v)) for v in time_values], "sequence_number": int(sequence_number),
                   "startup_gate_sha256": startup_gate_sha256, "xla_flags": xla_flags}
    attestation["attestation_sha256"] = attestation_digest(attestation)
    return attestation


class Verifier(object):
    """Client side. One instance per client process; it tracks the server's sequence numbers."""

    def __init__(self, startup_gate_sha256, xla_flags):
        if not startup_gate_sha256:
            raise InstrumentIncident("no startup gate record: the server has not certified its served object")
        self.startup_gate_sha256, self.xla_flags, self.last_sequence = startup_gate_sha256, xla_flags, None

    def verify(self, attestation, sent_request_id, sent_tick, d):
        """Raise InstrumentIncident unless every 09 C4 condition holds. Returns the attestation digest."""
        def incident(check, message):
            raise InstrumentIncident("attestation check {} failed: {}".format(check, message), check=check,
                                     attestation=attestation, request_id=sent_request_id, tick=sent_tick, d=d)
        if not isinstance(attestation, dict):
            incident(1, "missing")
        if attestation.get("schema") != SCHEMA or any(k not in attestation for k in FIELDS + ("attestation_sha256",)):
            incident(1, "unexpected schema or missing field")
        for name, check in (("tick", 2), ("d_requested", 3), ("executed_head_evaluations", 4)):
            if type(attestation[name]) is not int:
                incident(check, "{} must be a JSON integer".format(name))
        if attestation["request_id"] != sent_request_id or attestation["tick"] != sent_tick:
            incident(2, "bound to request {} tick {}, sent {} tick {}".format(
                attestation["request_id"], attestation["tick"], sent_request_id, sent_tick))
        if attestation["d_requested"] != d:
            incident(3, "d_requested {} != {}".format(attestation["d_requested"], d))
        if attestation["executed_head_evaluations"] != d:
            incident(4, "executed {} head evaluations, expected {}".format(attestation["executed_head_evaluations"], d))
        expected = sorted(f32_hex(v) for v in expected_time_values(d))
        if (not isinstance(attestation["time_values"], list)
                or any(type(value) is not str for value in attestation["time_values"])
                or sorted(attestation["time_values"]) != expected):
            incident(5, "visited time values differ from the float32 grid for d={}".format(d))
        if attestation["startup_gate_sha256"] != self.startup_gate_sha256:
            incident(6, "startup gate record differs from the one recorded at client start")
        if attestation["xla_flags"] != self.xla_flags:
            incident(6, "XLA flags differ from the certified configuration")
        sequence = attestation["sequence_number"]
        if type(sequence) is not int or (self.last_sequence is not None and sequence <= self.last_sequence):
            incident(7, "sequence number {} not after {}".format(sequence, self.last_sequence))
        if attestation["attestation_sha256"] != attestation_digest(attestation):
            incident(8, "digest mismatch")
        self.last_sequence = sequence
        return attestation["attestation_sha256"]


def to_json(attestation):
    return json.dumps(attestation, sort_keys=True)

