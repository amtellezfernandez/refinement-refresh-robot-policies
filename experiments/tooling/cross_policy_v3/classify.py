"""Protocol B attempt classes (02 B5) and the per-row final class (02 B7). Python 3.8+, stdlib only.

Model adapters raise the typed exceptions below. Anything else is classified by whether the attempt's
first-policy-query marker exists: before it, INFRA_PRE; after it, INFRA_POST. A scientific outcome is never
produced from an exception, and an exception is never turned into a scientific outcome.
"""

SCI_SUCCESS = "SCI_SUCCESS"
SCI_FAILURE = "SCI_FAILURE"
INFRA_PRE = "INFRA_PRE"
INFRA_POST = "INFRA_POST"
INSTRUMENT_INCIDENT = "INSTRUMENT_INCIDENT"
PROTOCOL_STOP = "PROTOCOL_STOP"
SCIENTIFIC = (SCI_SUCCESS, SCI_FAILURE)
ALL_CLASSES = (SCI_SUCCESS, SCI_FAILURE, INFRA_PRE, INFRA_POST, INSTRUMENT_INCIDENT, PROTOCOL_STOP)
# Classes after which the whole model run stops (02 B5, B9).
STOPPING = (INSTRUMENT_INCIDENT, PROTOCOL_STOP)

# Final classes of a planned attempt (one schedule row), after retries.
FINAL_SCIENTIFIC = SCIENTIFIC
MISSING = "MISSING"
INVALIDATED = "INVALIDATED"  # a second started execution of a counted attempt (02 B7)

SUCCESS_TERMINATIONS = ("success",)
FAILURE_TERMINATIONS = ("task_horizon", "terminated", "truncated", "nonfinite_action_valid_inputs")


class ClassifiedFailure(Exception):
    attempt_class = None

    def __init__(self, message, **details):
        Exception.__init__(self, message)
        self.details = details


class InfraPre(ClassifiedFailure):
    """Failure before the first policy query (env build, restore, reset-hash mismatch, server unreachable)."""
    attempt_class = INFRA_PRE


class InfraPost(ClassifiedFailure):
    """Infrastructure failure after the first policy query that is not an instrument incident."""
    attempt_class = INFRA_POST


class InstrumentIncident(ClassifiedFailure):
    """Attestation failure, broken action contract, invalid input, or unclear provenance.

    The action concerned has not been applied. The model run stops for review.
    """
    attempt_class = INSTRUMENT_INCIDENT


class ProtocolStop(ClassifiedFailure):
    """Unsupported pi0.5 refinement setting."""
    attempt_class = PROTOCOL_STOP


def classify_outcome(outcome):
    """A completed episode's outcome dict -> SCI_SUCCESS or SCI_FAILURE. Anything malformed raises."""
    if not isinstance(outcome, dict) or not isinstance(outcome.get("success"), bool):
        raise ValueError("outcome lacks a boolean success")
    reason = outcome.get("termination_reason")
    if outcome["success"]:
        if reason not in SUCCESS_TERMINATIONS:
            raise ValueError("success with termination_reason {!r}".format(reason))
        return SCI_SUCCESS
    if reason not in FAILURE_TERMINATIONS:
        raise ValueError("failure with termination_reason {!r}".format(reason))
    return SCI_FAILURE


def classify_exception(error, first_query_started):
    """Exception from an attempt -> attempt class. Untyped errors are split by the first-query marker."""
    if isinstance(error, ClassifiedFailure):
        if isinstance(error, InfraPre) and first_query_started:
            # An adapter cannot call a post-query failure "pre". The marker is the record that decides.
            return INFRA_POST
        return error.attempt_class
    name = type(error).__name__
    if name == "ResetStateMismatch":  # attempt_loop.check_reset, raised before any policy query
        return INFRA_POST if first_query_started else INFRA_PRE
    return INFRA_POST if first_query_started else INFRA_PRE


def interrupted_class(first_query_started):
    """An attempt record left open by a dead process."""
    return INFRA_POST if first_query_started else INFRA_PRE


def final_class(attempt_classes):
    """The final class of one planned attempt from its ordered attempt records' classes (02 B7).

    - one scientific record -> that class;
    - INFRA_PRE then one more record -> that record's class if scientific, else MISSING;
    - INFRA_POST, or INFRA_PRE twice -> MISSING;
    - any other sequence (a second execution after a scientific or post-query record, three records) -> INVALIDATED.
    Returns None while a permitted retry is still pending (exactly one INFRA_PRE record).
    INSTRUMENT_INCIDENT and PROTOCOL_STOP are returned as themselves: the run has stopped.
    """
    for value in attempt_classes:
        if value not in ALL_CLASSES:
            raise ValueError("unknown attempt class {!r}".format(value))
    if not attempt_classes:
        return None
    first = attempt_classes[0]
    if len(attempt_classes) == 1:
        if first in SCIENTIFIC or first in STOPPING:
            return first
        if first == INFRA_POST:
            return MISSING
        return None  # INFRA_PRE: one retry allowed
    if len(attempt_classes) == 2 and first == INFRA_PRE:
        second = attempt_classes[1]
        if second in SCIENTIFIC or second in STOPPING:
            return second
        return MISSING
    return INVALIDATED


def outcome_value(final):
    """Y for the analysis: 1, 0, or None (missing). Stopping classes have no outcome."""
    return {SCI_SUCCESS: 1, SCI_FAILURE: 0}.get(final)
