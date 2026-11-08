#!/usr/bin/env python3
"""Protocol B schedule generator: 800 rows per model (400 R22 + 400 R25), paired by frozen identity. Python 3.8+.

Reads ONE model-native state manifest and emits that model's schedule (02 B3, B8). It generates no state, runs no
simulator and launches nothing. It fails closed on any malformed manifest. The output is exclusive-create.

    make_schedule_v2.py --model pi05 --manifest STATE_MANIFEST.json --output SCHEDULE.json \
        [--exclude-manifest OTHER.json ...]

Expected manifest (the v2 `generate_states.py` format, plus `model`, keyed by model as 02 B3 requires):
    {"schema_version": 1, "cohort": "cross_policy_confirmation_v2", "model": "pi05", "state_count": 200,
     "states": [{"key", "suite", "task_id", "state_slot", "draw", "seed", "state_length", "state_sha256",
                 "task_bddl_sha256", "task_language"}, ...],
     "states_sha256": sha(states), "initial_states_npz_sha256": ..., "generator_sha256": ..., "versions": {...}}
The model-keyed state generator does not exist yet (02 B3, "tooling change owed"); this reader defines what it
must produce.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3common import (ATTEMPTS_PER_MODEL, CONDITIONS, CONFIRMATION_POPULATION, MODELS, PROTOCOL_B,  # noqa: E402
                      file_sha, identity_id, sha, state_key)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
STATE_FIELDS = ("key", "suite", "task_id", "state_slot", "draw", "seed", "state_length", "state_sha256",
                "task_bddl_sha256", "task_language")


class ManifestError(ValueError):
    """The manifest is malformed. Nothing is emitted."""


def fail(message):
    raise ManifestError(message)


def validate_manifest(document, model, population=CONFIRMATION_POPULATION):
    """Return the states keyed by state key, or raise ManifestError. Every check fails closed."""
    if not isinstance(document, dict):
        fail("manifest is not an object")
    if document.get("schema_version") != 1:
        fail("unsupported schema_version {!r}".format(document.get("schema_version")))
    if document.get("cohort") != population.cohort:
        fail("manifest cohort {!r} is not {!r}".format(document.get("cohort"), population.cohort))
    if document.get("model") != model:
        fail("manifest model {!r} is not {!r}; populations are model-native (N1)".format(document.get("model"), model))
    states = document.get("states")
    expected_keys = population.state_keys()
    if not isinstance(states, list) or len(states) != len(expected_keys):
        fail("expected {} states".format(len(expected_keys)))
    if document.get("state_count") != len(expected_keys):
        fail("state_count disagrees with the population")
    if document.get("states_sha256") != sha(states):
        fail("states_sha256 does not match the states list")
    for field in ("initial_states_npz_sha256", "generator_sha256"):
        if not HEX64.match(str(document.get(field, ""))):
            fail("missing or malformed {}".format(field))
    if not isinstance(document.get("versions"), dict) or not document["versions"]:
        fail("versions missing: stack provenance is required")
    by_key, hashes = {}, set()
    for row in states:
        if not isinstance(row, dict) or any(f not in row for f in STATE_FIELDS):
            fail("state row lacks a required field: {!r}".format(row))
        if row["suite"] not in population.suites or row["task_id"] not in population.task_ids \
                or row["state_slot"] not in population.state_slots:
            fail("state outside the population: {}".format(row["key"]))
        if row["key"] != state_key(row["suite"], row["task_id"], row["state_slot"]):
            fail("state key does not match its fields: {}".format(row["key"]))
        if row["key"] in by_key:
            fail("duplicate state key {}".format(row["key"]))
        for field in ("state_sha256", "task_bddl_sha256"):
            if not HEX64.match(str(row[field])):
                fail("malformed {} for {}".format(field, row["key"]))
        if row["state_sha256"] in hashes:
            fail("duplicate state hash at {}: two slots hold the same state".format(row["key"]))
        if not isinstance(row["state_length"], int) or row["state_length"] <= 0:
            fail("bad state_length for {}".format(row["key"]))
        if not isinstance(row["task_language"], str) or not row["task_language"]:
            fail("missing task_language for {}".format(row["key"]))
        hashes.add(row["state_sha256"])
        by_key[row["key"]] = row
    if set(by_key) != set(expected_keys):
        fail("state keys do not cover the population exactly")
    # One task, one BDDL file, one language string, across its five slots.
    for suite in population.suites:
        for task_id in population.task_ids:
            rows = [by_key[state_key(suite, task_id, k)] for k in population.state_slots]
            if len({r["task_bddl_sha256"] for r in rows}) != 1 or len({r["task_language"] for r in rows}) != 1:
                fail("inconsistent task description across slots of {} task {}".format(suite, task_id))
    return by_key


def validate_manifest_binding(schedule, document, model, population=CONFIRMATION_POPULATION):
    """Check that every scheduled state's bytes/task/provenance match its native manifest.

    Schedule population, pairing and ordering remain the separate schedule validator's
    responsibility. A file digest reference alone cannot validate redundant row data.
    """
    by_key = validate_manifest(document, model, population)
    if schedule.get("model") != model or schedule.get("state_binding") != "BOUND":
        fail("schedule model or state binding mismatch")
    for scheduled, native in (
        ("state_manifest_states_sha256", "states_sha256"),
        ("initial_states_npz_sha256", "initial_states_npz_sha256"),
        ("state_generator_sha256", "generator_sha256"),
    ):
        if schedule.get(scheduled) != document[native]:
            fail("schedule {} differs from the native manifest".format(scheduled))
    rows = schedule.get("schedule")
    if not isinstance(rows, list) or not rows:
        fail("schedule rows missing from manifest binding check")
    for row in rows:
        if not isinstance(row, dict):
            fail("schedule row is not an object")
        state = by_key.get(row.get("state_key"))
        if state is None:
            fail("scheduled state key is absent from the native manifest")
        ident = row.get("episode_identity")
        if not isinstance(ident, dict) or any(
                ident.get(key) != state[key] for key in ("suite", "task_id", "state_slot")):
            fail("schedule identity differs from the native manifest")
        for field in ("state_sha256", "task_bddl_sha256", "task_language"):
            if row.get(field) != state[field]:
                fail("schedule {} differs from the native manifest".format(field))
        expected = {key: state[key] for key in ("draw", "seed", "state_length")}
        if row.get("state_provenance") != expected:
            fail("schedule state provenance differs from the native manifest")
    return by_key


def check_disjoint(by_key, other_documents):
    """No state hash may appear in another population (the other model's, or the diagnostic population)."""
    ours = {row["state_sha256"] for row in by_key.values()}
    for other in other_documents:
        theirs = {row.get("state_sha256") for row in other.get("states", []) if isinstance(row, dict)}
        collision = ours & theirs
        if collision:
            fail("state hash shared with an excluded population: {}".format(sorted(collision)[0]))


def identities(population):
    return [{"suite": s, "task_id": t, "state_slot": k, "policy_seed": p} for s in population.suites
            for t in population.task_ids for k in population.state_slots for p in population.policy_seeds]


def order_key(population, model):
    return hashlib.sha256((population.cohort + "\x00" + model + "\x00order").encode("utf-8")).hexdigest()


def build_schedule(document, model, population=CONFIRMATION_POPULATION, exclude=()):
    """Return the schedule document. Deterministic: the same manifest gives the same bytes."""
    if model not in MODELS:
        fail("unknown model {!r}".format(model))
    by_key = validate_manifest(document, model, population)
    check_disjoint(by_key, exclude)
    key = order_key(population, model)
    idents = identities(population)
    # 02 B8: keyed identity permutation.
    idents.sort(key=lambda i: sha([key, identity_id(i)]))
    # 02 B8: within each task, the identities sorted by sha(model, identity, "cond"); the first half run R22 first.
    first = {}
    for suite in population.suites:
        for task_id in population.task_ids:
            group = [i for i in idents if i["suite"] == suite and i["task_id"] == task_id]
            group.sort(key=lambda i: sha([model, identity_id(i), "cond"]))
            half = len(group) // 2
            for index, ident in enumerate(group):
                first[identity_id(ident)] = "R22" if index < half else "R25"
    rows = []
    for pair_index, ident in enumerate(idents):
        state = by_key[state_key(ident["suite"], ident["task_id"], ident["state_slot"])]
        ident_id = identity_id(ident)
        order = ("R22", "R25") if first[ident_id] == "R22" else ("R25", "R22")
        for position, condition in enumerate(order):
            rows.append({
                "planned_run_id": "{}.{}.{}.{}".format(population.cohort, model, ident_id, condition),
                "schedule_index": len(rows), "pair_index": pair_index, "pair_position": position,
                "block_id": "{}.{}".format(model, ident_id), "model": model, "cohort": population.cohort,
                "condition_id": condition, "d": CONDITIONS[condition]["d"], "h": CONDITIONS[condition]["h"],
                "episode_identity": dict(ident), "identity_id": ident_id,
                "state_key": state["key"], "state_sha256": state["state_sha256"],
                "state_provenance": {"draw": state["draw"], "seed": state["seed"],
                                     "state_length": state["state_length"]},
                "task_bddl_sha256": state["task_bddl_sha256"], "task_language": state["task_language"]})
    validate_schedule(rows, population, model)
    return {"schema_version": 1, "protocol": PROTOCOL_B, "cohort": population.cohort, "model": model,
            "attempt_count": len(rows), "identity_count": len(idents), "state_binding": "BOUND",
            "state_manifest_states_sha256": document["states_sha256"],
            "initial_states_npz_sha256": document["initial_states_npz_sha256"],
            "state_generator_sha256": document["generator_sha256"],
            "schedule_generator_sha256": file_sha(__file__), "order_key": key,
            "schedule_sha256": sha(rows), "schedule": rows}


def validate_schedule(rows, population, model):
    """Independent re-check of the emitted rows: counts, uniqueness, pairing, adjacency, balance."""
    n_ident = population.identity_count()
    if len(rows) != 2 * n_ident:
        fail("expected {} rows".format(2 * n_ident))
    if population.reserved and len(rows) != ATTEMPTS_PER_MODEL:
        fail("confirmation schedule must have 800 rows")
    if len({r["planned_run_id"] for r in rows}) != len(rows):
        fail("duplicate planned_run_id")
    by_condition = {c: [r for r in rows if r["condition_id"] == c] for c in CONDITIONS}
    for condition, subset in by_condition.items():
        if len(subset) != n_ident or len({r["identity_id"] for r in subset}) != n_ident:
            fail("condition {} does not cover every identity exactly once".format(condition))
    expected = {identity_id(i) for i in identities(population)}
    if {r["identity_id"] for r in rows} != expected:
        fail("identities do not match the population")
    first_counts = {}
    for index in range(0, len(rows), 2):
        a, b = rows[index], rows[index + 1]
        if a["identity_id"] != b["identity_id"] or {a["condition_id"], b["condition_id"]} != set(CONDITIONS):
            fail("pair at rows {} and {} is not one identity under both conditions".format(index, index + 1))
        if a["state_sha256"] != b["state_sha256"] or a["episode_identity"] != b["episode_identity"]:
            fail("pair partners differ in state or identity")
        if a["model"] != model or b["model"] != model:
            fail("row model mismatch")
        suite = a["episode_identity"]["suite"]
        first_counts.setdefault(suite, {"R22": 0, "R25": 0})[a["condition_id"]] += 1
        for row in (a, b):
            if (row["d"], row["h"]) != (CONDITIONS[row["condition_id"]]["d"], CONDITIONS[row["condition_id"]]["h"]):
                fail("unexpected d or h")
    per_suite = n_ident // len(population.suites)
    for suite, counts in first_counts.items():
        if counts["R22"] != per_suite // 2 or counts["R25"] != per_suite - per_suite // 2:
            fail("condition order not balanced in {}: {}".format(suite, counts))
    for index, row in enumerate(rows):
        if row["schedule_index"] != index:
            fail("schedule_index out of order")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="another population whose state hashes must not appear (other model, diagnostics)")
    args = parser.parse_args()
    try:
        document = json.loads(args.manifest.read_text("utf-8"))
        exclude = [json.loads(p.read_text("utf-8")) for p in args.exclude_manifest]
        result = build_schedule(document, args.model, exclude=exclude)
    except (ManifestError, ValueError, KeyError, TypeError) as error:
        raise SystemExit("REFUSED: {}".format(error))
    result["state_manifest_file_sha256"] = file_sha(args.manifest)
    if args.output.exists():
        raise SystemExit("REFUSED: output exists; a reviewed schedule is never replaced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(args.output), "model": result["model"], "attempt_count": result["attempt_count"],
                      "schedule_sha256": result["schedule_sha256"], "file_sha256": file_sha(args.output)}))


if __name__ == "__main__":
    main()

