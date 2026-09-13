#!/usr/bin/env python3
"""Validate v0.2 structure and report completeness of DECLARED resources.

Dependency-free. The schema walker implements the vocabulary in the checked-in
schema, with semantic graph checks below. Neither check proves that events were
not omitted or that measurements are accurate. No resource aggregation or unit
conversion is inferred.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("post_training_trace.schema.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_json(value, path="$") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{path}: nonfinite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]")


def schema_check(value, spec, root, path="$") -> None:
    """Check the exact JSON Schema vocabulary used by our local schema."""
    if "$ref" in spec:
        target = root
        require(spec["$ref"].startswith("#/"), "only local schema references supported")
        for part in spec["$ref"][2:].split("/"):
            target = target[part]
        schema_check(value, target, root, path)
    type_checks = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "null": lambda x: x is None,
    }
    if "type" in spec:
        kinds = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        require(any(type_checks[kind](value) for kind in kinds), f"{path}: expected {kinds}")
    if "const" in spec:
        require(value == spec["const"], f"{path}: expected {spec['const']!r}")
    if "enum" in spec:
        require(value in spec["enum"], f"{path}: invalid enum {value!r}")
    if isinstance(value, dict):
        require(set(spec.get("required", [])) <= value.keys(), f"{path}: missing required field")
        props = spec.get("properties", {})
        for key, item in value.items():
            if key in props:
                schema_check(item, props[key], root, f"{path}.{key}")
            elif spec.get("additionalProperties", True) is False:
                raise ValueError(f"{path}: unexpected field {key!r}")
            elif isinstance(spec.get("additionalProperties"), dict):
                schema_check(item, spec["additionalProperties"], root, f"{path}.{key}")
    if isinstance(value, list):
        require(len(value) >= spec.get("minItems", 0), f"{path}: too few items")
        if spec.get("uniqueItems"):
            require(all(item not in value[:index] for index, item in enumerate(value)), f"{path}: duplicate item")
        for index, item in enumerate(value):
            if "items" in spec:
                schema_check(item, spec["items"], root, f"{path}[{index}]")
    if isinstance(value, str):
        require(len(value) >= spec.get("minLength", 0), f"{path}: empty string")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in spec:
        require(value >= spec["minimum"], f"{path}: below minimum")
    if "oneOf" in spec:
        matches = 0
        for alternative in spec["oneOf"]:
            try:
                schema_check(value, alternative, root, path)
                matches += 1
            except ValueError:
                pass
        require(matches == 1, f"{path}: status/amount/derivation combination invalid")


def has(states, *, carrier=None, role=None) -> bool:
    return any((carrier is None or state["carrier"] == carrier)
               and (role is None or role in state["roles"]) for state in states)


def validate_graph(graph: dict) -> dict:
    finite_json(graph)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_check(graph, schema, schema)
    states = {state["id"]: state for state in graph["states"]}
    require(len(states) == len(graph["states"]), "state identifiers must be unique")
    event_ids = [event["id"] for event in graph["events"]]
    require(len(set(event_ids)) == len(event_ids), "event identifiers must be unique")
    require(set(event_ids).isdisjoint(states), "state and event identifiers must be disjoint")
    require(set(graph["pretrained_roots"]) <= states.keys(), "unresolved pretrained root")
    require(all(states[key]["carrier"] == "MODEL" for key in graph["pretrained_roots"]), "pretrained roots must be MODEL states")
    producer, edges = {}, {key: set() for key in states}
    incomplete, not_applicable = [], []
    quantitative = 0
    for event in graph["events"]:
        event_id, op = event["id"], event["operator"]
        for key in event["inputs"] + event["outputs"] + event["specifications"]:
            require(key in states, f"{event_id}: unresolved state {key}")
        inputs = [states[key] for key in event["inputs"]]
        outputs = [states[key] for key in event["outputs"]]
        specs = [states[key] for key in event["specifications"]]
        require(inputs or op == "COLLECT", f"{event_id}: operator has no input")
        require(all(state["carrier"] == "SPECIFICATION" for state in specs), f"{event_id}: invalid specification")
        for key in event["outputs"]:
            require(key not in producer, f"{event_id}: multiple producers for {key}")
            require(key not in graph["pretrained_roots"], f"{event_id}: cannot overwrite pretrained root")
            producer[key] = event_id
            for source in event["inputs"] + event["specifications"]:
                edges[source].add(key)
        if op == "UPDATE":
            require(has(inputs, carrier="MODEL") and has(inputs, carrier="RECORDS") and has(outputs, carrier="MODEL"), f"{event_id}: UPDATE needs MODEL and RECORDS inputs and MODEL output")
        elif op == "RETRIEVE":
            memories = [state for state in inputs if state["carrier"] == "RECORDS" and "MEMORY" in state["roles"]]
            require(memories and all(state["lifetime"] != "EVENT" for state in memories) and has(inputs, role="QUERY") and has(outputs, carrier="WORKSPACE"), f"{event_id}: RETRIEVE needs persistent MEMORY, QUERY and WORKSPACE output")
        elif op == "IMAGINE":
            require(has(inputs, carrier="MODEL", role="DYNAMICS") and has(outputs, role="TRAJECTORY"), f"{event_id}: IMAGINE needs DYNAMICS and TRAJECTORY")
        elif op == "PROPOSE":
            require(has(inputs, carrier="MODEL", role="POLICY") and (has(outputs, role="CANDIDATE") or has(outputs, role="ACTION")), f"{event_id}: PROPOSE needs POLICY and CANDIDATE/ACTION")
        elif op == "EVALUATE":
            critic = any(has(inputs, carrier="MODEL", role=role) for role in ("VALUE", "REWARD", "DETECTOR"))
            require((critic or specs) and has(outputs, role="SCORE"), f"{event_id}: EVALUATE needs critic/specification and SCORE")
        elif op == "SELECT":
            require(has(inputs, role="CANDIDATE") and has(inputs, role="SCORE") and (has(outputs, role="ACTION") or has(outputs, role="CANDIDATE")), f"{event_id}: SELECT needs CANDIDATE, SCORE and selection output")
        elif op == "EXECUTE":
            require((has(inputs, role="ACTION") or has(inputs, role="TRAJECTORY")) and has(inputs, carrier="PHYSICAL") and has(outputs, carrier="PHYSICAL"), f"{event_id}: EXECUTE needs action and environment transition")
            require("execution_environment" in event and "changes_physical_world" in event, f"{event_id}: EXECUTE must declare real/simulated effects")
            require(event["changes_physical_world"] == (event["execution_environment"] == "real"), f"{event_id}: inconsistent physical-world effect")
        elif op == "WRITE":
            memories = [state for state in outputs if state["carrier"] == "RECORDS" and "MEMORY" in state["roles"]]
            require(memories and all(state["lifetime"] != "EVENT" for state in memories), f"{event_id}: WRITE needs persistent MEMORY output")
        elif op == "COLLECT":
            require(has(outputs, carrier="RECORDS"), f"{event_id}: COLLECT needs RECORDS output")
        resource_names = [resource["name"] for resource in event["resources"]]
        require(len(set(resource_names)) == len(resource_names), f"{event_id}: duplicate resource name")
        require(set(resource_names) <= set(event["required_resources"]), f"{event_id}: resource missing from declared requirements")
        by_name = {resource["name"]: resource for resource in event["resources"]}
        for name in event["required_resources"]:
            resource = by_name.get(name)
            if resource is None or resource["status"] == "unknown":
                incomplete.append({"event": event_id, "resource": name, "reason": "missing" if resource is None else "unknown"})
            elif resource["status"] == "not_applicable":
                not_applicable.append({"event": event_id, "resource": name})
            else:
                quantitative += 1
    indegree = {key: 0 for key in states}
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    ready = deque(key for key, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for target in edges[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    require(visited == len(states), "version lineage contains a cycle or in-place update")
    return {"structurally_valid": True, "complete_for_declared_resources": not incomplete,
            "quantitative_resource_rows": quantitative, "incomplete_resources": incomplete,
            "not_applicable_resources": not_applicable,
            "limitation": "Does not certify omitted events, measurement accuracy, boundary sufficiency, or unit conversion."}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def validate(path: Path) -> dict:
    return validate_graph(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path(__file__).parent / "examples/hybrid_system_trace.json")
    parser.add_argument("--require-complete", action="store_true", help="also reject missing/unknown declared resource rows")
    args = parser.parse_args()
    try:
        report = validate(args.path)
        print(json.dumps(report, indent=2))
        raise SystemExit(2 if args.require_complete and not report["complete_for_declared_resources"] else 0)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"INVALID: {error}\n")
