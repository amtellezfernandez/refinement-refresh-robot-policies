"""Execute a frozen, deterministic offline accounting-mutation experiment.

No GPU jobs, model calls, or robot rollouts occur. Original ledgers are read only.
First freeze protocol and input digests; then run and independently check results:
  python run_experiment.py --freeze
  python run_experiment.py --run
  python run_experiment.py --check
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LEDGER_ROOT = ROOT.parent / "real_smolvla"
sys.path.insert(0, str(LEDGER_ROOT))

import jsonschema
import numpy as np
from ledger_arrays import array_record
from result_metrics import completed_outcomes
from validate_ledger import validate_completed

LAYERS = ["syntax", "structure", "aggregates", "full_ledger"]
LAYER_LABELS = {"syntax": "JSON syntax", "structure": "+ Structure",
                "aggregates": "+ Aggregates", "full_ledger": "+ Full ledger"}

# This fixed list is written to protocol.json BEFORE validation trials run.
# No mutations are selected or dropped according to their observed outcomes.
CASES = [
    ("untouched", "control", "Untouched archived artifact", "No mutation; accepted in-band data does not establish completeness outside its scope."),
    ("invalid_json", "observable_defect", "Invalid JSON", "Remove final closing brace; violates JSON grammar."),
    ("missing_environment_event", "observable_defect", "Missing environment event", "Remove final consumer event but retain producer stream and selection counter."),
    ("duplicate_environment_event", "observable_defect", "Duplicate environment event", "Append final consumer event without changing independent streams/counters."),
    ("sampler_count_drift", "observable_defect", "Independent counter mismatch", "Increment measured sampler counter only."),
    ("stale_action_hash", "observable_defect", "Action with stale hash", "Change first consumer action value while retaining original hash."),
    ("consumer_action_rewrite", "observable_defect", "Producer/consumer action mismatch", "Change first consumer action and recompute its valid hash; producer stream stays original."),
    ("identity_drift", "observable_defect", "Episode identity mismatch", "Increment episode task ID only; inference identities and task outcomes stay original."),
    ("outcome_rate_drift", "observable_defect", "Outcome aggregate mismatch", "Replace overall success percentage with the opposite binary outcome; task/group outcomes stay original."),
    ("incomplete_result", "observable_defect", "Missing completed-result wrapper", "Remove overall result; remaining ledger alone cannot establish completed evaluator output."),
    ("reordered_environment_events", "observable_defect", "Reordered steps", "Swap first two consumer events; counts and action hashes stay valid."),
    ("balanced_duplicate_inference", "observable_defect", "Duplicate chunk; balanced counts", "Append a measured inference and increment both counters; aggregate counts agree but schedule duplicates."),
    ("missing_timing_unit", "observable_defect", "Missing timing field/unit", "Rename first inference cuda_ms to duration, losing the declared clock/unit field."),
    ("interleaved_warmup", "observable_defect", "Warm-up after measured event", "Move first warm-up event to end; all event and counter totals stay unchanged."),
    ("joint_action_rewrite", "unobservable_scenario", "Joint action rewrite", "Change both first action streams and recompute hashes. Without an external expected-action reference, an internally consistent record cannot certify which action occurred."),
    ("preprocessing_coverage_query", "unobservable_scenario", "Unlogged preprocessing coverage", "Unchanged bytes; ask whether all preprocessing was accounted for. No independent expected-stage manifest or preprocessing cost is available. This is an unknown coverage query, not an invented measured omission."),
    ("rendering_coverage_query", "unobservable_scenario", "Unlogged rendering coverage", "Unchanged bytes; ask whether rendering costs were fully included. No independent stage-level rendering record is available. This is an unknown coverage query, not an invented measured omission."),
]


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_bytes(payload):
    return hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()


def frozen_inputs():
    paths = [LEDGER_ROOT / "results/infrastructure_v2" / name
             for name in ("h1.json", "h5.json", "h10.json", "h10_repeat.json")]
    code = [LEDGER_ROOT / name for name in ("validate_ledger.py", "result_metrics.py", "ledger_arrays.py")]
    return {
        "sources": [{"path": p.relative_to(ROOT.parent).as_posix(), "sha256_lf_normalized": digest_bytes(p.read_bytes())}
                    for p in paths],
        "instrument_code": {p.name: digest_bytes(p.read_bytes()) for p in code},
        "runner_sha256_lf_normalized": digest_bytes(Path(__file__).read_bytes()),
        "structure_schema_sha256_lf_normalized": digest_bytes((ROOT / "structure.schema.json").read_bytes()),
        "case_manifest": [{"id": x[0], "ground_truth": x[1], "label": x[2], "intervention": x[3]} for x in CASES],
        "layers": LAYERS,
    }


def freeze():
    protocol = {
        "protocol_version": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "design": "4 archived v2 sources x 17 fixed cases x 4 cumulative validation layers; deterministic exhaustive cases, no random sampling.",
        "primary_estimand": "Observed rejections of the 52 predeclared in-band invalid artifacts, by cumulative layer; descriptive finite-suite counts only.",
        "controls": "4 untouched artifacts; report false rejections separately.",
        "unknowns": "12 unobservable-scenario probes excluded from detection denominators; report technical verdict and epistemic unknown separately.",
        "layer_contract": {
            "syntax": "Strict JSON parse; rejects nonstandard NaN/Infinity tokens.",
            "structure": "Syntax plus the archived structural JSON Schema: required fields, local types/ranges, no cross-record reconciliation.",
            "aggregates": "Structure plus task/group/overall outcomes, episode count, stream lengths, per-phase inference list/count agreement and warm-up total.",
            "full_ledger": "Aggregates plus existing validate_completed: hashes, stream action equality, identities, seeds, chronology and chunk schedule.",
        },
        "legacy_applicability": "The aggregate layer is an explicit ablation of v2 evidence, not a claim that historical v1 stored equivalent streams. V1 originals are excluded because they cannot support action/identity mutation trials.",
        "claim_limits": [
            "Constructed mutations were chosen with knowledge of validator requirements; this evaluates specified mechanisms, not natural error prevalence.",
            "Four source artifacts share one task/state and include a replay; they are stress-test contexts, not independent robot replicates.",
            "No confidence interval, human-reader benefit, policy quality, robustness under physical perturbations, or end-to-end lifecycle completeness is estimated.",
            "Jointly consistent alterations and out-of-boundary stages require an external oracle; a pass is not proof of truthful or complete collection.",
        ],
        **frozen_inputs(),
    }
    with (ROOT / "protocol.json").open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
        handle.write("\n")
    print("Frozen protocol and source/code digests before executing validation trials.")


def load_protocol():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    for key, current in frozen_inputs().items():
        if protocol[key] != current:
            raise ValueError(f"Frozen input drift: {key}; preserve this experiment and create a new protocol version")
    return protocol


def mutate(text, case):
    if case in ("untouched", "preprocessing_coverage_query", "rendering_coverage_query"):
        return text
    if case == "invalid_json":
        return text.rstrip()[:-1]
    data = json.loads(text)
    ledger = data["robopad_ledger"]
    episode = ledger["episodes"][0]
    consumer = episode["environment_steps"]
    if case == "missing_environment_event":
        consumer.pop()
    elif case == "duplicate_environment_event":
        consumer.append(copy.deepcopy(consumer[-1]))
    elif case == "sampler_count_drift":
        ledger["sampler_calls"]["measured"] += 1
    elif case in ("stale_action_hash", "consumer_action_rewrite", "joint_action_rewrite"):
        original = consumer[0]["action"]
        action = np.asarray(original["values"], dtype=original["dtype"])
        action.flat[0] += 0.125
        rewritten = array_record(action)
        if case == "stale_action_hash":
            rewritten["sha256"] = original["sha256"]
        consumer[0]["action"] = rewritten
        if case == "joint_action_rewrite":
            episode["producer_steps"][0]["action"] = copy.deepcopy(rewritten)
    elif case == "identity_drift":
        episode["identity"][1] += 1
    elif case == "outcome_rate_drift":
        data["overall"]["pc_success"] = 0. if data["overall"]["pc_success"] else 100.
    elif case == "incomplete_result":
        data.pop("overall")
    elif case == "reordered_environment_events":
        consumer[0], consumer[1] = consumer[1], consumer[0]
    elif case == "balanced_duplicate_inference":
        event = next(x for x in ledger["inferences"] if x["phase"] == "measured")
        ledger["inferences"].append(copy.deepcopy(event))
        for key in ("chunk_calls", "sampler_calls"):
            ledger[key]["measured"] += 1
    elif case == "missing_timing_unit":
        event = ledger["inferences"][0]
        event["duration"] = event.pop("cuda_ms")
    elif case == "interleaved_warmup":
        ledger["inferences"].append(ledger["inferences"].pop(0))
    else:
        raise ValueError(f"Unknown mutation {case}")
    return canonical(data)


def reject_nonstandard(value):
    raise ValueError(f"Nonstandard JSON numeric token: {value}")


def check_aggregates(data):
    completed_outcomes(data)
    ledger = data["robopad_ledger"]
    if data["overall"]["n_episodes"] != len(ledger["episodes"]):
        raise ValueError("episode total disagrees")
    for episode in ledger["episodes"]:
        if not (len(episode["producer_steps"]) == len(episode["environment_steps"]) == episode["select_calls"]):
            raise ValueError("aggregate producer/consumer/selection counts disagree")
    for phase in ("warmup", "measured"):
        count = sum(x["phase"] == phase for x in ledger["inferences"])
        if not (count == ledger["chunk_calls"][phase] == ledger["sampler_calls"][phase]):
            raise ValueError("aggregate phase/event/chunk/sampler counts disagree")
    if ledger["warmup_passes"] != ledger["chunk_calls"]["warmup"]:
        raise ValueError("aggregate warm-up count disagrees")


def evaluate(text, layer, schema):
    stage = "syntax"
    try:
        data = json.loads(text, parse_constant=reject_nonstandard)
        if layer != "syntax":
            stage = "structure"
            error = next(schema.iter_errors(data), None)
            if error is not None:
                return {"technical_verdict": "rejected", "rejected_at": stage,
                        "reason": error.message, "error_path": list(error.absolute_path)}
        if layer in ("aggregates", "full_ledger"):
            stage = "aggregates"
            check_aggregates(data)
        if layer == "full_ledger":
            stage = "full_ledger"
            validate_completed(data)
        return {"technical_verdict": "accepted", "rejected_at": None, "reason": None}
    except (ValueError, json.JSONDecodeError) as error:
        return {"technical_verdict": "rejected", "rejected_at": stage, "reason": str(error)}
    except Exception as error:
        # A crashed check is recorded separately and never scored as detection.
        return {"technical_verdict": "validator_error", "rejected_at": stage,
                "reason": f"{type(error).__name__}: {error}"}


def execute(protocol):
    schema_data = json.loads((ROOT / "structure.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema_data)
    schema = jsonschema.Draft202012Validator(schema_data)
    trials = []
    for source in protocol["sources"]:
        source_path = ROOT.parent / source["path"]
        original_text = source_path.read_text(encoding="utf-8")
        for case, truth, label, intervention in CASES:
            payload = mutate(original_text, case)
            outcomes = {}
            for layer in LAYERS:
                result = evaluate(payload, layer, schema)
                verdict = result["technical_verdict"]
                if truth == "unobservable_scenario":
                    epistemic = "unknown"
                elif verdict == "validator_error":
                    epistemic = "validator_error"
                elif truth == "control":
                    epistemic = "accepted_control" if verdict == "accepted" else "false_rejection"
                else:
                    epistemic = "detected" if verdict == "rejected" else "not_detected"
                outcomes[layer] = {**result, "outcome": epistemic}
            trials.append({"trial_id": f"{source_path.stem}/{case}", "source": source,
                           "case": case, "label": label, "ground_truth": truth,
                           "intervention": intervention,
                           "payload_sha256_lf_normalized": digest_bytes(payload.encode()),
                           "payload_differs_from_source": payload != original_text,
                           "validation_results": outcomes})
    return {
        "experiment": "Deterministic offline ledger mutation robustness",
        "protocol_sha256": hashlib.sha256(canonical(protocol).encode()).hexdigest(),
        "run_type": "executed local validation of constructed artifacts; no new robot evaluation",
        "source_count": len(protocol["sources"]), "case_count": len(CASES),
        "trial_count": len(trials), "layer_checks": len(trials) * len(LAYERS),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "jsonschema")},
        "trials": trials,
    }


def summarize(report):
    rows = []
    for layer in LAYERS:
        counts = Counter(x["validation_results"][layer]["outcome"] for x in report["trials"])
        rows.append({"layer": layer,
                     "observable_defects": sum(x["ground_truth"] == "observable_defect" for x in report["trials"]),
                     "detected": counts["detected"], "not_detected": counts["not_detected"],
                     "controls": sum(x["ground_truth"] == "control" for x in report["trials"]),
                     "accepted_controls": counts["accepted_control"], "false_rejections": counts["false_rejection"],
                     "unknown_scenarios": counts["unknown"],
                     "validator_errors": sum(x["validation_results"][layer]["technical_verdict"] == "validator_error" for x in report["trials"])})
    by_case = []
    for case, truth, label, _ in CASES:
        trials = [x for x in report["trials"] if x["case"] == case]
        by_case.append({"case": case, "ground_truth": truth, "label": label, "n": len(trials),
                        "layers": {layer: dict(Counter(x["validation_results"][layer]["outcome"] for x in trials))
                                   for layer in LAYERS}})
    return {"protocol_sha256": report["protocol_sha256"], "source_count": report["source_count"],
            "trial_count": report["trial_count"], "layer_checks": report["layer_checks"],
            "interpretation": "Finite-suite descriptive counts; unknown scenarios excluded from defect denominator; sources are related, not independent rollouts.",
            "layers": rows, "by_case": by_case}


def latex_table(summary):
    lines = [r"% Generated by instrument_validation/run_experiment.py.",
             r"\begin{table}[t]", r"\centering\small",
             r"\caption{Constructed-defect coverage across four archived contexts. Counts are "
             r"rejected observable defects out of 52; all four untouched controls are accepted. "
             r"Full reconciliation rejects every designed observable defect. Twelve scenarios "
             r"requiring external evidence are excluded from rejection denominators.}",
             r"\label{tab:mutation-validation}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Validation & Defects rejected & Controls accepted & Unknown\\", r"\midrule"]
    for row in summary["layers"]:
        lines.append(f"{LAYER_LABELS[row['layer']]} & {row['detected']}/{row['observable_defects']} & "
                     f"{row['accepted_controls']}/{row['controls']} & {row['unknown_scenarios']}" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_or_check(outputs, check=False):
    for name, text in outputs.items():
        path = ROOT / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Result differs: {name}")
        else:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    protocol = load_protocol()
    report = execute(protocol)
    summary = summarize(report)
    outputs = {"trials.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
               "summary.json": json.dumps(summary, indent=2, allow_nan=False) + "\n",
               "mutation_results_table.tex": latex_table(summary)}
    write_or_check(outputs, check=args.check)
    print(f"Executed {report['trial_count']} constructed trials / {report['layer_checks']} checks on read-only archives.")
    for row in summary["layers"]:
        print(f"{row['layer']}: {row['detected']}/{row['observable_defects']} rejected; "
              f"{row['accepted_controls']}/{row['controls']} controls accepted; "
              f"{row['unknown_scenarios']} unknown; {row['validator_errors']} validator errors")


if __name__ == "__main__":
    main()
