#!/usr/bin/env python3
"""Build the three frozen protocols and their schedules. Deterministic; no model code.

    python3 experiments/tooling/cross_policy/build_protocols.py          # write
    python3 experiments/tooling/cross_policy/build_protocols.py --check  # compare

Schedules bind each identity to its serialized state hash, which comes from
experiments/runs/<cohort>/states/state_manifest.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CONFIRMATION_SEEDS, LEROBOT_REVISION,  # noqa: E402
                    OBSERVATION_AGE_SEEDS, OPENPI_NORM_SHA256, OPENPI_REVISION, QUALITY_MARGIN_POINTS,
                    SCHEDULE_SEED, SMOLVLA_REVISION, STATE_SLOTS, SUITES, TASK_IDS, identity, identity_id, sha)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOLS = ROOT / "experiments/protocols"
RUNS = ROOT / "experiments/runs"
CONFIRMATION = "cross_policy_confirmation_v1"
AGE = "observation_age_confirmation_v1"
ASYNC = "asynchronous_confirmation_v1"

MODELS = {
    "smolvla": {
        "family": "SmolVLA (LeRobot, PyTorch)",
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "revision": SMOLVLA_REVISION,
        "model_safetensors_sha256": "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f",
        "runtime_source": {"repo": "huggingface/lerobot", "revision": LEROBOT_REVISION},
        "native": {"h": 1, "d": 10, "chunk_size": 50},
        "native_source": "Checkpoint config and existing runner requirement (run_paired.py:249).",
        "evaluation_path": "Pinned LeRobot SmolVLAPolicy + LiberoEnv, as in run_paired.py.",
    },
    "pi05": {
        "family": "pi0.5 (OpenPI, JAX)",
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero/",
        "normalization_asset": "assets/physical-intelligence/libero/norm_stats.json",
        "normalization_sha256": OPENPI_NORM_SHA256,
        "runtime_source": {"repo": "Physical-Intelligence/openpi", "revision": OPENPI_REVISION},
        "native": {"h": 5, "d": 10, "chunk_size": 10},
        "native_source": "examples/libero/main.py replan_steps=5; pi0.py sample_actions num_steps=10; "
                         "config.py pi05_libero action_horizon=10. READ FROM SOURCE; Alba to confirm.",
        "evaluation_path": "OpenPI server and client: create_trained_policy + WebsocketPolicyServer, "
                           "examples/libero/main.py observation and action handling.",
    },
}
CONDITIONS = [
    {"id": "NATIVE", "h": "native", "d": "native", "role": "Native baseline"},
    {"id": "R22", "h": 2, "d": 2, "role": "Low-refinement primary setting"},
    {"id": "R25", "h": 2, "d": 5, "role": "Fixed-prefix comparator"},
    {"id": "R52", "h": 5, "d": 2, "role": "Cheaper long-prefix setting"},
    {"id": "R55", "h": 5, "d": 5, "role": "Interaction/context setting"},
]
COMMON_RECORDING = [
    "git commit", "checkpoint revision", "code hash", "package versions", "CUDA version", "driver",
    "GPU UUID", "normalization-asset hash", "exact command line",
    "one write-ahead JSON record per attempt, updated in place", "action arrays", "success flag",
    "termination reason", "episode length", "simulator-state hash before rollout",
    "observation hash at each policy query", "timing", "GPU-memory samples", "generated action-chunk hash",
]
COMMON_RULES = [
    "Retain every planned attempt, including errors, failures, timeouts, and interrupted attempts.",
    "Never replace an attempt after observing its result.",
    "Preserve raw journals. Derived tables and figures must be reproducible from raw records.",
    "Keep model outputs in separate directories. Do not overwrite existing SmolVLA evidence.",
    "Use one model process per GPU. Do not run multiple model processes on the same GPU.",
    "Do not alter any past evidence, frozen predictions, submitted PDFs, or prior experiment directories.",
]
INVALID_EVIDENCE = "The earlier LeRobot PyTorch pi0.5 campaign (160 attempts) is invalid and excluded."


def resolve(condition, model):
    native = MODELS[model]["native"]
    h = native["h"] if condition["h"] == "native" else condition["h"]
    d = native["d"] if condition["d"] == "native" else condition["d"]
    return h, d


def load_states(cohort):
    """Return the state manifest, or an explicit PENDING binding if states are not generated yet."""
    path = RUNS / cohort / "states/state_manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text("utf-8"))
        return manifest, {row["key"]: row for row in manifest["states"]}
    keys = {"{}.task{:03d}.slot{}".format(suite, task, slot): {"state_sha256": None}
            for suite in SUITES for task in TASK_IDS for slot in STATE_SLOTS}
    return {"states_sha256": None, "binding": "PENDING_STATE_GENERATION"}, keys


def make_schedule(cohort, model, seeds, conditions, states):
    """Whole identity blocks in hashed order; condition order hashed within each block.

    Hash order is fixed before any outcome exists. It stops time-of-run drift from lining
    up with a condition or a task.
    """
    identities = [identity(suite, task, slot, seed) for suite in SUITES for task in TASK_IDS
                  for slot in STATE_SLOTS for seed in seeds]
    identities.sort(key=lambda row: sha([SCHEDULE_SEED, cohort, model, "block", row]))
    rows = []
    for block_index, ident in enumerate(identities):
        key = "{suite}.task{task_id:03d}.slot{state_slot}".format(**ident)
        state = states[key]
        ordered = sorted(conditions, key=lambda c: sha([SCHEDULE_SEED, cohort, model, ident, c["id"]]))
        for condition_index, condition in enumerate(ordered):
            block_id = "{}.{}.{}".format(cohort, model, identity_id(ident))
            rows.append({"schedule_index": len(rows), "block_index": block_index,
                         "condition_index": condition_index, "cohort": cohort, "model": model,
                         "block_id": block_id, "episode_identity": ident,
                         "state_key": key, "state_sha256": state["state_sha256"],
                         "condition_id": condition["id"],
                         "planned_run_id": "{}.{}".format(block_id, condition["id"]),
                         **{k: v for k, v in condition.items() if k not in ("id", "role")}})
    return rows


def schedule_document(cohort, model, rows, manifest):
    return {"schema_version": 1, "cohort": cohort, "model": model, "attempt_count": len(rows),
            "identity_count": len({r["block_id"] for r in rows}),
            "state_manifest_sha256": manifest["states_sha256"],
            "state_binding": manifest.get("binding", "BOUND"),
            "schedule_sha256": sha(rows), "schedule": rows}


def confirmation_protocol():
    return {
        "schema_version": 1, "protocol_id": CONFIRMATION, "status": "PREPARED_AWAITING_PREFLIGHT_AND_REVIEW",
        "launch_permitted": False,
        "question": "Is the d=2 over d=5 result at h=2 specific to SmolVLA, or does it also appear in a "
                    "second policy family, and how does h=5,d=2 compare with h=2,d=2?",
        "models": MODELS,
        "population": {"suites": list(SUITES), "task_ids": list(TASK_IDS), "state_slots": list(STATE_SLOTS),
                       "policy_seeds": list(CONFIRMATION_SEEDS), "identity_definition":
                       "suite / task_id / state_slot / policy_seed", "identities_per_model": 400,
                       "state_rule": "Serialized initial simulator state plus SHA-256 for every state slot; "
                                     "see states/state_manifest.json. Never relies on an RNG seed alone."},
        "conditions": CONDITIONS,
        "h_definition": "The number of consecutive actions executed from the latest generated action chunk "
                        "before requesting a new chunk. The policy-native chunk length is unchanged.",
        "conditions_resolved": {m: {c["id"]: dict(zip("hd", resolve(c, m))) for c in CONDITIONS} for m in MODELS},
        "expected_attempts": {"per_model": 2000, "total": 4000},
        "stop_rule_pi05": "If d=2 or d=5 is unsupported, produces non-finite actions, changes action "
                          "normalization or violates the policy interface: stop the pi0.5 experimental phase "
                          "and report unsupported_refinement_setting. Never skip, truncate, interpolate or "
                          "reuse actions to emulate a lower denoising count.",
        "exclusions": [INVALID_EVIDENCE,
                       "Must not reuse any identity from the SmolVLA discovery, SmolVLA confirmation, "
                       "stale-observation diagnostic, or invalid pi0.5 campaign."],
        "mandatory_preflight": [
            "Native parity on four suites, ten tasks per suite, ten episodes per task.",
            "Save per-task successes, action dimensions, action-unit checks, normalization checks, "
            "first-action hashes, and videos.",
            "Profile full chunk generation at d=2, d=5 and native d: 100 warm-up then 100 measured calls on a "
            "fixed input; host time, CUDA-event time, peak GPU memory, chunk size, full cost trace.",
            "Finite actions and valid shapes in all attempted conditions.",
            "Exact reset-state reuse across all conditions of the same identity."],
        "primary_analysis": {
            "selector": "Do not fit or refit a selector model.",
            "primary_contrast": "R22 minus R25", "secondary_contrast": "R52 minus R22",
            "report": ["success count and equal-suite macro success", "paired state-level differences",
                       "20,000 paired hierarchical bootstrap draws retaining both seeds within each state",
                       "model-specific 95% intervals", "all five conditions regardless of outcome",
                       "arithmetic per generated chunk, per control step, per attempted episode",
                       "host, CUDA and end-to-end rollout timing as separate quantities"],
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_structure_PROPOSED_R3": "Suites and tasks fixed; resample state slots with replacement "
                                               "within each task; both seeds stay inside each state.",
            "pooling": "Never pool the models. Report model effects separately, then state descriptively "
                       "whether signs agree or disagree."},
        "quality_preservation": {
            "contrast": "R52 minus R22", "margin_points": QUALITY_MARGIN_POINTS,
            "rule": "One-sided 95% lower confidence bound must be above -5.0 points.",
            "if_passed": "The five-action setting preserves quality within the stated engineering margin "
                         "for that model and evaluation population.",
            "if_not_passed": "Report as a quality loss or inconclusive.", "forbidden_word": "equivalent"},
        "recording": COMMON_RECORDING, "rules": COMMON_RULES,
    }


def age_protocol():
    return {
        "schema_version": 1, "protocol_id": AGE, "status": "PREPARED_AWAITING_PREFLIGHT_AND_REVIEW",
        "launch_permitted": False, "runs_after": CONFIRMATION, "pooling": "Do not pool with confirmation data.",
        "models": MODELS,
        "population": {"suites": list(SUITES), "task_ids": list(TASK_IDS), "state_slots": list(STATE_SLOTS),
                       "policy_seeds_PROPOSED_R2": list(OBSERVATION_AGE_SEEDS), "identities_per_model": 400},
        "held_constant": {"h": 1, "d": 5, "noise": "same action-noise key by simulation tick",
                          "same": ["initial state", "task instruction", "generation cadence"]},
        "conditions": [{"id": "FRESH", "refresh_every_queries": 1,
                        "definition": "Images and proprioception refreshed every policy query."},
                       {"id": "STALE", "refresh_every_queries": 5,
                        "definition": "Images and proprioception refreshed every five policy queries."}],
        "sole_intervention": "Supplied observation age. Policy-query frequency, denoising, executed action "
                             "position, task instruction, noise key, simulator timestep and control frequency "
                             "do not change.",
        "per_query_record": ["current simulator observation hash", "supplied policy-input hash",
                             "supplied observation age", "refresh flag", "generated chunk hash", "executed action",
                             "first action divergence relative to the paired condition",
                             "outcome and termination step"],
        "expected_attempts": {"per_model": 800, "total": 1600},
        "recording": COMMON_RECORDING, "rules": COMMON_RULES,
    }


def async_protocol():
    return {
        "schema_version": 1, "protocol_id": ASYNC,
        "status": "BLOCKED_NO_OFFICIAL_RUNTIME_FOR_THIS_DESIGN", "launch_permitted": False,
        "runs_after": "Synchronous confirmation and observation-age studies complete and audited.",
        "blocker_R1": "The specification requires the official asynchronous or real-time chunking runtime for both "
                      "policies. Pinned LeRobot async_inference and rollout drive real robots only "
                      "(constants.py SUPPORTED_ROBOTS) and have no simulator path; its pi05 is the PyTorch path. "
                      "OpenPI's client runtime is synchronous. Alba must rule; see audit R1.",
        "models": MODELS, "conditions": ["NATIVE", "R22", "R52"],
        "population": "Same paired identities as the synchronous confirmation; a separate asynchronous cohort.",
        "expected_attempts": {"per_model": 1200, "total": 2400},
        "per_control_step_record": ["image capture timestamp", "observation delivery timestamp",
                                    "policy-query start and end timestamp", "action enqueue timestamp",
                                    "action execution timestamp", "action queue depth", "observation age at execution",
                                    "deadline miss indicator", "queue-starvation indicator", "success and episode length"],
        "report": ["median, p95 and p99 end-to-end action latency", "effective control rate", "missed deadlines",
                   "queue starvation", "success", "GPU memory", "energy if available"],
        "claim_limit": "Do not claim that synchronous simulator timing predicts asynchronous or physical-robot latency.",
        "recording": COMMON_RECORDING, "rules": COMMON_RULES,
    }


def outputs():
    files = {}
    def add(path, value):
        files[path] = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    add(PROTOCOLS / (CONFIRMATION + ".json"), confirmation_protocol())
    add(PROTOCOLS / (AGE + ".json"), age_protocol())
    add(PROTOCOLS / (ASYNC + ".json"), async_protocol())
    confirmation_manifest, confirmation_states = load_states(CONFIRMATION)
    age_manifest, age_states = load_states(AGE)
    for model in MODELS:
        resolved = [dict(c, h=resolve(c, model)[0], d=resolve(c, model)[1]) for c in CONDITIONS]
        rows = make_schedule(CONFIRMATION, model, CONFIRMATION_SEEDS, resolved, confirmation_states)
        add(RUNS / CONFIRMATION / model / "schedule.json",
            schedule_document(CONFIRMATION, model, rows, confirmation_manifest))
        age = [{"id": "FRESH", "h": 1, "d": 5, "refresh_every_queries": 1, "role": "fresh"},
               {"id": "STALE", "h": 1, "d": 5, "refresh_every_queries": 5, "role": "stale"}]
        rows = make_schedule(AGE, model, OBSERVATION_AGE_SEEDS, age, age_states)
        add(RUNS / AGE / model / "schedule.json", schedule_document(AGE, model, rows, age_manifest))
        chosen = [c for c in resolved if c["id"] in ("NATIVE", "R22", "R52")]
        rows = make_schedule(ASYNC, model, CONFIRMATION_SEEDS, chosen, confirmation_states)
        add(RUNS / ASYNC / model / "schedule.json", schedule_document(ASYNC, model, rows, confirmation_manifest))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = outputs()
    stale = []
    for path, text in files.items():
        if args.check:
            if not path.is_file() or path.read_text("utf-8") != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if stale:
        raise SystemExit("stale or missing: {}".format(stale))
    print("{} {} files".format("checked" if args.check else "wrote", len(files)))


if __name__ == "__main__":
    main()
