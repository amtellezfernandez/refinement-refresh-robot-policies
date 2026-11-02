#!/usr/bin/env python3
"""Evaluate a versioned preflight run directory. Reads records only.

    preflight_verdict.py <run dir>

Missing evidence fails. No mechanical result is a pass of the preflight: native parity needs Alba's
criterion (R9) and the videos need looking at, so the best this tool reports is that they await review.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OPENPI_NORM_SHA256  # noqa: E402

PROFILE_CALLS, DENOISING = 100, (2, 5, 10)
PARITY_ATTEMPTS, STATE_COUNT = 400, 200
MODELS = {"smolvla": "smolvla", "pi05": "pi05_parity"}


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def attempts(root, directory):
    return [read(p) for p in sorted((root / directory / "parity_attempts").glob("*.attempt[0-9][0-9].json"))]


def check_dependency(root):
    for name in ("smolvla", "pi05_server", "pi05_parity"):
        packages = read(root / name / "provenance.json")["packages"]
        if not isinstance(packages, list) or not packages:
            return False, "{}: packages not recorded".format(name)
    return True, "package lists recorded for all three environments"


def check_checkpoint(root):
    for name in ("smolvla", "pi05_server"):
        tree = read(root / name / "provenance.json").get("checkpoint_tree")
        if not isinstance(tree, dict) or not tree.get("files"):
            return False, "{}: checkpoint tree not recorded".format(name)
    return True, "checkpoint trees hashed"


def check_normalization(root):
    record = read(root / "pi05_server/provenance.json")
    good = record.get("normalization_sha256") == OPENPI_NORM_SHA256 and (record.get("normalization_loaded") or {}).get("use_quantiles") is True
    return good, "file hash and loaded quantile statistics" if good else "hash or loaded statistics differ"


def check_received(root):
    smol = read(root / "smolvla/profile.json")
    for d in DENOISING:
        counts = smol[str(d)]["counts"]
        if counts["denoise_steps"] != counts["generations"] * d or counts["generations"] != PROFILE_CALLS:
            return False, "smolvla d={}: counts {}".format(d, counts)
    profile = read(root / "pi05_profile/profile.json")
    points = profile["points"]
    if profile["denoise_loop_calls"] != {str(d): d for d in DENOISING}:
        return False, "pi05: loop calls {} with the loaded weights".format(profile["denoise_loop_calls"])
    for d in DENOISING:
        if points[str(d)]["received_num_steps"] != [d] * PROFILE_CALLS:
            return False, "pi05 d={}: sampler did not receive it on every call".format(d)
    if len({points[str(d)]["fixed_noise_chunk_sha256"] for d in DENOISING}) != len(DENOISING):
        return False, "pi05: same noise gave the same chunk at different d"
    return True, "counted at the model boundary (SmolVLA); loop calls counted with loaded weights, received at the sampler, and changes the chunk (pi0.5)"


def check_timing(root):
    smol = read(root / "smolvla/profile.json")
    points = read(root / "pi05_profile/profile.json")["points"]
    for d in DENOISING:
        if len(smol[str(d)]["host_s"]) != PROFILE_CALLS or len(smol[str(d)]["cuda_ms"]) != PROFILE_CALLS or len(points[str(d)]["host_s"]) != PROFILE_CALLS:
            return False, "d={}: not {} measured calls".format(d, PROFILE_CALLS)
    for directory in MODELS.values():
        if any("rollout_wall_s" not in (a.get("timing") or {}) for a in attempts(root, directory) if a["status"] == "completed"):
            return False, "{}: rollout timing missing".format(directory)
    return True, "100 measured calls per d; rollout time per episode. pi0.5 CUDA-event time is unavailable (R5)"


def check_conditions(root):
    results = {name: read(root / name / "conditions.json")["status"] for name in ("smolvla_conditions", "pi05_conditions")}
    return all(v == "PASS" for v in results.values()), str(results)


def check_parity(root):
    for model, directory in MODELS.items():
        rows = attempts(root, directory)
        done = sum(a["status"] == "completed" for a in rows)
        if len(rows) != PARITY_ATTEMPTS or done != PARITY_ATTEMPTS:
            return False, "{}: {} completed of {} records, {} required".format(model, done, len(rows), PARITY_ATTEMPTS)
    return True, "400 of 400 completed per model"


def check_actions(root):
    for model, directory in MODELS.items():
        for attempt in attempts(root, directory):
            if attempt["status"] != "completed":
                continue
            shape = attempt.get("action_shape") or []
            ranges = (attempt.get("action_min"), attempt.get("action_max"))
            if len(shape) != 2 or shape[1] != 7 or any(not r or len(r) != 7 for r in ranges) or not attempt.get("first_action_sha256"):
                return False, "{}: {} shape, range or first-action hash".format(model, attempt["planned_run_id"])
    return True, "seven dimensions and per-dimension ranges recorded; unit acceptance awaits R9"


def check_videos(root):
    for directory in MODELS.values():
        manifest = read(root / directory / "videos/video_manifest.json")
        if manifest["failed"] or manifest["videos"] != PARITY_ATTEMPTS:
            return False, "{}: {} videos, {} failed".format(directory, manifest["videos"], manifest["failed"])
    return True, "readable, not blank, not frozen"


def check_states(root):
    manifest = read(root / "states/state_manifest.json")
    hashes = [s.get("state_sha256") for s in manifest["states"]]
    return len(hashes) == STATE_COUNT and all(hashes), "{} state hashes".format(len(hashes))


def check_binding(root):
    start = read(root / "start.json")
    return all(start.get(k) for k in ("run_id", "protocol_sha256", "release_sha256")), "run, protocol and release recorded at start"


CHECKS = {"protocol_hash_binding": check_binding, "dependency_provenance": check_dependency, "checkpoint_provenance": check_checkpoint,
          "official_normalization": check_normalization, "implementation_received_d": check_received, "timing": check_timing,
          "conditions_smoke_and_reset": check_conditions, "native_parity_complete": check_parity,
          "action_units_and_shapes": check_actions, "video_files": check_videos, "generated_state_hashes": check_states}


def parity_table(root):
    table = {}
    for model, directory in MODELS.items():
        counts = collections.defaultdict(lambda: [0, 0])
        for attempt in attempts(root, directory):
            identity = attempt["schedule_row"]["episode_identity"]
            key = "{}.task{:03d}".format(identity["suite"], identity["task_id"])
            counts[key][1] += 1
            counts[key][0] += bool(attempt.get("outcome", {}).get("success"))
        table[model] = {key: {"successes": v[0], "attempts": v[1]} for key, v in sorted(counts.items())}
    return table


def evaluate(root):
    root = Path(root)
    checks = {}
    for name, check in CHECKS.items():
        try:
            good, detail = check(root)
        except (OSError, KeyError, ValueError, TypeError) as error:
            good, detail = False, "evidence missing or malformed: {!r}".format(error)
        checks[name] = {"status": "PASS" if good else "FAIL", "detail": detail}
    failed = any(c["status"] == "FAIL" for c in checks.values())
    try:
        table = parity_table(root)
    except (OSError, KeyError, ValueError):
        table = {}
    return {"verdict": "PREFLIGHT_FAILED" if failed else "MECHANICAL_CHECKS_PASSED_AWAITING_PARITY_AND_VIDEO_REVIEW",
            "checks": checks, "parity_success_by_task": table,
            "awaiting": ["R9 parity criterion applied by Alba", "human video inspection", "action-unit acceptance"]}


if __name__ == "__main__":
    report = evaluate(sys.argv[1])
    print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(0 if report["verdict"] != "PREFLIGHT_FAILED" else 1)
