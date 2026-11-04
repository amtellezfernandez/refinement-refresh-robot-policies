#!/usr/bin/env python3
"""Replay archived or synthetic LIBERO action streams without a policy.

Usage:
  replay.py run BUNDLES.jsonl OUT_DIR [--mode reset_path|inject] [--no-render]
                                       [--release RELEASE.json --protocol PROTOCOL.md]

Bundles whose provenance is `archived_paper_evidence` are REFUSED (exit 5) unless a
release file binds the SHA-256 of the frozen protocol and names an approver.  This is
the governance gate: Alba reviews the protocol before any archived replay runs.

Outputs (never inside the input location):
  OUT_DIR/results.jsonl   one record per bundle
  OUT_DIR/summary.json    verdict counts, run identity, environment identity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import libero_replay as lr  # noqa: E402

SCHEMA = "replay_bundle.v1"
VERDICTS = ["INPUT_INVALID", "DUPLICATE", "UNREPLAYABLE", "RESET_MISMATCH", "OUTCOME_MISMATCH",
            "OUTCOME_MATCH_TICK_DIVERGED", "REPRODUCED_RESET_DRIFT", "REPRODUCED"]
# Frozen before any archived comparison (see PROTOCOL.md section 7). Max absolute difference,
# over all flattened state components, between the replay's post-settle state and the
# recorded initial_state. Above it the replay did not start from the recorded state.
RESET_TOL = 1e-6


def keyed_seed(identity, tick=None, domain="policy_noise"):
    key = [domain, identity] if tick is None else [domain, identity, tick]
    return int.from_bytes(hashlib.sha256(canonical_str(key)).digest()[:4], "little")


def canonical_str(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


# ---------------------------------------------------------------- input integrity
def validate(b):
    d = []
    for k in ("schema", "provenance", "episode_key", "suite", "task_id", "reset_source",
              "environment_seed", "recorded_initial_state", "actions", "recorded_success_flags",
              "label", "horizon"):
        if k not in b:
            d.append(f"missing field {k}")
    if d:
        return d
    if b["schema"] != SCHEMA:
        d.append("schema")
    if b["suite"] not in lr.WRAPPER_HORIZON:
        d.append("unknown suite")
    elif b["horizon"] != lr.WRAPPER_HORIZON[b["suite"]]:
        d.append(f"horizon {b['horizon']} != wrapper {lr.WRAPPER_HORIZON[b['suite']]}")
    acts, flags, lab = b["actions"], b["recorded_success_flags"], b["label"]
    for i, a in enumerate(acts):
        if a.get("dtype") != "<f4" or a.get("shape") != [7]:
            d.append(f"action {i} dtype/shape")
            break
        if lr.array_sha(lr.array_from_record(a)) != a.get("sha256"):
            d.append(f"action {i} hash mismatch")
            break
    if a_ticks := b.get("action_ticks"):
        if a_ticks != list(range(len(acts))):
            d.append("action ticks not contiguous from 0")
    if not (len(acts) == len(flags) == lab.get("steps")):
        d.append(f"length mismatch actions={len(acts)} flags={len(flags)} label.steps={lab.get('steps')}")
    if len(acts) == 0:
        d.append("empty action stream")
    if bool(lab.get("success")) != any(flags):
        d.append("label.success disagrees with recorded flags")
    if any(flags) and flags.index(True) != len(flags) - 1:
        d.append("recorded success before final step")
    if not any(flags) and len(flags) != b["horizon"]:
        d.append("failure shorter than horizon")
    exp_reason = "success" if any(flags) else "task_horizon"
    if lab.get("termination_reason") != exp_reason:
        d.append(f"termination_reason {lab.get('termination_reason')} != {exp_reason}")
    if "recorded_action_stream_sha256" in b:
        got = hashlib.sha256(canonical_str([a.get("sha256") for a in acts])).hexdigest()
        if got != b["recorded_action_stream_sha256"]:
            d.append("action stream hash differs from the episode-level record")
    ris = b["recorded_initial_state"]
    if lr.array_sha(lr.array_from_record(ris)) != ris.get("sha256"):
        d.append("recorded initial state hash mismatch")
    rs = b["reset_source"]
    if rs.get("kind") not in ("official", "vector"):
        d.append("reset_source kind")
    if rs.get("kind") == "official":
        want = keyed_seed([b["suite"], b["task_id"], rs.get("initial_state_id")], domain="environment")
        if b["environment_seed"] != want:
            d.append("environment_seed differs from keyed rule for this task-state")
    return d


# ---------------------------------------------------------------- governance gate
def gate(bundles, release, protocol):
    archived = [b for b in bundles if b.get("provenance") != "synthetic_fixture"]
    if not archived:
        return None
    if not release or not protocol:
        return (f"REFUSED: {len(archived)} bundle(s) are not synthetic fixtures; archived replay "
                "needs --release and --protocol (Alba's approval of the frozen protocol)")
    rel = json.loads(Path(release).read_text())
    psha = hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
    if rel.get("protocol_sha256") != psha or not rel.get("approved_by") or rel.get("decision") != "APPROVED":
        return "REFUSED: release does not approve this exact protocol"
    return None


# ---------------------------------------------------------------- replay
def pre_settle_state(b, cache):
    rs = b["reset_source"]
    if rs["kind"] == "vector":
        return lr.array_from_record(rs["state"]).astype(np.float64), {"kind": "vector"}
    key = (b["suite"], b["task_id"])
    if key not in cache:
        cache[key] = lr.official_init_states(*key)
    states, path = cache[key]
    sid = rs["initial_state_id"]
    if not (isinstance(sid, int) and 0 <= sid < len(states)):
        raise ValueError(f"initial_state_id {sid} out of range (no modulo fallback)")
    return np.asarray(states[sid], dtype=np.float64), {"kind": "official", "file_sha256": lr.file_sha(path),
                                                       "initial_state_id": sid}


def replay_one(rep, b, mode, cache):
    t0 = time.perf_counter()
    pre, src = pre_settle_state(b, cache)
    recorded = lr.array_from_record(b["recorded_initial_state"]).astype(np.float64)
    ours = rep.reset(pre, b["environment_seed"])
    reset_exact = ours.shape == recorded.shape and np.array_equal(ours, recorded)
    reset_maxabs = float(np.max(np.abs(ours - recorded))) if ours.shape == recorded.shape else None
    if mode == "inject" and ours.shape == recorded.shape:
        rep.inject(recorded)
    flags, first = [], None
    for i, a in enumerate(b["actions"]):
        s, _, _ = rep.step(lr.array_from_record(a))
        flags.append(s)
        if s and first is None:
            first = i
    rec_flags = [bool(x) for x in b["recorded_success_flags"]]
    n = min(len(flags), len(rec_flags))
    div = next((i for i in range(n) if flags[i] != rec_flags[i]), None)
    if div is None and len(flags) != len(rec_flags):
        div = n
    rec_success = bool(b["label"]["success"])
    rep_success = first is not None
    return {
        "reset_source": src, "reset_state_exact": reset_exact, "reset_state_max_abs_diff": reset_maxabs,
        "replay_state_after_reset_sha256": lr.array_sha(ours),
        "replay_success": rep_success, "replay_first_success_tick": first,
        "recorded_success": rec_success,
        "recorded_first_success_tick": rec_flags.index(True) if any(rec_flags) else None,
        "first_divergent_tick": div, "actions_applied": len(flags),
        "terminal_predicates": rep.predicates(), "terminal_state_sha256": rep.state_sha(),
        "replay_wall_s": time.perf_counter() - t0,
    }


def classify(defects, dup, r):
    if defects:
        return "INPUT_INVALID"
    if dup:
        return "DUPLICATE"
    if r is None:
        return "UNREPLAYABLE"
    if r["reset_state_max_abs_diff"] is None or r["reset_state_max_abs_diff"] > RESET_TOL:
        return "RESET_MISMATCH"
    if r["replay_success"] != r["recorded_success"]:
        return "OUTCOME_MISMATCH"
    if r["first_divergent_tick"] is not None:
        return "OUTCOME_MATCH_TICK_DIVERGED"
    if not r["reset_state_exact"]:
        return "REPRODUCED_RESET_DRIFT"
    return "REPRODUCED"


def environment_identity():
    import importlib.metadata as m
    pk = {p: m.version(p) for p in ("mujoco", "robosuite", "hf_libero", "bddl", "numpy", "torch")}
    from libero.libero import get_libero_path
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "packages": pk, "assets_path": str(Path(get_libero_path("assets")).resolve())}


def run(bundles, out_dir, mode="reset_path", render=True, release=None, protocol=None):
    refusal = gate(bundles, release, protocol)
    if refusal:
        print(refusal, file=sys.stderr)
        return 5
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run
    seen_keys, seen_streams = {}, {}
    replayers, cache, results = {}, {}, []
    with (out / "results.jsonl").open("x") as f:
        for b in bundles:
            defects = validate(b)
            key = b.get("episode_key")
            stream = hashlib.sha256(canonical([a.get("sha256") for a in b.get("actions", [])])).hexdigest()
            dup = []
            if key in seen_keys:
                dup.append(f"episode_key repeats bundle {seen_keys[key]}")
            if stream in seen_streams and b.get("actions"):
                dup.append(f"action stream identical to {seen_streams[stream]}")
            seen_keys.setdefault(key, len(results))
            seen_streams.setdefault(stream, key)
            r, err = None, None
            if not any(x.startswith("missing field") for x in defects):
                try:
                    k = (b["suite"], b["task_id"])
                    if k not in replayers:
                        for old in replayers.values():
                            old.close()
                        replayers = {k: lr.Replayer(*k, render=render)}
                    r = replay_one(replayers[k], b, mode, cache)
                except Exception:
                    err = traceback.format_exc(limit=3)
            rec = {"episode_key": key, "provenance": b.get("provenance"), "mode": mode, "render": render,
                   "input_defects": defects, "duplicate": dup, "error": err, "replay": r,
                   "verdict": classify(defects, dup, r)}
            results.append(rec)
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
    for rp in replayers.values():
        rp.close()
    summary = {"verdicts": dict(Counter(r["verdict"] for r in results)), "n": len(results), "mode": mode,
               "render": render, "environment": environment_identity(),
               "harness_sha256": {p.name: lr.file_sha(p) for p in sorted(HERE.glob("*.py"))}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary["verdicts"]))
    return 0


def read_bundles(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("bundles")
    r.add_argument("out")
    r.add_argument("--mode", choices=["reset_path", "inject"], default="reset_path")
    r.add_argument("--no-render", action="store_true")
    r.add_argument("--release")
    r.add_argument("--protocol")
    a = ap.parse_args()
    sys.exit(run(read_bundles(a.bundles), a.out, a.mode, not a.no_render, a.release, a.protocol))


if __name__ == "__main__":
    main()
