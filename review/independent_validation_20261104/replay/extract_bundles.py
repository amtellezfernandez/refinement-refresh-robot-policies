#!/usr/bin/env python3
"""Build replay bundles from the archived confirmation evidence, and freeze the sample.

This reads archived records and writes bundles; it does not replay anything.  Every
bundle carries provenance `archived_paper_evidence`, which replay.py refuses without an
approved release bound to the frozen protocol.

  extract_bundles.py ARCHIVE OUT_BUNDLES.jsonl SAMPLE_MANIFEST.json

Population: the 1,120 canonical confirmation attempts, canonical attempt per planned run
chosen by phase precedence confirmation > primary_deadline > continuation_partial >
continuation_rest (the archived amendment's order; independently re-derived in
audit/independent_audit.py).

Stratified sample (Stage 1), frozen here: frame = the 320 primary attempts (h2_d2, h2_d5);
strata = task (8) x condition (2) x archived outcome (success / horizon failure);
within each stratum, order by sha256("rrrp-replay-stage1-20261104|" + planned_run_id) and
take the first 2 (all, if the stratum holds fewer).  Outcome labels are used only to
stratify; no replay result exists when the sample is drawn.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from collections import defaultdict

PHASES = ["confirmation", "primary_deadline", "confirmation_continuation_partial", "confirmation_continuation_rest"]
SALT = "rrrp-replay-stage1-20261104|"
PER_STRATUM = 2


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


def main(archive, out_bundles, out_sample):
    members = {}
    with tarfile.open(archive, "r:gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.split("/")[0] in PHASES:
                members[m.name] = t.extractfile(m).read()
    chosen = {}
    for phase in PHASES:
        for line in io.BytesIO(members[f"{phase}/episodes.jsonl"]).read().decode().splitlines():
            e = json.loads(line)
            chosen.setdefault(e["planned_run_id"], (phase, e))
    need = {e["run_id"]: pid for pid, (_, e) in chosen.items()}
    traces = defaultdict(list)
    for phase in PHASES:
        for line in io.BytesIO(members[f"{phase}/steps.jsonl"]).read().decode().splitlines():
            s = json.loads(line)
            if s["run_id"] in need and s["kind"] == "step":
                traces[s["run_id"]].append(s)
    n = 0
    with open(out_bundles, "x") as f:
        for pid in sorted(chosen):
            phase, e = chosen[pid]
            st = traces[e["run_id"]]
            i = e["episode_identity"]
            b = {"schema": "replay_bundle.v1", "provenance": "archived_paper_evidence", "episode_key": pid,
                 "archived_run_id": e["run_id"], "archived_phase": phase, "condition": f"h{e['h']}_d{e['denoising_steps']}",
                 "suite": i["suite"], "task_id": i["task_id"],
                 "reset_source": {"kind": "official", "initial_state_id": i["initial_state_id"]},
                 "environment_seed": e["environment_seed"], "recorded_initial_state": e["initial_state"],
                 "actions": [s["producer_action"] for s in st], "action_ticks": [s["tick"] for s in st],
                 "recorded_success_flags": [bool(s["driver_is_success"]) for s in st],
                 "label": {"success": bool(e["success"]), "steps": e["steps"], "termination_reason": e["termination_reason"]},
                 "horizon": e["task_horizon"], "recorded_action_stream_sha256": e["action_stream_sha256"]}
            f.write(json.dumps(b) + "\n")
            n += 1
    strata = defaultdict(list)
    for pid, (_, e) in chosen.items():
        c = (e["h"], e["denoising_steps"])
        if c not in ((2, 2), (2, 5)):
            continue
        i = e["episode_identity"]
        strata[(i["suite"], i["task_id"], f"h{c[0]}_d{c[1]}", "success" if e["success"] else "failure")].append(pid)
    sample, table = [], []
    for k in sorted(strata):
        ordered = sorted(strata[k], key=lambda p: hashlib.sha256((SALT + p).encode()).hexdigest())
        take = ordered[:PER_STRATUM]
        sample += take
        table.append({"stratum": list(k), "frame": len(strata[k]), "sampled": take})
    manifest = {"population": n, "primary_frame": sum(len(v) for v in strata.values()),
                "strata_nonempty": len(strata), "stage1_sample_size": len(sample), "salt": SALT,
                "per_stratum": PER_STRATUM, "stage1_sample": sample,
                "stage1_sample_sha256": hashlib.sha256(canonical(sorted(sample))).hexdigest(), "strata": table}
    with open(out_sample, "x") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("population", "primary_frame", "strata_nonempty",
                                               "stage1_sample_size", "stage1_sample_sha256")}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
