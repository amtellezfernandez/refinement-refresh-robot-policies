#!/usr/bin/env python3
"""Negative controls for independent_audit.py.

Each mutation rewrites one member of an in-memory COPY of the archive, repacks it in a
scratch directory, runs the audit, and requires that exactly the named checks fail (at
least those; any other failure is reported).  The unmodified repack must pass every check.

Usage: test_audit_mutations.py ARCHIVE PLAN REPORT SCRATCH_DIR OUT.json
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    members = {}
    with tarfile.open(path, "r:gz") as t:
        for m in t.getmembers():
            if m.isfile():
                members[m.name] = t.extractfile(m).read()
    return members


def pack(members, path):
    with tarfile.open(path, "w:gz", compresslevel=1) as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def lines(b):
    return b.decode().splitlines()


def edit_jsonl(members, name, fn):
    rows = [json.loads(l) for l in lines(members[name]) if l.strip()]
    rows = fn(rows)
    members[name] = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def first(rows, pred):
    return next(i for i, r in enumerate(rows) if pred(r))


def m_flip_episode_success(ms):
    def f(rows):
        i = first(rows, lambda r: r["h"] == 2 and r["denoising_steps"] == 2 and r["success"])
        rows[i]["success"] = False
        return rows
    edit_jsonl(ms, "confirmation/episodes.jsonl", f)


def m_flip_step_success(ms):
    eps = [json.loads(l) for l in lines(ms["confirmation/episodes.jsonl"])]
    target = next(e["run_id"] for e in eps if e["h"] == 2 and e["denoising_steps"] == 2 and e["success"])

    def f(rows):
        i = max(j for j, r in enumerate(rows) if r.get("run_id") == target and r["kind"] == "step")
        rows[i]["driver_is_success"] = rows[i]["vector_is_success"] = False
        return rows
    edit_jsonl(ms, "confirmation/steps.jsonl", f)


def m_alter_action(ms):
    def f(rows):
        i = first(rows, lambda r: r["kind"] == "step" and r["tick"] == 5)
        rows[i]["producer_action"]["values"][0] += 0.25
        return rows
    edit_jsonl(ms, "confirmation/steps.jsonl", f)


def m_delete_step(ms):
    def f(rows):
        i = first(rows, lambda r: r["kind"] == "step" and r["tick"] == 7)
        return rows[:i] + rows[i + 1:]
    edit_jsonl(ms, "confirmation/steps.jsonl", f)


def m_duplicate_episode(ms):
    def f(rows):
        return rows + [rows[3]]
    edit_jsonl(ms, "confirmation/episodes.jsonl", f)


def m_alter_initial_state(ms):
    def f(rows):
        rows[0]["initial_state"]["values"][15] += 1e-6
        return rows
    edit_jsonl(ms, "confirmation/episodes.jsonl", f)


def m_alter_env_seed(ms):
    def f(rows):
        rows[1]["environment_seed"] += 1
        return rows
    edit_jsonl(ms, "confirmation/episodes.jsonl", f)


def m_flip_canonical(ms):
    def f(rows):
        rows[10]["success"] = not rows[10]["success"]
        return rows
    edit_jsonl(ms, "confirmation_canonical_episodes.jsonl", f)


def m_drop_repeated_deadline_episode(ms):
    # rows[0] is task007.state029.seed3001.h2_d5, which continuation_partial repeats:
    # the plan stays covered, but the canonical attempt changes.
    def f(rows):
        return rows[1:]
    edit_jsonl(ms, "primary_deadline/episodes.jsonl", f)


def m_drop_unrepeated_deadline_episode(ms):
    def f(rows):
        return rows[:-1]
    edit_jsonl(ms, "primary_deadline/episodes.jsonl", f)


MUTATIONS = {
    "unmodified_repack": (lambda ms: None, set()),
    "flip_episode_success_only": (m_flip_episode_success, {"per_episode_trace_consistency"}),
    "flip_final_step_success": (m_flip_step_success, {"per_episode_trace_consistency", "primary_counts_139_122"}),
    "alter_one_action_value": (m_alter_action, {"per_episode_trace_consistency"}),
    "delete_one_step_event": (m_delete_step, {"per_episode_trace_consistency", "raw_step_count"}),
    "duplicate_episode_record": (m_duplicate_episode, {"confirmation:episode_run_ids_unique"}),
    "alter_initial_state_value": (m_alter_initial_state, {"per_episode_trace_consistency"}),
    "alter_environment_seed": (m_alter_env_seed, {"per_episode_trace_consistency"}),
    "flip_canonical_file_success": (m_flip_canonical, {"canonical_file_success_equals_trace"}),
    "drop_repeated_deadline_episode": (m_drop_repeated_deadline_episode,
                                       {"canonical_matches_archived_canonical_file", "raw_episode_count"}),
    "drop_unrepeated_deadline_episode": (m_drop_unrepeated_deadline_episode,
                                         {"canonical_covers_plan_exactly", "raw_episode_count"}),
}


def main(archive, plan, report, scratch, out):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    base = load(archive)
    results = []
    for name, (fn, must_fail) in MUTATIONS.items():
        ms = dict(base)
        fn(ms)
        tar = scratch / f"{name}.tar.gz"
        pack(ms, tar)
        res = scratch / f"{name}.json"
        subprocess.run([sys.executable, "-I", str(HERE / "independent_audit.py"), str(tar), plan, report, str(res)],
                       capture_output=True, text=True)
        got = json.loads(res.read_text())
        failed = {c["check"] for c in got["checks"] if not c["pass"]}
        ok = must_fail <= failed and (bool(must_fail) or not failed)
        results.append({"mutation": name, "must_fail": sorted(must_fail), "failed": sorted(failed), "pass": ok})
        print(("PASS " if ok else "FAIL ") + name, sorted(failed), flush=True)
        tar.unlink()
    Path(out).write_text(json.dumps({"results": results, "passed": sum(r["pass"] for r in results),
                                     "total": len(results)}, indent=1) + "\n")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
