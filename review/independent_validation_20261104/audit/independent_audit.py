#!/usr/bin/env python3
"""Independent audit of the archived SmolVLA confirmation evidence.

Written 2026-11-04 by Warren Smith's side, without importing or calling any of
the repository's analysis code (analyze_*.py, primary_reporting.py,
audit_outcome_instrument.py).  Standard library only.  Reads the archive
read-only; never writes inside it.

What it re-derives from raw step traces rather than from summary fields:
  * per-episode success, length and termination, from per-step flags
  * every action hash, initial-state hash and action-stream hash, by
    re-encoding the archived numeric arrays (the archive's own hash rule:
    sha256(json([dtype, shape]) + raw little-endian bytes))
  * environment seeds, from the documented keyed-seed rule
  * canonical attempt selection, from a precedence rule implemented here
  * all success counts, paired contrasts and a state-resampling bootstrap
    (independent RNG, so the interval is expected to match only approximately)

Usage:
  python3 -I independent_audit.py ARCHIVE.tar.gz PLAN.json PRIMARY_REPORT.json OUT.json
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import struct
import sys
import tarfile
from collections import Counter, defaultdict

PHASES = ["confirmation", "primary_deadline",
          "confirmation_continuation_partial", "confirmation_continuation_rest"]
HORIZON = {"libero_object": 280, "libero_goal": 300, "libero_spatial": 280}
PRIMARY_SEL, PRIMARY_CMP = (2, 2), (2, 5)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=str).encode("utf-8")


def sha_obj(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def array_sha(record):
    """Re-encode an archived array record; return sha256 per the archive's rule."""
    dtype, shape, values = record["dtype"], record["shape"], record["values"]
    flat = []

    def walk(v):
        if isinstance(v, list):
            for x in v:
                walk(x)
        else:
            flat.append(v)
    walk(values)
    n = 1
    for s in shape:
        n *= s
    if len(flat) != n:
        return None
    code = {"<f4": "f", "<f8": "d", "<i8": "q", "<i4": "i"}.get(dtype)
    if code is None:
        return None
    header = json.dumps([dtype, list(shape)], separators=(",", ":")).encode()
    body = struct.pack("<%d%s" % (n, code), *flat)
    return hashlib.sha256(header + body).hexdigest()


def keyed_seed(identity, tick=None, domain="policy_noise"):
    key = [domain, identity] if tick is None else [domain, identity, tick]
    return int.from_bytes(hashlib.sha256(canonical(key)).digest()[:4], "little")


def read_members(path):
    out = {}
    with tarfile.open(path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                out[m.name] = tar.extractfile(m).read()
    return out


def jsonl(blob):
    return [json.loads(line) for line in io.BytesIO(blob).read().decode().splitlines() if line.strip()]


def main(archive, plan_path, report_path, out_path):
    findings, checks = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            findings.append(f"{name}: {detail}")

    archive_sha = hashlib.sha256(open(archive, "rb").read()).hexdigest()
    members = read_members(archive)
    plan = json.load(open(plan_path))
    report = json.load(open(report_path))

    # ---------- plan population ----------
    planned = {r["planned_run_id"]: r for r in plan["schedule"]}
    design = plan["design"]
    conds = sorted({(c["h"], c["denoising_steps"]) for c in design["conditions"]})
    expected_n = len(design["tasks"]) * len(design["initial_state_ids"]) * len(design["policy_seeds"]) * len(conds)
    check("plan_size", len(planned) == len(plan["schedule"]) == expected_n == 1120,
          f"{len(plan['schedule'])} rows, {len(planned)} unique, expected {expected_n}")
    check("plan_schedule_sha", sha_obj(plan["schedule"]) == plan["scheduling"]["schedule_sha256"],
          "recomputed sha of schedule vs recorded")

    # ---------- per-phase raw records ----------
    raw = []  # list of dict with derived fields
    step_total = 0
    per_phase = {}
    for phase in PHASES:
        eps = jsonl(members[f"{phase}/episodes.jsonl"])
        steps = jsonl(members[f"{phase}/steps.jsonl"])
        manifest = json.loads(members[f"{phase}/runtime_manifest.json"])
        man_sha = sha_obj(manifest)
        by_run = defaultdict(list)
        for s in steps:
            by_run[s["run_id"]].append(s)
        ep_ids = [e["run_id"] for e in eps]
        check(f"{phase}:episode_run_ids_unique", len(ep_ids) == len(set(ep_ids)), f"{len(ep_ids)} records")
        orphan = [r for r in by_run if r not in set(ep_ids)]
        orphan_info = []
        for r in orphan:
            kinds = Counter(s["kind"] for s in by_run[r])
            orphan_info.append({"run_id": r, "kinds": dict(kinds)})
        per_phase[phase] = {"episodes": len(eps), "step_events": len(steps),
                            "trace_runs_without_episode_record": orphan_info,
                            "manifest": {k: manifest.get(k) for k in
                                         ("platform", "python", "mujoco_version", "numpy_version",
                                          "torch_version", "lerobot_expected_revision")},
                            "manifest_packages": {k: manifest.get("packages", {}).get(k) for k in
                                                  ("mujoco", "robosuite", "hf_libero", "bddl", "numpy",
                                                   "torch", "lerobot", "gymnasium")},
                            "wrapper_sha256": manifest.get("verified_pinned_source_sha256", {}).get("envs/libero.py")}
        for e in eps:
            rid = e["run_id"]
            tr = by_run.get(rid, [])
            starts = [s for s in tr if s["kind"] == "episode_start"]
            ends = [s for s in tr if s["kind"] == "episode_completed"]
            st = [s for s in tr if s["kind"] == "step"]
            prob = []
            if len(starts) != 1 or len(ends) != 1:
                prob.append(f"starts={len(starts)} ends={len(ends)}")
            ticks = [s["tick"] for s in st]
            if ticks != list(range(len(st))):
                prob.append("ticks not contiguous from 0")
            flags = [bool(s["driver_is_success"]) for s in st]
            vflags = [bool(s["vector_is_success"]) for s in st]
            if flags != vflags:
                prob.append("driver/vector success flags differ")
            trace_success = any(flags)
            first_success = flags.index(True) if trace_success else None
            if trace_success and first_success != len(st) - 1:
                prob.append("success not at final step (no stop on success)")
            if any((s["reward"] == 1.0) != bool(s["driver_is_success"]) for s in st):
                prob.append("reward/success mismatch")
            if any(s["reward"] not in (0.0, 1.0) for s in st):
                prob.append("reward outside {0,1}")
            hz = HORIZON[e["episode_identity"]["suite"]]
            if not trace_success and len(st) != hz:
                prob.append(f"failure length {len(st)} != horizon {hz}")
            if e["task_horizon"] != hz:
                prob.append(f"recorded horizon {e['task_horizon']} != wrapper table {hz}")
            if trace_success != bool(e["success"]):
                prob.append("episode success != trace success")
            if len(st) != e["steps"]:
                prob.append("episode steps != trace steps")
            exp_reason = "success" if trace_success else "task_horizon"
            if e["termination_reason"] != exp_reason:
                prob.append(f"termination_reason {e['termination_reason']} != {exp_reason}")
            # hashes
            act_hashes = []
            for s in st:
                h = array_sha(s["producer_action"])
                if h is None or h != s["producer_action"].get("sha256", h) or h != s["environment_action_sha256"]:
                    prob.append(f"action hash mismatch at tick {s['tick']}")
                    break
                if s["producer_action"]["shape"] != [7]:
                    prob.append("action shape not [7]")
                    break
                act_hashes.append(h)
            if len(act_hashes) == len(st):
                stream = sha_obj(act_hashes)
                if stream != e["action_stream_sha256"] or (ends and stream != ends[0]["action_stream_sha256"]):
                    prob.append("action stream hash mismatch")
            ish = array_sha(e["initial_state"])
            if ish != e["initial_state"]["sha256"]:
                prob.append("initial state hash mismatch (episode record)")
            if starts and array_sha(starts[0]["initial_state"]) != ish:
                prob.append("trace initial state differs from episode initial state")
            if starts and starts[0]["initial_observation_sha256"] != e["initial_observation_sha256"]:
                prob.append("initial observation hash differs trace vs episode")
            ident = e["episode_identity"]
            env_seed = keyed_seed([ident["suite"], ident["task_id"], ident["initial_state_id"]], domain="environment")
            if env_seed != e["environment_seed"]:
                prob.append("environment seed differs from keyed rule")
            gen_ticks = [s["tick"] for s in st if s["new_generation"]]
            expected_gen = list(range(0, len(st), e["h"]))
            if gen_ticks != expected_gen:
                prob.append("generation ticks differ from fixed-prefix schedule")
            if any(s["noise_seed"] != keyed_seed(ident, s["tick"]) for s in st if s["new_generation"]):
                prob.append("noise seed differs from keyed rule")
            if e["runtime_manifest_sha256"] != man_sha:
                prob.append("runtime manifest digest differs")
            pr = planned.get(e["planned_run_id"])
            if pr is None:
                prob.append("planned_run_id not in plan")
            elif (pr["h"], pr["denoising_steps"], pr["block_id"]) != (e["h"], e["denoising_steps"], e["block_id"]) \
                    or pr["episode_identity"] != ident:
                prob.append("episode fields differ from plan row")
            step_total += len(st)
            raw.append({"phase": phase, "run_id": rid, "planned_run_id": e["planned_run_id"],
                        "block_id": e["block_id"], "ident": ident, "cond": (e["h"], e["denoising_steps"]),
                        "success": trace_success, "steps": len(st), "first_success": first_success,
                        "initial_state_sha256": ish, "env_seed": e["environment_seed"],
                        "problems": prob})
    bad = [r for r in raw if r["problems"]]
    check("raw_episode_count", len(raw) == 1122, f"{len(raw)} completed raw records")
    check("raw_step_count", step_total == 172243, f"{step_total} completed steps")
    check("per_episode_trace_consistency", not bad,
          f"{len(bad)} episodes with problems; first: {bad[0]['run_id'] + ' ' + str(bad[0]['problems']) if bad else ''}")

    # ---------- canonical selection, independently ----------
    order = {p: i for i, p in enumerate(PHASES)}
    by_plan = defaultdict(list)
    for r in raw:
        by_plan[r["planned_run_id"]].append(r)
    canonical_sel = {}
    repeats = []
    for pid, rs in by_plan.items():
        rs = sorted(rs, key=lambda r: order[r["phase"]])
        canonical_sel[pid] = rs[0]
        for extra in rs[1:]:
            repeats.append({"planned_run_id": pid, "kept_phase": rs[0]["phase"], "kept_success": rs[0]["success"],
                            "excluded_phase": extra["phase"], "excluded_success": extra["success"],
                            "excluded_steps": extra["steps"], "kept_steps": rs[0]["steps"],
                            "same_initial_state": extra["initial_state_sha256"] == rs[0]["initial_state_sha256"]})
    check("canonical_covers_plan_exactly", set(canonical_sel) == set(planned),
          f"{len(canonical_sel)} selected; missing {len(set(planned) - set(canonical_sel))}; "
          f"extra {len(set(canonical_sel) - set(planned))}")
    canon_file = jsonl(members["confirmation_canonical_episodes.jsonl"])
    canon_ids = {r["planned_run_id"]: r["run_id"] for r in canon_file}
    mismatch = [p for p in planned if canon_ids.get(p) != canonical_sel.get(p, {}).get("run_id")]
    # run_id carries attempt namespace; compare with phase-specific run ids
    check("canonical_matches_archived_canonical_file", not mismatch and len(canon_file) == 1120,
          f"{len(mismatch)} planned runs where the archived canonical file picked a different attempt")
    canon_success_disagree = [r["planned_run_id"] for r in canon_file
                              if r["planned_run_id"] not in canonical_sel
                              or bool(r["success"]) != canonical_sel[r["planned_run_id"]]["success"]]
    check("canonical_file_success_equals_trace", not canon_success_disagree, f"{len(canon_success_disagree)} disagree")

    # ---------- pairing / provenance ----------
    block_states = defaultdict(set)
    taskstate_states = defaultdict(set)
    for r in canonical_sel.values():
        block_states[r["block_id"]].add(r["initial_state_sha256"])
        i = r["ident"]
        taskstate_states[(i["suite"], i["task_id"], i["initial_state_id"])].add(r["initial_state_sha256"])
    check("paired_initial_state_within_block", all(len(v) == 1 for v in block_states.values()),
          f"{sum(len(v) > 1 for v in block_states.values())} blocks with >1 initial state")
    check("same_initial_state_across_seeds", all(len(v) == 1 for v in taskstate_states.values()),
          f"{sum(len(v) > 1 for v in taskstate_states.values())} task-states with >1 initial state; "
          f"{len(taskstate_states)} task-states")
    all_states = defaultdict(set)
    for (s, t, sid), v in taskstate_states.items():
        for h in v:
            all_states[h].add((s, t, sid))
    check("distinct_task_states_have_distinct_initial_states", all(len(v) == 1 for v in all_states.values()),
          f"{sum(len(v) > 1 for v in all_states.values())} hashes shared across task-states")

    # ---------- counts ----------
    cond_counts = {}
    for c in conds:
        rs = [r for r in canonical_sel.values() if r["cond"] == c]
        cond_counts[f"h{c[0]}_d{c[1]}"] = {"n": len(rs), "successes": sum(r["success"] for r in rs)}

    def pair_rows(a, b):
        rows = []
        for blk in sorted({r["block_id"] for r in canonical_sel.values()}):
            ra = [r for r in canonical_sel.values() if r["block_id"] == blk and r["cond"] == a]
            rb = [r for r in canonical_sel.values() if r["block_id"] == blk and r["cond"] == b]
            if len(ra) == 1 and len(rb) == 1:
                rows.append((ra[0], rb[0]))
        return rows

    prim = pair_rows(PRIMARY_SEL, PRIMARY_CMP)
    sel_s = sum(a["success"] for a, _ in prim)
    cmp_s = sum(b["success"] for _, b in prim)
    so = sum(a["success"] and not b["success"] for a, b in prim)
    co = sum(b["success"] and not a["success"] for a, b in prim)
    check("primary_pairs_160", len(prim) == 160, f"{len(prim)} pairs")
    check("primary_counts_139_122", (sel_s, cmp_s) == (139, 122), f"{sel_s}/160 vs {cmp_s}/160")
    check("h5_d2_138", cond_counts["h5_d2"]["successes"] == 138, str(cond_counts["h5_d2"]))
    # primary analysis in the repo uses confirmation + primary_deadline only
    prim_phases = Counter(r["phase"] for a, b in prim for r in (a, b))
    dispatched = sum(r["steps"] for a, b in prim for r in (a, b))
    check("primary_dispatched_actions_50451", dispatched == 50451, str(dispatched))
    ctrl_sel = sum(a["steps"] for a, _ in prim) / 160
    ctrl_cmp = sum(b["steps"] for _, b in prim) / 160
    check("primary_control_step_means", abs(ctrl_sel - 149.025) < 1e-9 and abs(ctrl_cmp - 166.29375) < 1e-9,
          f"{ctrl_sel} vs {ctrl_cmp}")

    tasks = [tuple(t) for t in design["tasks"]]
    per_task = []
    for t in tasks:
        rows = [(a, b) for a, b in prim if (a["ident"]["suite"], a["ident"]["task_id"]) == t]
        per_task.append({"suite": t[0], "task_id": t[1], "n": len(rows),
                         "selected": sum(a["success"] for a, _ in rows),
                         "comparator": sum(b["success"] for _, b in rows),
                         "selected_only": sum(a["success"] and not b["success"] for a, b in rows),
                         "comparator_only": sum(b["success"] and not a["success"] for a, b in rows)})
    rep_tasks = {(r["suite"], r["task_id"]): r for r in report["per_task"]}
    pt_bad = [p for p in per_task if (p["selected"], p["comparator"], p["selected_only"], p["comparator_only"]) !=
              (rep_tasks[(p["suite"], p["task_id"])]["selected_successes"],
               rep_tasks[(p["suite"], p["task_id"])]["comparator_successes"],
               rep_tasks[(p["suite"], p["task_id"])]["selected_only"],
               rep_tasks[(p["suite"], p["task_id"])]["comparator_only"])]
    check("per_task_table_matches_report", not pt_bad, f"{len(pt_bad)} task rows differ")

    # ---------- uncertainty, independent implementations ----------
    # (a) bootstrap: resample initial states within each task, keep both seeds, equal task weights
    by_task_state = defaultdict(list)
    for a, b in prim:
        i = a["ident"]
        by_task_state[((i["suite"], i["task_id"]), i["initial_state_id"])].append(int(a["success"]) - int(b["success"]))
    task_states = {t: sorted({s for (tt, s) in by_task_state if tt == t}) for t in tasks}

    def boot(seed, draws=20000):
        rng = random.Random(seed)
        out = []
        for _ in range(draws):
            tmeans = []
            for t in tasks:
                sts = task_states[t]
                vals = []
                for _k in range(len(sts)):
                    vals += by_task_state[(t, rng.choice(sts))]
                tmeans.append(sum(vals) / len(vals))
            out.append(sum(tmeans) / len(tmeans))
        out.sort()
        return out[int(0.025 * draws)], out[int(0.975 * draws) - 1]
    point = (sel_s - cmp_s) / 160
    b1 = boot(1)
    b2 = boot(2)
    check("bootstrap_interval_approx_report", abs(b1[0] - 0.04375) <= 0.0125 and abs(b1[1] - 0.16875) <= 0.0125,
          f"independent RNG seed 1: [{b1[0]:.4f}, {b1[1]:.4f}]; seed 2: [{b2[0]:.4f}, {b2[1]:.4f}]; "
          f"report [0.04375, 0.16875]")

    # (b) exact two-sided McNemar / sign test on discordant pairs (treats pairs as independent)
    def binom_two_sided(k, n):
        pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
        obs = pk[k]
        return min(1.0, sum(p for p in pk if p <= obs + 1e-15))
    mcnemar_p = binom_two_sided(so, so + co)
    # (c) task-level sign test (tasks as the unit): count of positive vs negative task differences
    pos = sum(p["selected"] > p["comparator"] for p in per_task)
    neg = sum(p["selected"] < p["comparator"] for p in per_task)
    task_sign_p = binom_two_sided(pos, pos + neg) if pos + neg else 1.0
    # (d) leave-one-task-out
    loto = []
    for i, t in enumerate(tasks):
        rest = [p for j, p in enumerate(per_task) if j != i]
        loto.append({"omitted": f"{t[0]}:{t[1]}", "difference": sum((p["selected"] - p["comparator"]) / p["n"]
                                                                    for p in rest) / len(rest)})
    # (e) cluster-robust (state-block) permutation: flip sign of each task-state cluster's paired differences
    clusters = list(by_task_state.values())
    obs_sum = sum(sum(c) for c in clusters)
    rng = random.Random(7)
    perm_ge = 0
    P = 50000
    for _ in range(P):
        s = sum(sum(c) * (1 if rng.random() < 0.5 else -1) for c in clusters)
        if abs(s) >= abs(obs_sum):
            perm_ge += 1
    cluster_perm_p = (perm_ge + 1) / (P + 1)

    result = {
        "schema": "independent_audit.v1",
        "archive_sha256": archive_sha,
        "plan_sha256": hashlib.sha256(open(plan_path, "rb").read()).hexdigest(),
        "implementation": "standard library only; no repository analysis code imported",
        "phases": per_phase,
        "raw_completed_episodes": len(raw),
        "raw_completed_steps": step_total,
        "excluded_repeats": repeats,
        "condition_counts_canonical": cond_counts,
        "primary": {"pairs": len(prim), "selected_h2_d2": sel_s, "comparator_h2_d5": cmp_s,
                    "difference": point, "selected_only": so, "comparator_only": co,
                    "phases_of_primary_records": dict(prim_phases), "dispatched_actions": dispatched,
                    "control_step_means": [ctrl_sel, ctrl_cmp], "per_task": per_task,
                    "bootstrap_states_within_task": {"seed1": b1, "seed2": b2, "draws": 20000},
                    "exact_mcnemar_two_sided_p_pairs_as_independent": mcnemar_p,
                    "task_sign_test": {"positive": pos, "negative": neg, "zero": 8 - pos - neg,
                                       "two_sided_p": task_sign_p},
                    "cluster_sign_flip_permutation_p_task_state_clusters": cluster_perm_p,
                    "leave_one_task_out": loto},
        "checks": checks,
        "findings": findings,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(json.dumps({"checks_passed": sum(c["pass"] for c in checks), "checks": len(checks),
                      "findings": findings, "primary": {k: result["primary"][k] for k in
                      ("selected_h2_d2", "comparator_h2_d5", "selected_only", "comparator_only",
                       "bootstrap_states_within_task", "exact_mcnemar_two_sided_p_pairs_as_independent",
                       "task_sign_test", "cluster_sign_flip_permutation_p_task_state_clusters")},
                      "conditions": cond_counts, "repeats": repeats}, indent=1))
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
