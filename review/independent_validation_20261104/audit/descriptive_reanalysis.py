#!/usr/bin/env python3
"""POST-HOC, EXPLORATORY re-analysis of the archived primary attempts (no replay, no new runs).

Written 4 November 2026 after the primary result was known; nothing here was preregistered and
nothing here may be reported as confirmatory.  Standard library only.

  A. Horizon censoring: success counts if the horizon had been T (successes with tick < T).
  B. Success timing in pairs where both conditions succeed.
  C. Task-as-random-effect interval (t on the 8 task-level differences) next to the archived
     task-conditional bootstrap.
  D. Effective unit: both policy seeds share one initial state per task-state.

Usage: descriptive_reanalysis.py ARCHIVE.tar.gz OUT.json
"""
import io, json, math, sys, tarfile
from collections import defaultdict

PH = ["confirmation", "primary_deadline", "confirmation_continuation_partial", "confirmation_continuation_rest"]
T975 = {7: 2.364624251592785}


def main(archive, out):
    m = {}
    with tarfile.open(archive, "r:gz") as t:
        for x in t.getmembers():
            if x.isfile():
                m[x.name] = t.extractfile(x).read().decode()
    chosen = {}
    for p in PH:
        for l in m[f"{p}/episodes.jsonl"].splitlines():
            e = json.loads(l)
            if (e["h"], e["denoising_steps"]) in ((2, 2), (2, 5)):
                chosen.setdefault(e["planned_run_id"], (p, e))
    want = {e["run_id"] for _, e in chosen.values()}
    first = {}
    for p in PH:
        for l in m[f"{p}/steps.jsonl"].splitlines():
            s = json.loads(l)
            if s["kind"] == "step" and s["run_id"] in want and s["driver_is_success"]:
                first.setdefault(s["run_id"], s["tick"])
    blocks = defaultdict(dict)
    for p, e in chosen.values():
        i = e["episode_identity"]
        blocks[(i["suite"], i["task_id"], i["initial_state_id"], i["policy_seed"])][e["denoising_steps"]] = \
            first.get(e["run_id"])
    horizon = {"libero_object": 280, "libero_goal": 300}
    A = []
    for T in (100, 150, 200, 220, 240, 260, 280, 300):
        s2 = sum(1 for k, v in blocks.items() if v[2] is not None and v[2] < min(T, horizon[k[0]]))
        s5 = sum(1 for k, v in blocks.items() if v[5] is not None and v[5] < min(T, horizon[k[0]]))
        A.append({"horizon_cap": T, "d2": s2, "d5": s5, "difference_pp": round(100 * (s2 - s5) / 160, 2)})
    both = [(v[2], v[5]) for v in blocks.values() if v[2] is not None and v[5] is not None]
    d2_earlier = sum(a < b for a, b in both); d5_earlier = sum(a > b for a, b in both); ties = sum(a == b for a, b in both)
    late = {"d2_successes_after_tick_200": sum(1 for v in blocks.values() if v[2] is not None and v[2] >= 200),
            "d5_successes_after_tick_200": sum(1 for v in blocks.values() if v[5] is not None and v[5] >= 200)}
    tasks = defaultdict(lambda: [0, 0, 0])
    for k, v in blocks.items():
        t = tasks[(k[0], k[1])]
        t[0] += v[2] is not None; t[1] += v[5] is not None; t[2] += 1
    diffs = [(a - b) / n for a, b, n in tasks.values()]
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    half = T975[len(diffs) - 1] * sd / math.sqrt(len(diffs))
    seeds_disagree = defaultdict(int)
    for (s, t, sid, seed), v in blocks.items():
        pass
    per_state = defaultdict(list)
    for (s, t, sid, seed), v in blocks.items():
        per_state[(s, t, sid)].append(((v[2] is not None), (v[5] is not None)))
    same_outcome_across_seeds = {"d2": sum(1 for x in per_state.values() if x[0][0] == x[1][0]),
                                 "d5": sum(1 for x in per_state.values() if x[0][1] == x[1][1]), "of": len(per_state)}
    res = {"status": "POST_HOC_EXPLORATORY", "pairs": len(blocks),
           "A_horizon_censoring": A,
           "B_success_timing_both_succeed": {"pairs": len(both), "d2_earlier": d2_earlier, "d5_earlier": d5_earlier,
                                             "ties": ties, "mean_tick_d2": sum(a for a, _ in both) / len(both),
                                             "mean_tick_d5": sum(b for _, b in both) / len(both), **late},
           "C_task_random_t_interval": {"task_differences": diffs, "mean": mean, "sd": sd,
                                        "t95": [mean - half, mean + half], "df": len(diffs) - 1},
           "D_seed_pairs_same_outcome": same_outcome_across_seeds}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
