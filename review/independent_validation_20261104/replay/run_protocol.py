#!/usr/bin/env python3
"""Execute the frozen replay protocol (PROTOCOL.md) stage by stage.

  run_protocol.py BUNDLES.jsonl SAMPLE_MANIFEST.json OUT_DIR --release R.json --protocol PROTOCOL.md
                  [--through 0|1|2] [--no-render]

Stage 0  reset fidelity, no actions: one reset per distinct (suite, task, initial_state_id)
         in the population; post-settle state vs recorded initial_state.
Stage 1  stratified sample (manifest), reset_path mode run twice (determinism) + inject mode.
Stage 2  all primary attempts (h2_d2, h2_d5), reset_path + inject.

Each stage writes OUT_DIR/stageN/ and a stage verdict.  A stage starts only if the previous
stage's verdict is in PROCEED; otherwise the orchestrator exits 6 without starting it.
The release gate of replay.py applies to every stage (exit 5).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import libero_replay as lr  # noqa: E402
import replay as R  # noqa: E402

PROCEED = {"PASS_EXACT", "PASS_WITHIN_TOLERANCE"}
PRIMARY = {"h2_d2", "h2_d5"}
SENSITIVITY_DRAWS = 10
SENSITIVITY_SCALE = 1e-9


def stage0(bundles, out, render):
    seen, rows = {}, []
    for b in bundles:
        # archived bundles always use official init states; vector sources (fixtures) key by episode
        k = (b["suite"], b["task_id"], b["reset_source"].get("initial_state_id", b["episode_key"]))
        seen.setdefault(k, b)
    cache, rep, rep_key = {}, None, None
    for k, b in sorted(seen.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if rep_key != k[:2]:
            if rep:
                rep.close()
            rep, rep_key = lr.Replayer(*k[:2], render=render), k[:2]
        pre, src = R.pre_settle_state(b, cache)
        ours = rep.reset(pre, b["environment_seed"])
        rec = lr.array_from_record(b["recorded_initial_state"]).astype(np.float64)
        diff = float(np.max(np.abs(ours - rec))) if ours.shape == rec.shape else None
        rows.append({"task_state": list(k), "exact": bool(ours.shape == rec.shape and np.array_equal(ours, rec)),
                     "max_abs_diff": diff, "init_file_sha256": src.get("file_sha256")})
    if rep:
        rep.close()
    if any(r["max_abs_diff"] is None or r["max_abs_diff"] > R.RESET_TOL for r in rows):
        verdict = "FAIL_RESET"
    elif all(r["exact"] for r in rows):
        verdict = "PASS_EXACT"
    else:
        verdict = "PASS_WITHIN_TOLERANCE"
    res = {"stage": 0, "n_task_states": len(rows), "exact": sum(r["exact"] for r in rows),
           "max_abs_diff_overall": max((r["max_abs_diff"] or 0.0) for r in rows) if rows else None,
           "verdict": verdict, "rows": rows}
    (out / "stage0.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def sensitivity_probe(b, render, rng):
    """Archived outcome present in an ensemble of initial-state perturbations of size 1e-9?"""
    rep = lr.Replayer(b["suite"], b["task_id"], render=render)
    pre, _ = R.pre_settle_state(b, {})
    outcomes = []
    for _ in range(SENSITIVITY_DRAWS):
        rep.reset(pre, b["environment_seed"])
        st = lr.array_from_record(b["recorded_initial_state"]).astype(np.float64)
        st = st + rng.normal(0, SENSITIVITY_SCALE, size=st.shape) * (st != 0)
        rep.inject(st)
        ok = False
        for a in b["actions"]:
            s, _, _ = rep.step(lr.array_from_record(a))
            if s:
                ok = True
                break
        outcomes.append(ok)
    rep.close()
    return outcomes


def episode_stage(n, bundles, out, render, release, protocol, stage0_verdict, double=True):
    sd = out / f"stage{n}"
    sd.mkdir(parents=True, exist_ok=False)
    runs = {}
    for name, mode in (("reset_path_a", "reset_path"), ("reset_path_b", "reset_path"), ("inject", "inject")):
        if name == "reset_path_b" and not double:
            continue
        code = R.run(bundles, sd / name, mode=mode, render=render, release=release, protocol=protocol)
        if code != 0:
            return {"stage": n, "verdict": "REFUSED", "code": code}
        runs[name] = [json.loads(l) for l in (sd / name / "results.jsonl").read_text().splitlines()]
    a = runs["reset_path_a"]
    nondet = []
    if "reset_path_b" in runs:
        for x, y in zip(a, runs["reset_path_b"]):
            if (x["replay"] or {}).get("terminal_state_sha256") != (y["replay"] or {}).get("terminal_state_sha256"):
                nondet.append(x["episode_key"])
    v = Counter(r["verdict"] for r in a)
    bad_input = v["INPUT_INVALID"] + v["DUPLICATE"]
    unrep = v["UNREPLAYABLE"] + v["RESET_MISMATCH"]
    mism = [r for r in a if r["verdict"] == "OUTCOME_MISMATCH"]
    adjud = []
    if mism and stage0_verdict == "PASS_WITHIN_TOLERANCE":
        rng = np.random.default_rng(20261104 + n)
        byk = {b["episode_key"]: b for b in bundles}
        for r in mism:
            outs = sensitivity_probe(byk[r["episode_key"]], render, rng)
            adjud.append({"episode_key": r["episode_key"], "archived": r["replay"]["recorded_success"],
                          "ensemble": outs, "archived_outcome_reachable": r["replay"]["recorded_success"] in outs})
    unexplained = [m for m in mism if stage0_verdict == "PASS_EXACT"] + \
                  [x for x in adjud if not x["archived_outcome_reachable"]]
    if bad_input:
        verdict = "ARCHIVE_DEFECT"
    elif nondet:
        verdict = "UNINTERPRETABLE"
    elif unrep:
        verdict = "UNINTERPRETABLE"
    elif unexplained:
        verdict = "DISAGREEMENT"
    elif mism:
        verdict = "CHAOTIC_INCONCLUSIVE"
    elif v["OUTCOME_MATCH_TICK_DIVERGED"]:
        verdict = "PASS_OUTCOME_ONLY"
    elif stage0_verdict == "PASS_EXACT" and v["REPRODUCED"] == len(a):
        verdict = "PASS_EXACT"
    else:
        verdict = "PASS_WITHIN_TOLERANCE"
    by_cond = {}
    for r, b in zip(a, bundles):
        c = b.get("condition", "fixture")
        rr = r["replay"] or {}
        s = by_cond.setdefault(c, {"n": 0, "archived_successes": 0, "replayed_successes": 0})
        s["n"] += 1
        s["archived_successes"] += bool(b["label"]["success"])
        s["replayed_successes"] += bool(rr.get("replay_success"))
    res = {"stage": n, "n": len(a), "verdict_counts": dict(v),
           "inject_verdict_counts": dict(Counter(r["verdict"] for r in runs["inject"])),
           "nondeterministic": nondet, "adjudication": adjud, "counts_by_condition": by_cond, "verdict": verdict}
    (sd / "stage_verdict.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles")
    ap.add_argument("sample")
    ap.add_argument("out")
    ap.add_argument("--release", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--through", type=int, default=2)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    bundles = R.read_bundles(a.bundles)
    refusal = R.gate(bundles, a.release, a.protocol)
    if refusal:
        print(refusal, file=sys.stderr)
        sys.exit(5)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    render = not a.no_render
    s0 = stage0(bundles, out, render)
    print("stage0", s0["verdict"], s0["exact"], "/", s0["n_task_states"], flush=True)
    if a.through < 1:
        sys.exit(0)
    if s0["verdict"] not in PROCEED:
        print("STOP: stage 0 did not pass; stage 1 not started", file=sys.stderr)
        sys.exit(6)
    sample = set(json.loads(Path(a.sample).read_text())["stage1_sample"])
    s1 = episode_stage(1, [b for b in bundles if b["episode_key"] in sample], out, render, a.release, a.protocol,
                       s0["verdict"])
    print("stage1", s1["verdict"], s1.get("verdict_counts"), flush=True)
    if a.through < 2:
        sys.exit(0)
    if s1["verdict"] not in PROCEED | {"PASS_OUTCOME_ONLY"}:
        print("STOP: stage 1 did not pass; stage 2 not started", file=sys.stderr)
        sys.exit(6)
    s2 = episode_stage(2, [b for b in bundles if b.get("condition") in PRIMARY], out, render, a.release,
                       a.protocol, s0["verdict"], double=False)
    print("stage2", s2["verdict"], s2.get("counts_by_condition"), flush=True)


if __name__ == "__main__":
    main()
