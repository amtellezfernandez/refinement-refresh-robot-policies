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
import math
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import libero_replay as lr  # noqa: E402
import replay as R  # noqa: E402
import governance as gov  # noqa: E402

PROCEED = {"PASS_EXACT", "PASS_WITHIN_TOLERANCE"}
PRIMARY = {"h2_d2", "h2_d5"}
SENSITIVITY_DRAWS = 10
SENSITIVITY_SCALE = 1e-9


def sample_reset_screen(bundles, out, render):
    """Screen every sampled identity, durably stopping at the first failed reset."""
    cache, rep, rep_key, rows = {}, None, None, []
    path = out / "sample_reset_screen.jsonl"
    try:
        with path.open("x", encoding="utf-8") as stream:
            for b in bundles:
                row = {"episode_key": b.get("episode_key"), "environment_seed": b.get("environment_seed")}
                interrupted = None
                try:
                    defects = R.validate(b)
                    if defects:
                        raise ValueError("invalid input: " + "; ".join(defects))
                    key = (b["suite"], b["task_id"])
                    if key != rep_key:
                        if rep:
                            rep.close()
                        rep = lr.Replayer(*key, render=render)
                        rep_key = key
                    pre, src = R.pre_settle_state(b, cache)
                    ours = rep.reset(pre, b["environment_seed"])
                    recorded = lr.array_from_record(b["recorded_initial_state"]).astype(np.float64)
                    diff = float(np.max(np.abs(ours - recorded))) if ours.shape == recorded.shape else None
                    if diff is None or not math.isfinite(diff) or diff > R.RESET_TOL:
                        raise ValueError("sample reset mismatch: " + str(diff))
                    row.update(exact=bool(np.array_equal(ours, recorded)), max_abs_diff=diff,
                               init_file_sha256=src.get("file_sha256"), verdict="PASS")
                except Exception:
                    row.update(verdict="FAIL_SAMPLE_RESET", error=traceback.format_exc())
                except BaseException as exc:
                    row.update(verdict="INTERRUPTED", error=traceback.format_exc())
                    interrupted = exc
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                gov.checkpoint_evidence(out)
                if interrupted is not None:
                    raise interrupted
                if row["verdict"] != "PASS":
                    return {"verdict": "FAIL_SAMPLE_RESET", "n": len(rows)}
    finally:
        if rep:
            rep.close()
        gov.checkpoint_evidence(out)
    return {"verdict": "PASS_EXACT" if all(r["exact"] for r in rows) else "PASS_WITHIN_TOLERANCE",
            "n": len(rows)}


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
    if any(r["max_abs_diff"] is None or not math.isfinite(r["max_abs_diff"])
           or r["max_abs_diff"] > R.RESET_TOL for r in rows):
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


def episode_stage(n, bundles, out, render, release, protocol, stage0_verdict, double=True,
                  artifact_paths=None):
    sd = out / f"stage{n}"
    sd.mkdir(parents=True, exist_ok=False)
    keys = [b.get("episode_key") for b in bundles]
    if (not keys or any(not isinstance(k, str) or not k for k in keys)
            or len(keys) != len(set(keys))):
        stopped = {"stage": n, "verdict": "IMPLEMENTATION_FAILURE", "error": "invalid scheduled identities"}
        (sd / "stage_verdict.json").write_text(json.dumps(stopped, indent=1) + "\n")
        if artifact_paths:
            gov.checkpoint_evidence(artifact_paths["evidence_root"])
        return stopped
    runs = {}
    for name, mode in (("reset_path_a", "reset_path"), ("reset_path_b", "reset_path"), ("inject", "inject")):
        if name == "reset_path_b" and not double:
            continue
        try:
            code = R.run(bundles, sd / name, mode=mode, render=render, release=release,
                         protocol=protocol, stage=n, artifact_paths=artifact_paths,
                         capability=gov._ORCHESTRATOR_CAPABILITY if artifact_paths else None,
                         reference_records=runs.get("reset_path_a") if artifact_paths and artifact_paths.get("mode") == "sample_first_57_v1" and name != "reset_path_a" else None,
                         exact_reset=stage0_verdict == "PASS_EXACT")
        except (KeyboardInterrupt, SystemExit):
            stopped = {"stage": n, "verdict": "INTERRUPTED", "arm": name}
            (sd / "stage_verdict.json").write_text(json.dumps(stopped, indent=1) + "\n")
            if artifact_paths:
                gov.checkpoint_evidence(artifact_paths["evidence_root"])
            raise
        except Exception:
            stopped = {"stage": n, "verdict": "IMPLEMENTATION_FAILURE", "arm": name,
                       "error": traceback.format_exc()}
            (sd / "stage_verdict.json").write_text(json.dumps(stopped, indent=1) + "\n")
            if artifact_paths:
                gov.checkpoint_evidence(artifact_paths["evidence_root"])
            return stopped
        if code != 0:
            result_path = sd / name / "results.jsonl"
            last = None
            if result_path.exists():
                try:
                    lines = result_path.read_text().splitlines()
                    if lines:
                        last = json.loads(lines[-1])
                except (OSError, ValueError):
                    last = None
            last_verdict = last.get("verdict") if isinstance(last, dict) else None
            if code == 5:
                verdict = "REFUSED"
            elif last_verdict in {"INPUT_INVALID", "DUPLICATE"}:
                verdict = "ARCHIVE_DEFECT"
            elif last_verdict == "OUTCOME_MISMATCH":
                verdict = "STOPPED"
            elif last_verdict in {"RESET_MISMATCH", "UNREPLAYABLE", "NONDETERMINISTIC", "ARM_DISAGREEMENT"}:
                verdict = "UNINTERPRETABLE"
            else:
                verdict = "IMPLEMENTATION_FAILURE"
            stopped = {"stage": n, "verdict": verdict, "code": code,
                       "arm": name, "last_result": last}
            if last_verdict == "OUTCOME_MISMATCH":
                stopped["note"] = "A mismatch requires protocol-defined adjudication; no adjudication was run."
            (sd / "stage_verdict.json").write_text(json.dumps(stopped, indent=1) + "\n")
            if artifact_paths:
                gov.checkpoint_evidence(artifact_paths["evidence_root"])
            return stopped
        try:
            rows = [json.loads(l) for l in (sd / name / "results.jsonl").read_text().splitlines()]
            if len(rows) != len(keys):
                raise ValueError("result count or scheduled identities invalid")
            for row, key, bundle in zip(rows, keys, bundles):
                if not isinstance(row, dict) or row.get("episode_key") != key or row.get("verdict") not in R.VERDICTS:
                    raise ValueError("invalid, duplicate, unexpected, or out-of-order result identity/verdict")
                if ((row.get("input_defects") and row["verdict"] != "INPUT_INVALID")
                        or (row.get("duplicate") and row["verdict"] not in {"INPUT_INVALID", "DUPLICATE"})):
                    raise ValueError("result verdict suppresses an input defect or duplicate")
                if row["verdict"] not in {"INPUT_INVALID", "DUPLICATE", "UNREPLAYABLE", "RESET_MISMATCH"}:
                    replay = row.get("replay")
                    required = {"terminal_state_sha256", "replay_success", "recorded_success",
                                "replay_first_success_tick", "recorded_first_success_tick", "first_divergent_tick"}
                    if not isinstance(replay, dict) or not required.issubset(replay):
                        raise ValueError("result row lacks required replay fields")
                    if (not isinstance(replay["terminal_state_sha256"], str)
                            or not replay["terminal_state_sha256"]
                            or type(replay["replay_success"]) is not bool
                            or type(replay["recorded_success"]) is not bool):
                        raise ValueError("result row has invalid replay values")
                    if replay["recorded_success"] != bundle.get("label", {}).get("success"):
                        raise ValueError("result row recorded outcome differs from scheduled input")
                    for field in ("replay_first_success_tick", "recorded_first_success_tick", "first_divergent_tick"):
                        tick = replay[field]
                        if tick is not None and (type(tick) is not int or tick < 0):
                            raise ValueError("result row has invalid tick")
                    if (replay["replay_success"] != (replay["replay_first_success_tick"] is not None)
                            or replay["recorded_success"] != (replay["recorded_first_success_tick"] is not None)):
                        raise ValueError("result row has inconsistent success and tick")
                    verdict = row["verdict"]
                    matches = replay["replay_success"] == replay["recorded_success"]
                    divergent = replay["first_divergent_tick"] is not None
                    if ((verdict == "OUTCOME_MISMATCH" and matches)
                            or (verdict == "OUTCOME_MATCH_TICK_DIVERGED" and (not matches or not divergent))
                            or (verdict in {"REPRODUCED", "REPRODUCED_RESET_DRIFT"}
                                and (not matches or divergent or replay["replay_first_success_tick"]
                                     != replay["recorded_first_success_tick"]))):
                        raise ValueError("result row verdict contradicts replay values")
            runs[name] = rows
        except (OSError, ValueError, TypeError, KeyError):
            stopped = {"stage": n, "verdict": "IMPLEMENTATION_FAILURE", "arm": name,
                       "error": traceback.format_exc()}
            (sd / "stage_verdict.json").write_text(json.dumps(stopped, indent=1) + "\n")
            if artifact_paths:
                gov.checkpoint_evidence(artifact_paths["evidence_root"])
            return stopped
    a = runs["reset_path_a"]
    nondet = []
    if "reset_path_b" in runs:
        for x, y in zip(a, runs["reset_path_b"]):
            if any((x["replay"] or {}).get(field) != (y["replay"] or {}).get(field) for field in
                   ("terminal_state_sha256", "replay_success", "replay_first_success_tick", "first_divergent_tick")):
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
    ap.add_argument("--through", type=int, choices=(0, 1, 2), default=2)
    ap.add_argument("--execution-mode", choices=("frozen", "sample_first_57_v1"), default="frozen")
    ap.add_argument("--archive", help="original confirmation archive (required for sample-first)")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    bundles = R.read_bundles(a.bundles)
    if not bundles:
        print("REFUSED: empty replay population", file=sys.stderr)
        sys.exit(5)
    render = not a.no_render
    real = gov.is_real_archive(bundles)
    if a.execution_mode != "frozen" and not real:
        print("REFUSED: sample-first mode requires the pinned archived population", file=sys.stderr)
        sys.exit(5)
    if a.execution_mode == "sample_first_57_v1":
        try:
            if not a.archive or gov.sha_file(a.archive) != gov.ARCHIVE_SHA256:
                raise gov.ReplayRefused("sample-first requires the pinned original archive")
        except (gov.ReplayRefused, OSError) as error:
            print("REFUSED: {}".format(error), file=sys.stderr)
            sys.exit(5)
    artifact_paths = None
    if real:
        try:
            report = gov.verify_run(bundles, a.bundles, a.sample, a.release,
                                    a.protocol, a.through, render, mode=a.execution_mode,
                                    archive_path=a.archive)
        except (gov.ReplayRefused, OSError, ValueError, TypeError, KeyError) as error:
            print("REFUSED: {}".format(error), file=sys.stderr)
            sys.exit(5)
        artifact_paths = {"bundles": a.bundles, "sample": a.sample,
                          "release_sha256": report["release_sha256"], "mode": a.execution_mode,
                          "evidence_root": str(Path(a.out).resolve())}
    else:
        # Retain Warren's existing fixture-only fire drills. These fixtures
        # have no standing as scientific evidence or approval for paper data.
        refusal = R.gate(bundles, a.release, a.protocol)
        if refusal:
            print(refusal, file=sys.stderr)
            sys.exit(5)
        report = None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if report is not None:
        report["command_line"] = sys.argv
        report["output_directory"] = str(out.resolve())
        if a.execution_mode == "sample_first_57_v1":
            report["source_archive"] = {"path": str(Path(a.archive).resolve()),
                                        "sha256": gov.ARCHIVE_SHA256}
            sample_ids = json.loads(Path(a.sample).read_text("utf-8"))["stage1_sample"]
            by_key = {b["episode_key"]: b for b in bundles}
            sample = [by_key[k] for k in sample_ids]
            report["sample_attempts"] = [{"episode_key": b["episode_key"],
                                          "environment_seed": b["environment_seed"]} for b in sample]
        gov.record_preflight(out / "counted_run_identity.json", report)
        gov.checkpoint_evidence(out)
    if a.execution_mode == "sample_first_57_v1":
        try:
            screen = sample_reset_screen(sample, out, render)
            if screen["n"] != 57 or screen["verdict"] not in PROCEED:
                print("STOP: sample reset screen failed", file=sys.stderr)
                sys.exit(6)
            result = episode_stage(1, sample, out, render, a.release, a.protocol,
                                   screen["verdict"], artifact_paths=artifact_paths)
            if result["verdict"] not in PROCEED | {"PASS_OUTCOME_ONLY"}:
                print("STOP: sample replay did not pass", file=sys.stderr)
                sys.exit(6)
            sys.exit(0)
        finally:
            gov.checkpoint_evidence(out)
    s0 = stage0(bundles, out, render)
    print("stage0", s0["verdict"], s0["exact"], "/", s0["n_task_states"], flush=True)
    if real and s0["n_task_states"] != 80:
        print("STOP: Stage 0 did not check all 80 task-states", file=sys.stderr)
        sys.exit(6)
    if s0["verdict"] not in PROCEED:
        print("STOP: Stage 0 did not pass; no further stage started", file=sys.stderr)
        sys.exit(6)
    if a.through == 0:
        sys.exit(0)
    manifest = json.loads(Path(a.sample).read_text("utf-8"))
    sample_ids = set(manifest["stage1_sample"])
    sample = [b for b in bundles if b["episode_key"] in sample_ids]
    if real and len(sample) != 57:
        print("STOP: Stage 1 must contain exactly 57 frozen attempts", file=sys.stderr)
        sys.exit(6)
    s1 = episode_stage(1, sample, out, render, a.release, a.protocol,
                       s0["verdict"], artifact_paths=artifact_paths)
    print("stage1", s1["verdict"], s1.get("verdict_counts"), flush=True)
    if s1["verdict"] not in PROCEED | {"PASS_OUTCOME_ONLY"}:
        print("STOP: Stage 1 failed; Stage 2 not started", file=sys.stderr)
        sys.exit(6)
    if a.through == 1:
        sys.exit(0)
    primary = [b for b in bundles if b.get("condition") in PRIMARY]
    if real and len(primary) != 320:
        print("STOP: Stage 2 must contain exactly 320 primary attempts", file=sys.stderr)
        sys.exit(6)
    s2 = episode_stage(2, primary, out, render, a.release, a.protocol,
                       s0["verdict"], double=False, artifact_paths=artifact_paths)
    print("stage2", s2["verdict"], s2.get("counts_by_condition"), flush=True)
    if s2["verdict"] not in PROCEED | {"PASS_OUTCOME_ONLY"}:
        print("STOP: Stage 2 did not confirm archived outcomes", file=sys.stderr)
        sys.exit(6)
    if real:
        counts = s2["counts_by_condition"]
        if (counts.get("h2_d2", {}).get("n") != 160
                or counts.get("h2_d5", {}).get("n") != 160
                or counts["h2_d2"]["replayed_successes"] != 139
                or counts["h2_d5"]["replayed_successes"] != 122):
            print("STOP: replay-derived primary counts do not match the frozen study", file=sys.stderr)
            sys.exit(6)
    sys.exit(0)


if __name__ == "__main__":
    main()
