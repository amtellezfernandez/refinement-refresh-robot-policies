#!/usr/bin/env python3
"""Report the frozen primary pair from retained records; never runs inference.

Task disaggregation, leave-one-task-out sensitivity, and expenditure intervals
are descriptive additions made after the primary outcomes were observed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_execution import (canonical_sha, check_initial_state, read_jsonl,
                               reconcile_rollout_journal)
from run_paired import file_sha, require

PRIMARY = {(2, 2), (2, 5)}
METRICS = ("success", "reference_tflops", "rollout_seconds", "control_steps")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(row):
    i = row["episode_identity"]
    return (i["suite"], i["task_id"], i["initial_state_id"], i["policy_seed"])


def summarize(root):
    rows = read_jsonl(root / "episodes.jsonl")
    plan = json.loads((root / "primary_plan.json").read_text())
    costs = json.loads((root / "calibration.json").read_text())["generation_flops"]
    require(len(rows) == 320, "primary sample must contain all 320 attempts")
    expected = {r["planned_run_id"]: r for r in plan}
    require(len(expected) == 320 and len({r['planned_run_id'] for r in rows}) == 320,
            "duplicate or incomplete planned identities")
    require(set(expected) == {r["planned_run_id"] for r in rows}, "plan coverage differs")
    blocks = defaultdict(dict)
    for r in rows:
        p = expected[r["planned_run_id"]]
        require(identity(r) == identity(p) and r["h"] == p["h"] and
                r["denoising_steps"] == p["denoising_steps"], "planned identity differs")
        require((r["h"], r["denoising_steps"]) in PRIMARY, "non-primary result")
        check_initial_state(r["initial_state"])
        g = (r["steps"] + r["h"] - 1) // r["h"]
        require(r["counts"] == dict(generations=g, denoise_steps=g*r["denoising_steps"],
                    vision_encoder_calls=2*g, prefill_calls=g,
                    action_expert_calls=g*r["denoising_steps"]), "neural counts differ")
        require(all(len(r["timing"][f]) == g and np.isfinite(r["timing"][f]).all()
                    and min(r["timing"][f]) >= 0
                    for f in ("generation_host_s", "generation_cuda_ms")), "timing fields invalid")
        require(r["denoising_steps"] not in blocks[identity(r)], "duplicate pair member")
        blocks[identity(r)][r["denoising_steps"]] = r
    require(len(blocks) == 160, "incorrect paired block count")
    for b in blocks.values():
        require(set(b) == {2, 5}, "incomplete condition pair")
        require(b[2]["initial_state"]["sha256"] == b[5]["initial_state"]["sha256"] and
                b[2]["initial_observation_sha256"] == b[5]["initial_observation_sha256"],
                "paired reset or observation differs")
    reconcile_rollout_journal(root, rows)
    tasks = sorted({k[:2] for k in blocks})
    require(tasks == [(s, t) for s in ("libero_goal", "libero_object") for t in range(5, 9)],
            "task set differs")
    values = np.zeros((8, 10, 2, 4))
    per_task, states = [], []
    for ti, task in enumerate(tasks):
        joint = Counter()
        for si, state in enumerate(range(20, 30)):
            pair_seeds = [blocks[(*task, state, seed)] for seed in (3000, 3001)]
            for b in pair_seeds:
                joint[(b[2]["success"], b[5]["success"])] += 1
            for ci, d in enumerate((2, 5)):
                pair = [b[d] for b in pair_seeds]
                values[ti, si, ci] = np.mean([
                    [r["success"], r["counts"]["generations"] * costs[str(d)] / 1e12,
                     r["timing"]["rollout_wall_s"], r["steps"]] for r in pair], axis=0)
            states.append(dict(suite=task[0], task_id=task[1], state_id=state,
                               selected_seed_mean=values[ti, si, 0].tolist(),
                               comparator_seed_mean=values[ti, si, 1].tolist(),
                               paired_difference=(values[ti, si, 0]-values[ti, si, 1]).tolist()))
        per_task.append(dict(suite=task[0], task_id=task[1], attempts_per_condition=20,
            selected_successes=int(joint[(True, True)] + joint[(True, False)]),
            comparator_successes=int(joint[(True, True)] + joint[(False, True)]),
            both_success=int(joint[(True, True)]), both_failure=int(joint[(False, False)]),
            selected_only=int(joint[(True, False)]), comparator_only=int(joint[(False, True)]),
            difference=float((values[ti, :, 0, 0]-values[ti, :, 1, 0]).mean())))
    rng = np.random.default_rng(20260912)
    draws = np.zeros((20000, 2, 4))
    # Same task order and RNG draws as the original primary analysis.
    for v in values:
        draws += v[rng.integers(0, 10, (20000, 10))].mean(axis=1)/8
    estimates = values.mean(axis=(0, 1))
    differences = estimates[0]-estimates[1]
    intervals = np.quantile(draws[:, 0]-draws[:, 1], [.025, .975], axis=0)
    summary = {name: dict(selected_mean=float(estimates[0, i]),
               comparator_mean=float(estimates[1, i]), paired_difference=float(differences[i]),
               descriptive_95_interval=intervals[:, i].tolist()) for i, name in enumerate(METRICS)}
    success_diffs = values[:, :, 0, 0].mean(axis=1)-values[:, :, 1, 0].mean(axis=1)
    loto = [dict(omitted_suite=t[0], omitted_task_id=t[1],
                 difference=float(np.delete(success_diffs, i).mean())) for i, t in enumerate(tasks)]
    termination = {str(d): dict(Counter(r["termination_reason"] for r in rows
                                       if r["denoising_steps"] == d)) for d in (2, 5)}
    calibration = json.loads((root / "calibration.json").read_text())
    timing_audit = []
    for d, entries in sorted(calibration["timings"].items(), key=lambda x: int(x[0])):
        host = np.array([r["host_s"]*1000 for r in entries])
        cuda = np.array([r["cuda_ms"] for r in entries])
        require(len(entries) == 60 and not np.array_equal(host, cuda), "timing arrays duplicated")
        timing_audit.append(dict(d=int(d), calls=60, host_mean_ms=float(host.mean()),
             cuda_mean_ms=float(cuda.mean()), mean_host_minus_cuda_ms=float((host-cuda).mean())))
    result = dict(attempts=320, task_state_blocks=80, task_state_seed_blocks=160,
        dispatched_actions=sum(r["steps"] for r in rows), metrics=summary, per_task=per_task,
        leave_one_task_out=loto, termination_reasons=termination, calibration_timing_audit=timing_audit,
        bootstrap=dict(draws=20000, seed=20260912, resampling="states within each fixed task; both seeds and paired conditions retained; equal task weights"),
        reporting_status="Primary success comparison unchanged. Task disaggregation, leave-one-task-out sensitivity, and expenditure intervals added descriptively after outcomes.",
        timing_status="Observed synchronous rollout times are descriptive: GPU process monitoring was not retained for priority completion.",
        checks=["exact original primary plan coverage", "paired initial states and observations",
                "neural stage counts", "all dispatched actions and terminal outcomes",
                "separate host and CUDA generation timing arrays"])
    require(np.isclose(summary["success"]["paired_difference"], 0.10625), "primary estimate differs")
    require(np.allclose(summary["success"]["descriptive_95_interval"], [.04375, .16875]),
            "primary interval differs")
    return result, states


def write_tables(result, output):
    output.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Task & Selected $d=2$ & Comparator $d=5$ & Gap (pp) & Wins & Losses\\", r"\midrule"]
    for r in result["per_task"]:
        label = r["suite"].removeprefix("libero_").title()+" "+str(r["task_id"])
        lines.append(f"{label} & {r['selected_successes']}/20 & {r['comparator_successes']}/20 & "
                     f"${100*r['difference']:+.1f}$ & {r['selected_only']} & {r['comparator_only']}"+r"\\")
    totals = {k:sum(r[k] for r in result['per_task']) for k in
              ('selected_successes', 'comparator_successes', 'selected_only', 'comparator_only')}
    lines += [r"\midrule", f"All tasks & {totals['selected_successes']}/160 & "
              f"{totals['comparator_successes']}/160 & "
              f"${100*result['metrics']['success']['paired_difference']:+.1f}$ & "
              f"{totals['selected_only']} & {totals['comparator_only']}"+r"\\",
              r"\bottomrule", r"\end{tabular}"]
    (output / "primary_task_table.tex").write_text("\n".join(lines)+"\n")
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Measure per attempt & $d=2$ & $d=5$ & Paired difference [95\% interval]\\", r"\midrule"]
    for key, label in [("reference_tflops", "Reference TFLOPs"),
                       ("rollout_seconds", "Observed rollout seconds"), ("control_steps", "Control steps")]:
        r=result["metrics"][key]; lo, hi=r["descriptive_95_interval"]
        lines.append(f"{label} & {r['selected_mean']:.1f} & {r['comparator_mean']:.1f} & "
                     f"${r['paired_difference']:+.1f}$ $[{lo:+.1f},{hi:+.1f}]$"+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (output / "primary_resource_uncertainty.tex").write_text("\n".join(lines)+"\n")


def prepare(args):
    source, root, evidence = args.source, args.output, args.evidence
    root.mkdir(parents=True, exist_ok=True)
    rows, events, runtime = [], [], {}
    for phase in ("confirmation", "primary_deadline"):
        phase_rows = [r for r in read_jsonl(source/phase/"episodes.jsonl")
                      if (r["h"],r["denoising_steps"]) in PRIMARY]
        manifest = json.loads((source/phase/"runtime_manifest.json").read_text())
        require(all(r['runtime_manifest_sha256'] == canonical_sha(manifest) for r in phase_rows),
                "source manifest digest differs")
        rows += phase_rows
        included={r["run_id"] for r in phase_rows}
        events += [e for e in read_jsonl(source/phase/"steps.jsonl") if e.get("run_id") in included]
        runtime[phase] = {k:manifest[k] for k in ["python", "platform", "torch_version", "cuda_version",
            "precision", "numpy_version", "mujoco_version", "gpu", "control", "noise_contract",
            "cost_boundary", "native_checkpoint_config", "effective_policy_config",
            "checkpoint_expected_revision", "checkpoint_files_sha256", "lerobot_expected_revision",
            "verified_pinned_source_sha256"]}
        runtime[phase]["source_sha256_by_filename"]={Path(k).name:v for k,v in manifest["source_sha256"].items()}
        runtime[phase]["package_metadata"]={k:manifest["packages"][k]
            for k in ("torch", "numpy", "transformers", "robosuite", "hf_libero")}
        runtime[phase]["original_manifest_canonical_sha256"]=canonical_sha(manifest)
        runtime[phase]["included_attempts"]=len(phase_rows)
        runtime[phase]["warmup_generations_per_condition"]=manifest["argv"]["warmup"]
        require(file_sha(Path(__file__).parent/"run_paired.py") ==
                runtime[phase]["source_sha256_by_filename"]["run_paired.py"],
                "bundled evaluator is not the recorded evaluator revision")
    runtime["notes"] = ["Derived runtime extract: machine paths and unrelated package inventory omitted.",
        "Loaded torch_version and numpy_version take precedence over conflicting installed distribution metadata.",
        "use_amp=false, autocast=false, TF32=false. Parameter dtype inventory was not retained.",
        "Pinned language/action attention uses its eager implementation with float32 attention scores; vision backend not separately recorded.",
        "Process monitoring was retained only for the initial transfer phase, not priority completion."]
    for name, data in [("episodes.jsonl", rows), ("steps.jsonl", events)]:
        with (root/name).open("w") as stream:
            for row in data:
                stream.write(json.dumps(row, separators=(",", ":"))+"\n")
    planned=json.loads((source/"confirmation_plan.json").read_text())["schedule"]
    write_json(root/"primary_plan.json", [r for r in planned if (r['h'],r['denoising_steps']) in PRIMARY])
    write_json(root/"runtime_provenance.json", runtime)
    shutil.copyfile(evidence/"calibration.json", root/"calibration.json")
    write_json(root/"source_hashes.json", {str(p.relative_to(source)):file_sha(p)
        for p in [source/"confirmation/episodes.jsonl", source/"confirmation/steps.jsonl",
                  source/"primary_deadline/episodes.jsonl", source/"primary_deadline/steps.jsonl",
                  source/"deadline_schedule_amendment.json"]})
    for name in ("frozen_prediction.json", "frozen_budgets.json", "language_shape_audit.json",
                 "language_shape_flops.json"):
        data=json.loads((evidence/name).read_text())
        if name == "language_shape_audit.json":
            data['checkpoint']="HuggingFaceVLA/smolvla_libero (revision in runtime_provenance.json)"
        write_json(root/name, data)
    for name in ("primary_reporting.py", "analyze_execution.py", "run_paired.py"):
        shutil.copyfile(Path(__file__).parent/name, root/name)
    result, states = summarize(root)
    write_json(root/"primary_report.json", result)
    write_json(args.tables/"primary_report.json", result)
    write_json(root/"paired_state_values.json", dict(metric_order=METRICS, states=states))
    write_tables(result, args.tables)
    readme = """# Primary transfer analysis supplement

This package contains all 320 originally planned primary attempts, their action
journals, 80 paired state summaries (both policy seeds retained), the original
primary schedule, reference calibration, frozen predictions, and a derived
runtime-provenance extract. It excludes unfinished secondary comparisons.

Run with Python 3.12+, NumPy and Matplotlib:

    python primary_reporting.py --check .

This verifies plan coverage, paired resets/observations, neural-stage counts,
every dispatched action and terminal outcome, then reproduces success counts,
the task-conditional primary bootstrap, per-task joint counts, descriptive
leave-one-task-out sensitivity, and expenditure intervals. It runs no inference.
The original source digests identify the pre-extraction records; the complete
machine-specific runtime manifests are not included. This is internal record
consistency, not external attestation of simulator execution.

`primary_report.json` records analysis definitions and post-outcome additions.
`paired_state_values.json` contains seed-averaged condition values and differences.
Rate and episode arithmetic use the original length-20 reference calibration,
not the task-shape-specific audit counts. Observed synchronous rollout times are
descriptive because process monitoring was absent during priority completion.
All means include unsuccessful attempts. Intervals condition on eight fixed tasks,
are not multiplicity-adjusted, and do not establish new-task-population effects.

The runtime extract reports loaded module versions separately from installed
distribution metadata, which disagree for Torch and NumPy. Parameter dtype and
vision attention-backend inventories were not retained. The evaluator source is
provided for inspection; checkpoint weights, simulator assets and a complete
training/evaluation environment are not bundled.

The response-model candidates are the Cartesian product h={1,2,3,5},
d={2,3,5,8,10,20}. Allocation features are [1, log(h), log(d/10),
log(h)^2, log(d/10)^2, log(h)*log(d/10)]; compute-only features are
[1, log(Fgen(d)/(h*Fgen(10))), log(Fgen(d)/(h*Fgen(10)))^2]. There is no further
centering or standardization. The objective is summed binary logistic loss plus
ridge/2 times the squared non-intercept coefficients, and 1e-6/2 times the squared
intercept. Ridge is 1 or 10; Newton fitting uses at most 100 iterations, clipped
logits [-30,30], and max coefficient-update tolerance 1e-8. Model selection uses
four whole-task folds and mean natural-log binary cross-entropy. This procedure
has no demonstrated selection advantage over the simple rule.
"""
    (root/"README.md").write_text(readme)
    write_json(root/"MANIFEST.json", {p.name:file_sha(p) for p in sorted(root.iterdir())
                                    if p.is_file() and p.name != "MANIFEST.json"})
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.iterdir()):
            if p.is_file():
                archive.write(p, "primary_transfer_supplement/"+p.name)
    print(json.dumps(result, indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--archive", type=Path)
    args=parser.parse_args()
    if args.check:
        for name, digest in json.loads((args.check/"MANIFEST.json").read_text()).items():
            require(file_sha(args.check/name) == digest, "package digest differs: "+name)
        actual, states=summarize(args.check)
        expected=json.loads((args.check/"primary_report.json").read_text())
        require(actual == expected, "recomputed primary report differs")
        require(states == json.loads((args.check/"paired_state_values.json").read_text())["states"],
                "paired state values differ")
        print(f"PASS: {actual['attempts']} primary attempts, {actual['dispatched_actions']} actions; all analysis reproduced.")
    else:
        require(all((args.source,args.evidence,args.output,args.tables,args.archive)), "missing build paths")
        prepare(args)


if __name__ == "__main__":
    main()
