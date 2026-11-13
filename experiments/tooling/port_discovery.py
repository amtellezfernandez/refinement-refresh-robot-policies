#!/usr/bin/env python3
"""Audit the completed discovery journals and generate manuscript inputs.

Never reads partial confirmation outcomes or changes the frozen experiment.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_execution import (read_jsonl, reconcile_rollout_manifest,
                               reconcile_rollout_journal, check_initial_state)
from run_paired import file_sha, require, select_schedule, sha

ROOT = Path(__file__).resolve().parents[1] / "runs/refinement_refresh_final_001"
OUT = ROOT / "paper_analysis"


def main():
    OUT.mkdir(exist_ok=True)
    rows = read_jsonl(ROOT / "discovery/episodes.jsonl")
    completed = json.loads((ROOT / "discovery/summary.json").read_text())
    require(completed["status"] == "completed" and len(rows) == 480, "incomplete discovery")
    runtime = reconcile_rollout_manifest(ROOT / "discovery", completed, rows)
    reconcile_rollout_journal(ROOT / "discovery", rows)
    declared = json.loads((ROOT / "discovery_plan.json").read_text())
    require(select_schedule(declared) == runtime["selected_schedule"], "frozen schedule differs")
    blocks = defaultdict(list)
    for r in rows:
        check_initial_state(r["initial_state"])
        blocks[r["block_id"]].append(r)
        g, d = r["counts"]["generations"], r["denoising_steps"]
        require(g == (r["steps"] + r["h"] - 1)//r["h"], "generation count differs")
        require(r["counts"]["denoise_steps"] == g*d, "denoise count differs")
        require(r["counts"]["prefill_calls"] == g and r["counts"]["vision_encoder_calls"] == 2*g
                and r["counts"]["action_expert_calls"] == g*d, "neural stage counts differ")
        for name in ("generation_host_s", "generation_cuda_ms"):
            v = r["timing"][name]
            require(len(v) == g and np.isfinite(v).all() and min(v) >= 0, "invalid timings")
    for block in blocks.values():
        require(len(block)==12 and len({r["initial_state"]["sha256"] for r in block})==1
                and len({r["initial_observation_sha256"] for r in block})==1, "pairing failure")
    monitor = {}
    for phase in ("calibration", "discovery"):
        samples = read_jsonl(ROOT / (phase + "_gpu.jsonl"))
        require(samples and all(set(s["compute_pids"]) <= {s["child_pid"]} for s in samples),
                "other CUDA compute process sampled")
        monitor[phase] = dict(samples=len(samples), duration_s=samples[-1]["time"]-samples[0]["time"],
                              max_sample_gap_s=max(b["time"]-a["time"] for a,b in zip(samples,samples[1:])))
    calibration = json.loads((ROOT/"calibration.json").read_text())
    costs = calibration["generation_flops"]
    budgets = json.loads((ROOT/"frozen_budgets.json").read_text())
    frozen = json.loads((ROOT/"frozen_prediction.json").read_text())
    require(budgets["calibration_sha256"] == file_sha(ROOT/"calibration.json"), "calibration hash differs")
    require(frozen["discovery_sha256"] == file_sha(ROOT/"discovery/episodes.jsonl"), "prediction input differs")
    confirmation = json.loads((ROOT/"confirmation_plan.json").read_text())
    select_schedule(confirmation)
    dev_tasks = {(r["episode_identity"]["suite"],r["episode_identity"]["task_id"]) for r in rows}
    test_tasks = {(r["episode_identity"]["suite"],r["episode_identity"]["task_id"]) for r in confirmation["schedule"]}
    require(not dev_tasks & test_tasks, "task selection/confirmation overlap")
    # One shared state bootstrap preserves all conditions and both seed repeats.
    cells = sorted({(r["h"],r["denoising_steps"]) for r in rows})
    values = np.zeros((4,5,12))
    for ti, task in enumerate(sorted(dev_tasks)):
        for si, state in enumerate(range(10,15)):
            for ci,c in enumerate(cells):
                subset = [r for r in rows if (r["episode_identity"]["suite"],r["episode_identity"]["task_id"])==task
                          and r["episode_identity"]["initial_state_id"]==state
                          and (r["h"],r["denoising_steps"])==c]
                require(len(subset)==2 and {r["episode_identity"]["policy_seed"] for r in subset}=={2000,2001},
                        "state-seed imbalance")
                values[ti,si,ci] = np.mean([r["success"] for r in subset])
    rng = np.random.default_rng(20260913)
    draws = sum(v[rng.integers(0,5,(20000,5))].mean(axis=1)/4 for v in values)
    estimates = values.mean(axis=(0,1))
    bounds = np.quantile(draws,[.025,.975],axis=0)
    cell_results=[]
    for ci,(h,d) in enumerate(cells):
        subset=[r for r in rows if (r["h"],r["denoising_steps"])==(h,d)]
        cell_results.append(dict(h=h,d=d,successes=sum(r["success"] for r in subset),attempts=len(subset),
             success=float(estimates[ci]),descriptive_95_interval=bounds[:,ci].tolist(),
             per_task_successes=[sum(r["success"] for r in subset if r["episode_identity"]["task_id"]==t) for t in (5,6,7,8)],
             relative_rate=costs[str(d)]/h/costs["10"],gf_per_control_step=costs[str(d)]/h/1e9,
             mean_tf_per_attempt=float(np.mean([r["counts"]["generations"]*costs[str(d)]/1e12 for r in subset])),
             mean_rollout_s=float(np.mean([r["timing"]["rollout_wall_s"] for r in subset]))))
    native=cells.index((1,10))
    contrasts=[]
    for ci,c in enumerate(cells):
        contrasts.append(dict(h=c[0],d=c[1],difference=float(estimates[ci]-estimates[native]),
                    descriptive_95_interval=np.quantile(draws[:,ci]-draws[:,native],[.025,.975]).tolist()))
    report=dict(status="audited_completed_discovery",attempts=480,blocks=40,steps=sum(r["steps"] for r in rows),
        cells=cell_results,contrasts_against_native=contrasts,gpu_monitor=monitor,
        checks=["schedule/runtime/summary digests", "every dispatched action and keyed noise", "initial-state pairing",
                "generation and solver counts", "driver/vector outcomes", "frozen selection input", "disjoint confirmation tasks"],
        uncertainty="Post-discovery descriptive bootstrap, 20000 draws, seed 20260913; resample five states within each fixed task, preserving both seeds and all cells; no multiplicity correction or noninferiority claim.",
        source_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in [ROOT/"discovery/episodes.jsonl",ROOT/"discovery/steps.jsonl",
              ROOT/"discovery/runtime_manifest.json",ROOT/"frozen_prediction.json",ROOT/"calibration.json"]})
    (OUT/"discovery_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    table=[r"\begin{tabular}{@{}rrrrrrrr@{}}",r"\toprule",
           r" & & \multicolumn{4}{c}{Successes by task} & & \\",
           r"\cmidrule(lr){3-6}",
           r"$h$ & $d$ & Task 5 & Task 6 & Task 7 & Task 8 & Total & Rate/native\\",r"\midrule"]
    for r in cell_results:
        table.append(f"{r['h']} & {r['d']} & "+" & ".join(f"{x}/10" for x in r["per_task_successes"])+
                     f" & {r['successes']}/40 & {r['relative_rate']:.3f}"+r"\\")
    table += [r"\bottomrule",r"\end{tabular}"]
    (OUT/"discovery_table.tex").write_text("\n".join(table)+"\n")
    resource=[r"\begin{tabular}{@{}rrrrrr@{}}",r"\toprule",
              r"$h$ & $d$ & GFLOPs/step $\downarrow$ & TFLOPs/attempt $\downarrow$ & Rollout s/attempt $\downarrow$ & Success $\uparrow$\\",r"\midrule"]
    for r in cell_results:
        resource.append(f"{r['h']} & {r['d']} & {r['gf_per_control_step']:.1f} & {r['mean_tf_per_attempt']:.1f} & {r['mean_rollout_s']:.1f} & {r['successes']}/40"+r"\\")
    resource += [r"\bottomrule",r"\end{tabular}"]
    (OUT/"discovery_resource_table.tex").write_text("\n".join(resource)+"\n")
    print(json.dumps(dict(attempts=480,steps=report["steps"],gpu_monitor=monitor,cells=cell_results),indent=2))


if __name__=="__main__": main()
