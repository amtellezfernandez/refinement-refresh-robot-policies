#!/usr/bin/env python3
"""Unattended frozen-checkpoint discovery, prospective selection and confirmation.

No outcome-dependent episode caps. Exclusive artifacts; failures retain journals.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from run_paired import canonical, file_sha, require, select_schedule, sha

GRID = list(itertools.product((1, 2, 5), (2, 5, 10, 20)))
UNTESTED = sorted(set(itertools.product((1, 2, 3, 5), (2, 3, 5, 8, 10, 20))) - set(GRID))
DEV = [("libero_spatial", i) for i in (5, 6, 7, 8)]
CONF = [(suite, i) for suite in ("libero_object", "libero_goal") for i in (5, 6, 7, 8)]
SEEDS = [2000, 2001]


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, default=str, allow_nan=False)
        f.write("\n")


def plan(tasks, states, seeds, conditions, phase):
    identities = [(s, t, i, k) for s, t in tasks for i in states for k in seeds]
    identities.sort(key=lambda x: sha([20260912, phase, "blocks", x]))
    rows = []
    for bi, ident in enumerate(identities):
        s, t, i, k = ident
        bid = f"{phase}.{s}.task{t:03d}.state{i:03d}.seed{k}"
        ordered = sorted(conditions, key=lambda c: sha([20260912, phase, ident, c]))
        for ci, (h, d) in enumerate(ordered):
            rows.append(dict(schedule_index=len(rows), block_index=bi, condition_index=ci,
                             block_id=bid, episode_identity=dict(suite=s, task_id=t, initial_state_id=i,
                                                                policy_seed=k),
                             condition_id=f"h{h}_d{d}", planned_run_id=f"{bid}.h{h}_d{d}",
                             h=h, denoising_steps=d, native_reference=(h, d) == (1, 10)))
    result = dict(schema="robopad.proposed_evaluation_plan", schema_version="1.0",
                  design_revision="2026-09-12-final-campaign", phase=phase,
                  design=dict(tasks=tasks, initial_state_ids=list(states), policy_seeds=seeds,
                              conditions=[dict(h=h, denoising_steps=d) for h, d in conditions],
                              task_count=len(tasks), block_count=len(identities),
                              conditions_per_block=len(conditions), planned_attempt_count=len(rows)),
                  scheduling=dict(seed=20260912, schedule_sha256=sha(rows)), schedule=rows)
    select_schedule(result)
    return result


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                            text=True, capture_output=True, check=True, timeout=15)
    return [int(line.split(",")[0]) for line in result.stdout.splitlines() if line.strip()]


def run_child(args, name, argv):
    require(not gpu_processes(), "Unrelated CUDA compute process before " + name)
    with (args.output / (name + ".log")).open("x", buffering=1) as log, \
         (args.output / (name + "_gpu.jsonl")).open("x", buffering=1) as monitor:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"START {name} PID {child.pid}", flush=True)
        try:
            while child.poll() is None:
                processes = gpu_processes()
                monitor.write(json.dumps(dict(time=time.time(), child_pid=child.pid,
                                               compute_pids=processes)) + "\n")
                if any(pid != child.pid for pid in processes):
                    raise RuntimeError(f"GPU contention detected in {name}: {processes}")
                time.sleep(5)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
            raise
        require(child.returncode == 0, f"{name} exited {child.returncode}; see {name}.log")
    print(f"COMPLETE {name}", flush=True)


def calibration(args):
    """Full generation dense arithmetic, actual solver times and operator audit."""
    import copy
    import torch
    from contextlib import nullcontext
    from torch.utils.flop_counter import FlopCounterMode
    from torch.utils._python_dispatch import TorchDispatchMode
    from run_paired import load_policy_and_processors, seed_environment, make_noise
    from lerobot.policies.common.flow_matching import euler_integrate

    args.output.mkdir(exist_ok=False, parents=True)
    require(gpu_processes() in ([], [os.getpid()]), "GPU occupied before calibration")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_environment(0)
    policy, _, _, native = load_policy_and_processors(args.checkpoint)
    # Audited real development observation from task 0/state 0; task outcomes unused.
    artifact = args.root / "results/development_gate_concurrent_002/inputs.pt"
    inputs = torch.load(artifact, map_location="cuda", weights_only=False)
    batch = inputs["prepared_observations"]["native"]
    identity = dict(suite="libero_spatial", task_id=0, initial_state_id=0, policy_seed=1000)
    noise = make_noise(policy, identity, 0)
    totals = {}
    operators = {}
    solver = {}
    timings = collections.defaultdict(list)
    class Inventory(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = collections.Counter()
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops[str(func)] += 1
            return func(*args, **(kwargs or {}))
    amp = lambda: torch.autocast(device_type="cuda") if policy.config.use_amp else nullcontext()
    with torch.inference_mode(), amp():
        for d in (2, 3, 5, 8, 10, 20):
            ts = []
            def velocity(x, t):
                ts.append(float(t[0]))
                return torch.ones_like(x)
            test = euler_integrate(velocity, torch.ones((1, 2), device="cuda"), d)
            require(torch.allclose(test, torch.zeros_like(test), atol=1e-6), "solver interval truncated")
            require(len(ts) == d and np.allclose(ts, 1 - np.arange(d) / d), "solver schedule mismatch")
            policy.config.num_steps = d
            actual_ts = []
            original = policy.model.denoise_step
            def denoise(*a, **kw):
                actual_ts.append(float(kw["timestep"][0]))
                return original(*a, **kw)
            policy.model.denoise_step = denoise
            try:
                inv = Inventory()
                with inv, FlopCounterMode(display=False) as counter:
                    action = policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone())
                torch.cuda.synchronize()
            finally:
                policy.model.denoise_step = original
            require(len(actual_ts) == d and np.allclose(actual_ts, ts), "model solver timesteps mismatch")
            require(action.shape[1] == 50 and torch.isfinite(action).all().item(), "invalid full chunk")
            totals[str(d)] = int(counter.get_total_flops())
            require(totals[str(d)] > 0, "no counted neural arithmetic")
            operators[str(d)] = dict(inventory=dict(inv.ops),
                                     flop_counts={str(k): int(v) for k, v in counter.get_flop_counts()["Global"].items()})
            solver[str(d)] = dict(actual_timesteps=actual_ts, constant_field_end=test.cpu().tolist())
            for _ in range(20):
                policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone())
        # Randomized timing blocks, warm-up kept above and profiling kept separate.
        for bi in range(3):
            for d in sorted((2, 3, 5, 8, 10, 20), key=lambda d: sha([bi, d, "timing"])):
                policy.config.num_steps = d
                for i in range(20):
                    current = copy.deepcopy(batch)
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before = time.perf_counter()
                    start.record()
                    policy.predict_action_chunk(current, noise=noise)
                    end.record()
                    end.synchronize()
                    timings[str(d)].append(dict(block=bi, call=i, host_s=time.perf_counter()-before,
                                                cuda_ms=float(start.elapsed_time(end))))
    base_d = np.array([2, 3, 5, 8, 10, 20])
    coeff = np.polyfit(base_d, [totals[str(d)] for d in base_d], 1)
    residual = max(abs(totals[str(d)] - np.polyval(coeff, d))/totals[str(d)] for d in base_d)
    write(args.output / "calibration.json", dict(status="completed", generation_flops=totals,
          affine=dict(base=float(coeff[1]), step=float(coeff[0]), max_relative_residual=float(residual)),
          use="Measured per-d values; no imposed affine fit", solver=solver, operators=operators,
          timings=timings, native=native, inputs_sha256=file_sha(artifact),
          input_shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},
          torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
          boundary="Complete predict_action_chunk neural dense arithmetic: all supported matrix products, convolutions and attention across vision, language prefill and action expert; multiply-add=2 FLOPs. Embedding lookups, normalization, activations, elementwise arithmetic, indexing and external preprocessing excluded by the declared conventional dense-arithmetic boundary. Full operator inventory retained for audit; this is not an exact count of every scalar instruction.",
          timing_boundary="Host prediction plus end-event synchronization; CUDA current-stream prediction interval. Profiling and warmup excluded. Display processes remain on shared display GPU; no unrelated CUDA compute process permitted."))


def features(hd, costs, kind):
    h, d = np.asarray(hd, float).T
    rate = np.array([costs[str(int(di))]/hi for hi,di in zip(h,d)]) / costs["10"]
    x, y = np.log(h), np.log(d/10)
    if kind == "compute":
        z = np.log(rate)
        return np.array([np.ones_like(z), z, z*z]).T
    return np.array([np.ones_like(x), x, y, x*x, y*y, x*y]).T


def fit(x, y, ridge):
    beta = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1])*ridge
    penalty[0,0] = 1e-6
    for _ in range(100):
        prob = 1/(1+np.exp(-np.clip(x@beta,-30,30)))
        delta = np.linalg.solve(x.T@((prob*(1-prob))[:,None]*x)+penalty,
                                x.T@(prob-y)+penalty@beta)
        beta -= delta
        if np.max(abs(delta)) < 1e-8:
            break
    return beta


def predict(x, beta):
    return 1/(1+np.exp(-np.clip(x@beta,-30,30)))


def selection(args, costs):
    records = [json.loads(line) for line in (args.output/"discovery/episodes.jsonl").read_text().splitlines()]
    require(len(records) == 480, "selection requires complete discovery")
    groups = np.array([f"{r['episode_identity']['suite']}:{r['episode_identity']['task_id']}" for r in records])
    hd = [(r["h"], r["denoising_steps"]) for r in records]
    y = np.array([r["success"] for r in records], float)
    models = []
    for kind, ridge in itertools.product(("compute", "allocation"), (1., 10.)):
        x = features(hd,costs,kind)
        cv = np.zeros(len(y))
        for task in sorted(set(groups)):
            train = groups != task
            cv[~train] = predict(x[~train], fit(x[train], y[train], ridge))
        loss = float(-np.mean(y*np.log(np.clip(cv,1e-9,1))+(1-y)*np.log(np.clip(1-cv,1e-9,1))))
        models.append(dict(kind=kind, ridge=ridge, grouped_log_loss=loss,
                           coefficients=fit(x,y,ridge).tolist()))
    best = min(models, key=lambda m:(m["grouped_log_loss"],m["kind"],m["ridge"]))
    candidates = GRID+UNTESTED
    predicted = predict(features(candidates,costs,best["kind"]),np.array(best["coefficients"]))
    p = dict(zip(candidates,map(float,predicted)))
    observed = {c:float(np.mean([r["success"] for r in records if (r["h"],r["denoising_steps"])==c])) for c in GRID}
    rate = lambda c:costs[str(c[1])]/c[0]
    selections=[]
    chosen={(1,10)}
    for fraction in (.25,.5,1.):
        ceiling=fraction*costs["10"]
        eligible=[c for c in candidates if rate(c)<=ceiling*(1+1e-12)]
        grid=[c for c in GRID if rate(c)<=ceiling*(1+1e-12)]
        if not eligible or not grid:
            selections.append(dict(fraction=fraction,status="no eligible configuration"))
            continue
        recommendation=min(eligible,key=lambda c:(-p[c],rate(c),c))
        alternative=min(grid,key=lambda c:(-observed[c],rate(c),c))
        # If selection equals its comparator, use strongest distinct eligible grid
        # alternative; no manufactured claim of an untested recommendation.
        if alternative==recommendation and len(grid)>1:
            alternative=min([c for c in grid if c!=recommendation],key=lambda c:(-observed[c],rate(c),c))
        chosen.update((recommendation,alternative))
        selections.append(dict(fraction=fraction,primary=fraction==.5,ceiling=ceiling,
             recommendation=recommendation,alternative=alternative,
             prediction_type="untested_configuration" if recommendation in UNTESTED else "task_transfer",
             predicted_success=p[recommendation],alternative_predicted_success=p[alternative],
             predicted_difference=p[recommendation]-p[alternative],
             predicts_superiority=p[recommendation]>p[alternative],
             status="frozen_before_confirmation"))
    frozen=dict(models=models,selected_model=best,comparisons=selections,
                discovery_sha256=file_sha(args.output/"discovery/episodes.jsonl"),
                candidates=[dict(h=c[0],d=c[1],prediction=p[c],rate=rate(c)) for c in candidates],
                uncertainty="Descriptive paired state bootstrap, 20000 draws; primary budget .5; others secondary. Fixed coverage, no noninferiority claim. Selection uncertainty assessed by held-task CV.")
    write(args.output/"frozen_prediction.json",frozen)
    confirmation=plan(CONF,range(20,30),[3000,3001],sorted(chosen),"confirmation")
    write(args.output/"confirmation_plan.json",confirmation)
    return frozen


def analyze(args, frozen, costs):
    records=[json.loads(l) for l in (args.output/"confirmation/episodes.jsonl").read_text().splitlines()]
    indexed={(r["episode_identity"]["suite"],r["episode_identity"]["task_id"],r["episode_identity"]["initial_state_id"],
              r["episode_identity"]["policy_seed"],r["h"],r["denoising_steps"]):r for r in records}
    rng=np.random.default_rng(20260912)
    comparisons=[]
    for comparison in frozen["comparisons"]:
        if "recommendation" not in comparison: continue
        a,b=tuple(comparison["recommendation"]),tuple(comparison["alternative"])
        differences=[]
        for suite,task in CONF:
            differences.append([np.mean([float(indexed[(suite,task,state,seed,*a)]["success"])-
                                           float(indexed[(suite,task,state,seed,*b)]["success"])
                                          for seed in (3000,3001)]) for state in range(20,30)])
        ds=np.array(differences)
        sampled=np.zeros(20000)
        for task_values in ds:
            sampled+=task_values[rng.integers(0,10,size=(20000,10))].mean(axis=1)/8
        comparisons.append(dict(**comparison,observed_difference=float(ds.mean()),
                                 descriptive_95_percent_interval=np.quantile(sampled,[.025,.975]).tolist(),
                                 observed_sign_agrees=bool((ds.mean()>0)==comparison["predicts_superiority"])))
    resource=[]
    for c in sorted({(r["h"],r["denoising_steps"]) for r in records}):
        rows=[r for r in records if (r["h"],r["denoising_steps"])==c]
        resource.append(dict(h=c[0],d=c[1],attempts=len(rows),success=float(np.mean([r["success"] for r in rows])),
                  nominal_flops_per_control_step=costs[str(c[1])]/c[0],
                  mean_episode_flops=float(np.mean([r["counts"]["generations"]*costs[str(c[1])] for r in rows])),
                  mean_rollout_wall_s=float(np.mean([r["timing"]["rollout_wall_s"] for r in rows]))))
    write(args.output/"confirmation_analysis.json",dict(comparisons=comparisons,resources=resource,
          scope="Eight declared tasks; bootstrap resamples states within fixed tasks, keeping both policy seeds paired. Intervals descriptive; no benchmark-wide or noninferiority claim."))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--calibrate",action="store_true")
    args=parser.parse_args()
    if args.calibrate:
        calibration(args)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).parent
    protocol=dict(created_unix=time.time(),grid=GRID,untested_candidates=UNTESTED,
        development_tasks=DEV,development_states=list(range(10,15)),development_seeds=SEEDS,
        confirmation_tasks=CONF,confirmation_states=list(range(20,30)),confirmation_seeds=[3000,3001],
        budgets=[.25,.5,1.],primary_budget=.5,discovery_attempts=480,
        confirmation_attempts_per_configuration=160,confirmation_max_attempts=1120,
        model_selection="Leave-one-development-task-out log loss; compute-only log-rate quadratic or log-h/log-d quadratic with interaction; ridge 1 or 10. No monotonicity imposed. Refit chosen model on all discovery. Highest predicted success among budget-eligible candidates. Comparator highest observed development success among eligible grid; strongest distinct alternative if same. Ties prefer lower compute then lexicographic order.",
        confirmation="Freeze predictions before any confirmation outcomes; deduplicate settings; state-paired seed averaging. Fixed coverage proposal, not powered noninferiority. Primary .5 budget; others descriptive secondary.",
        diagnostic="After confirmation: h1,d5 with images AND proprioception refreshed every five ticks; same 40 development identities. Compare with fresh h1,d5 and h5,d5 discovery. Deliberate distribution shift; no numerical attribution of entire prefix penalty.",
        historical_overlap="Held out from this selection procedure only; historical nonexposure unverified.",
        compute="Full-generation conventional dense neural arithmetic, operator inventory and explicit exclusions in calibration. Measured values per d, whole chunks counted for episode expense.",
        code_sha256={p.name:file_sha(p) for p in (Path(__file__),scripts/"run_paired.py")},
        scheduling="No wall-clock truncation or success-dependent stopping. Continues past 12h if necessary. Abort on unrelated CUDA compute process; retain all output.")
    write(args.output/"protocol.json",protocol)
    write(args.output/"discovery_plan.json",plan(DEV,range(10,15),SEEDS,GRID,"discovery"))
    try:
        run_child(args,"calibration",[sys.executable,str(Path(__file__)),"--root",str(args.root),
                  "--output",str(args.output/"calibration"),"--checkpoint",str(args.checkpoint),"--calibrate"])
        cal=json.loads((args.output/"calibration/calibration.json").read_text())
        costs=cal["generation_flops"]
        write(args.output/"frozen_budgets.json",dict(native=costs["10"],
              ceilings={str(b):b*costs["10"] for b in (.25,.5,1.)},calibration_sha256=file_sha(args.output/"calibration/calibration.json")))
        def evaluate(name,extra=()):
            run_child(args,name,[sys.executable,str(scripts/"run_paired.py"),"--checkpoint",str(args.checkpoint),
                      "--output",str(args.output/name),"--plan",str(args.output/(name+"_plan.json")),
                      "--warmup","20",*extra])
        evaluate("discovery")
        frozen=selection(args,costs)
        evaluate("confirmation")
        analyze(args,frozen,costs)
        write(args.output/"stale_plan.json",plan(DEV,range(10,15),SEEDS,[(1,5)],"stale"))
        evaluate("stale",["--observation-refresh","5"])
        write(args.output/"completed.json",dict(status="completed",ended_unix=time.time(),
                                               stages=["calibration","discovery","prediction","confirmation","analysis","stale"]))
    except BaseException as exc:
        write(args.output/"failure.json",dict(status="failed_or_interrupted",error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    main()
