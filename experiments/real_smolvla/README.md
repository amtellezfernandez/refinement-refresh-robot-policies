# Real SmolVLA/LIBERO deployment accounting evidence

Start with the [recomputed archive audit](ARCHIVE_AUDIT.md) and its
[machine-readable results](archive_audit.json). Run `python audit_archive.py --check`
to validate the raw inputs and check that the derived audit remains current.
Offline validation needs NumPy; it does not load model weights or require a GPU.
The four v2 artifacts support one-task instrumentation and replay checks. The
22 v1 artifacts below support descriptive accounting only.

Infrastructure update: see [Ledger v3](LEDGER_V3.md) for completion/configuration
checks, and [Ledger v2](LEDGER_V2.md) for the earlier smoke test. Historical results below
use aggregate-only v1 instrumentation. Their matching outcomes/counters do not
verify episode pairing or deterministic action replay; v2 explicitly rejects
those artifacts for these checks. The v1 timing is host elapsed time with a
post-call CUDA synchronization, not isolated CUDA-event latency. Original
artifacts are retained unchanged for auditability.

This directory contains a small, reproducible deployment pilot run on the
public `HuggingFaceVLA/smolvla_libero` checkpoint (revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`) using LeRobot and LIBERO
`libero_spatial`, task id 0. The evaluator was instrumented to distinguish
control-loop calls from actual neural action-chunk generations.

The three conditions use the same checkpoint, task, seed (`1000`), two
episodes, and ten flow-matching denoising steps. Only `n_action_steps` varies.
The raw `eval_info.json` files are retained in `results/`; videos and model
weights are intentionally not checked into this repository.

## Result

| action chunk | control steps | neural chunks | control/neural | success | eval seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 355 | 355 | 1.00x | 1/2 (50%) | 162.13 |
| 5 | 357 | 72 | 4.96x | 1/2 (50%) | 43.24 |
| 10 | 356 | 36 | 9.89x | 1/2 (50%) | 28.75 |

The ordered outcome lists match across conditions (`[success, failure]`).
Initial-state pairing is unverified in these v1 artifacts, and two episodes do
not establish quality preservation. The table illustrates why control calls
and neural generations require separate counters: the observed call/chunk
ratio changes from 1.00 to 9.89. It does not establish a fixed control cadence
or a quality-matched speedup. Inference timing uses host elapsed time with a
post-call CUDA synchronization, includes cold calls, and is not CUDA-event
latency. Process startup and model loading are outside rollout `eval_s`.

## LIBERO suite gate

One task-0 episode was also run at `h=10,d=10` for each supported suite. The
checkout names the long-horizon suite `libero_10` (not `libero_long`; the latter
was rejected and that log is retained):

| suite | task-0 success | control steps | neural chunks |
|---|---:|---:|---:|
| `libero_spatial` | 1/1 | 76 | 8 |
| `libero_object` | 0/1 | 280 | 28 |
| `libero_goal` | 0/1 | 300 | 30 |
| `libero_10` | 0/1 | 520 | 52 |

These are compatibility/gate measurements, not a benchmark score. The failed
object, goal, and long-horizon episodes are retained rather than filtered.

As a cross-task sanity check, we also evaluated one episode each on spatial
tasks 0 and 1. Chunk 5 used 176 control steps and 36 neural chunks (4.89x),
while chunk 10 used 183 control steps and 19 neural chunks (9.63x); both
conditions solved both episodes (2/2). These two extra episodes are archived as
`eval_spatial01_h5.json` and `eval_spatial01_h10.json` and are not pooled with
the primary historical table (pairing is unverified).

Finally, a small solver-effort cross-check on the same two tasks varied both
chunk size and flow-matching denoising steps:

| chunk | denoising steps | control steps | neural chunks | success | rollout s |
|---:|---:|---:|---:|---:|---:|
| 5 | 5  | 177 | 36 | 2/2 | 19.30 |
| 10 | 5  | 176 | 19 | 2/2 | 15.50 |
| 5 | 20 | 355 | 71 | 1/2 | 71.66 |
| 10 | 20 | 185 | 19 | 2/2 | 26.11 |

These cells are exploratory and intentionally not pooled into a statistical
claim: the sample is two tasks with one episode each. The d=20,h=5 cell is one
observed case where the higher denoising setting did not yield two successes;
it does not establish a general relationship between solver effort and quality.

The primary h=10 condition was then rerun from the same requested seed. It
reproduced 356 control steps, 36 neural chunks, and the same ordered outcomes
`[true, false]`; the repeat ledger is `eval_h10_repeat.json`. Those aggregates
do not verify deterministic replay. Only the separately archived v2 h=10
comparison verifies step-level action replay.

Re-run details and the server environment are documented in
`../../EXPERIMENT_DESIGN.md` and the repository-level experiment audit.
The exact evaluator modification is archived as
`lerobot_eval_instrumentation.patch` and applies to the pinned LeRobot commit.

## Five-task-per-suite sweep

After the pilot, five task IDs (0--4) from each supported suite were evaluated
once at both refresh schedules, with the same checkpoint, seed, and `d=10`.
This is a broad exploratory sweep, not the proposed confirmatory sample:

| suite | h=5 success | h=10 success | h=5 neural chunks | h=10 neural chunks |
|---|---:|---:|---:|---:|
| spatial | 5/5 (100%) | 4/5 (80%) | 99 | 70 |
| object | 3/5 (60%) | 3/5 (60%) | 196 | 94 |
| goal | 4/5 (80%) | 4/5 (80%) | 166 | 72 |
| `libero_10` | 2/5 (40%) | 1/5 (20%) | 416 | 235 |

The raw ledgers are under `results/broad_sweep/`. Outcomes are not pooled into
a causal efficiency claim: each cell has one episode per task, and changing
the refresh schedule changes the trajectory and therefore the terminal task
distribution. The differing observed outcomes motivate a controlled follow-up;
they do not estimate the causal quality effect of changing the refresh schedule.
The repository contains a launcher for a broad h=1 reference, but no completed
broad h=1 artifacts. Do not describe that reference as an executed experiment.

## Local verification

```bash
python -m unittest -v test_ledger test_result_metrics
python audit_archive.py --check
python summarize_results.py
python summarize_broad.py
```

The RNG-isolation unit test requires Torch and is explicitly skipped when Torch
is absent. Other offline tests and the archive audit require NumPy only. These
checks validate archived evidence and analysis code; they do not execute LIBERO
or certify an unperformed experiment. See `../../EXPERIMENT_DESIGN.md` for the
prospective workshop protocol.
