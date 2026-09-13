# Ledger v2: infrastructure validation, not a performance result

The historical ledgers outside `results/infrastructure_v2/` are v1. They contain neither initial-state nor
action hashes and cannot be upgraded retrospectively. The v2 validator rejects
them. Do not use a successful v2 test to certify historical pairing claims.

## Run preservation

Both broad launchers allocate a UTC-timestamped `mktemp` run root. No output
directory is removed. Every condition has its own log and output directory;
`ledger_events.jsonl` is exclusively created and line-flushed during execution.
The final `eval_info.json` is the completed-run artifact; a journal alone may be
an interrupted run and is not eligible for performance summaries.

## Pairing contract

Only seeded, synchronous, batch-one LIBERO runs with predefined initial states
are supported. Unsupported configurations fail rather than silently weakening
the contract. Episode identity is `(suite, task, initial-state index, seed)`.
Python, NumPy, Torch (including CUDA), and the environment are seeded at every
reset. The full simulator state after reset/settling is stored with its dtype,
shape, values, and SHA-256. Policy RNG is isolated from environment RNG and keyed
by episode identity and simulation tick, independent of refresh rate. Strict
Torch deterministic algorithms are enabled; launchers set the required cuBLAS
workspace configuration before Python starts.

`--paired` requires identical episode identities, seed records, and initial-state
hashes. It does **not** require equal success or equal actions between different
refresh settings. `--replay` additionally requires the same h/solver setting and
exact step-level action records. Exact replay is a check on these runs and this
software/hardware stack, not a guarantee across machines or library versions.

## Timing and independent accounting

Twenty initial warm-up predictions are explicitly labeled and counted separately.
They do not advance the simulator, and policy queues/RNG are restored before the
first measured action. Each chunk prediction uses CUDA start/end events, with
pending CUDA work synchronized before the start. `cuda_ms` is stream elapsed
time around `_get_action_chunk`, not kernel-only occupancy, energy, or end-to-end
control latency. Environment stepping and preprocessing outside this function
are excluded. Do not add nested timing measurements together.

The evaluator's total `eval_s` still includes warm-up, rendering, environment
stepping and bookkeeping. Separate measured and warm-up CUDA-event totals are
recomputed in `archive_audit.json`; do not use total rollout time as warm-up-free
model latency or pool it with legacy v1 host timing.

The model's `sample_actions` entry is counted independently of the policy chunk
wrapper. The evaluator records the postprocessed action before dispatch; an
environment-step wrapper independently records the actual dispatched array.
The validator recomputes hashes from values, reconciles both streams with policy
selection counts, checks consecutive step IDs, and verifies the chunk schedule
against h, separately from warm-up. These are software boundary checks, not
hardware-profiler counters. Reset settling steps are outside the policy ledger.

## Commands

Completed infrastructure smoke check (2026-09-10): Spatial task 0, initial state
0, seed 1000, ten denoising steps. All three h settings and a repeated h=10
started with the same state hash. The h=10 repeat matched all 86 step-level
action arrays exactly. Measured selection/chunk counts were 92/92 (h=1),
66/14 (h=5), and 86/9 (h=10 and repeat); each job additionally recorded 20
warm-up predictions. All six unit-test groups passed, including fault injections.
Raw final ledgers are archived under `results/infrastructure_v2/`; journals and
videos remain under `/mnt/NVME1/projectx_robopad/logs/ledger_v2_check_CfOn4S60/`.
This is an instrumentation/replay smoke test, not a quality-preservation study.

Use the isolated LeRobot environment (Torch and NumPy required):

```bash
python -m unittest -v test_ledger test_result_metrics
python audit_archive.py --check
python validate_ledger.py RUN/eval_info.json
python validate_ledger.py H1/eval_info.json --paired H10/eval_info.json
python validate_ledger.py H10/eval_info.json --replay H10_REPEAT/eval_info.json
```

`lerobot_eval_instrumentation.patch` applies to the pinned **unmodified** LeRobot
checkout; v1 must not be stacked underneath it. `robopad_ledger.py` must be on
`PYTHONPATH` together with its `ledger_arrays.py` helper. Existing runs must
finish before replacing evaluator source.

The command-line validator and summary code require a completed evaluator
artifact and reconcile task/group/overall success records. Standalone v2
objects remain accepted only by the lower-level `validate` function for unit
tests. The runtime does not yet store independent terminal-state/outcome
records or a full checkpoint/hardware/software manifest; task-level outcome
reconciliation does not supply either missing provenance record.
