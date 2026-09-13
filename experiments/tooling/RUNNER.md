# Paired checkpoint evaluation

`run_paired.py` uses one strictly loaded SmolVLA checkpoint and one active LIBERO
environment. It requires the pinned LeRobot source and its Linux/CUDA runtime.
It does not modify upstream source or old experiment ledgers.

```bash
python run_paired.py \
  --checkpoint /opt/robopad/checkpoints/smolvla_libero \
  --output /opt/robopad/results/development_unique_name \
  --development
```

The development gate evaluates Spatial task 0, state 0, policy seed 1000 across
all four h/d configurations, native (1,10) first. This is a new runner and noise
contract; it is not expected to reproduce the old aggregate pilot numbers.

After inspecting that run and measuring allocation, the full planned schedule is:

```bash
python run_paired.py \
  --checkpoint /opt/robopad/checkpoints/smolvla_libero \
  --output /opt/robopad/results/paired_unique_name \
  --plan ../planned_evaluation.json
```

When a full 320-attempt collection is not feasible, the checked-in focused pilot
uses the same execution contract on one suite:

```bash
python run_paired.py \
  --checkpoint /opt/robopad/checkpoints/smolvla_libero \
  --output /opt/robopad/results/focused_pilot_unique_name \
  --plan ../pilot_evaluation.json
```

This plan contains 80 attempts: two fixed Spatial tasks, five initial states,
two policy seeds, and four conditions. It is a paired pilot with narrower task
coverage than the eight-task plan. Neither allocation is a certified independent
holdout or a powered test of preserved quality.

An upfront `--max-blocks 16` selects 64 attempts: eight tasks, one shared initial
state per task, two policy seeds, four conditions. Caps must be multiples of 16;
32/48/64/80 blocks use two/three/four/five states per task. Selection follows the
frozen manifest order and preserves both policy seeds for each selected state.
No result-dependent stopping or cap changes occur inside the runner. Availability
is verified at reset, but absence of overlap with unarchived studies is not
certified by this software.

Outputs are exclusively created; use a new output directory for every execution.
Each invocation receives a unique attempt namespace. `selected_schedule.json`
exists before inference. `runtime_manifest.json` records checkpoint file hashes,
effective config, package versions, GPU/driver information, source hashes, and
measurement boundaries. `steps.jsonl` and `episodes.jsonl` flush incrementally.
`summary.json` appears with `status=completed` only after the selected schedule
finishes. Failures after model initialization produce `failure.json`; no final
summary is synthesized from interrupted journals.

Actual denoising, image encoder, prefill, action expert, and generation calls are
counted at their model methods. CUDA events measure the chunk-generation stream
interval, not hardware FLOPs or energy. Twenty warm-up generations occur once
per condition, outside measured rollouts. Rollout wall time includes observation
processing, noise, inference, action handling, simulation/rendering and step
journal writes; reset, warm-up, model loading and environment creation are separate.
Peak allocated/reserved CUDA memory is reset after warm-up for each episode.

The driver bounds-checks and assigns initial-state IDs before hard reset. It
records reset state and observation hashes, then checks matching starts across
conditions before further policy execution. Noise is passed explicitly through
a local CUDA generator keyed by episode identity and simulation tick. Changing h
cannot consume another episode's random stream. Independent base-environment
records are reconciled with produced actions and vector-environment rewards,
termination and success. This checks internal consistency; a separate upstream
parity comparison remains necessary to establish behavioral equivalence to the
unmodified upstream evaluator.

Offline tests (standard library plus NumPy):

```bash
python -m unittest -v test_run_paired
```

These tests cover manifests, balance, keyed identities, array integrity, vector
success interfaces, hook counts and restoration. They do not substitute for
running the actual checkpoint and simulator.
