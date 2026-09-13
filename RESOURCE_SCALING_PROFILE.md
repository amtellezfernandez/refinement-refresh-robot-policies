# Post-Training Resource Scaling Profile v0.2

Use one card per **named comparison**. A measured zero differs from an unknown
applicable cost and from a quantity excluded by the declared boundary. Never
assign zero to inherited but unmeasured pretraining.

## Evaluation protocol

- Comparison ID, baseline, intervention, reason for comparison:
- Scope: adaptation / deployment evaluation / lifecycle:
- Checkpoint revision and file/config/normalization hashes:
- Environment/code revisions, dependency lock, hardware and timing clocks:
- Task/state population, episode IDs, split and holdout verification:
- Outcome estimand, task weights, denominator, uncertainty method:
- Comparable candidate configurations, resource limits with units, expected use volume:
- Required quality threshold and precision needed to assess it:
- Principal contrast, selection rule, multiplicity handling, confirmation identities:
- Responsiveness metric and threshold when applicable; explicit exclusion otherwise:
- Decision status: supported choice / unresolved alternatives / insufficient evidence;
  evidence and uncertainty supporting that status:
- Fixed coordinates, swept coordinates, inherited costs and exclusions:
- Run artifacts, attempt/retry IDs, manifest digest, independently logged counters:

## Resource records

Each row requires `name`, `kind`, `amount`, `unit`, `status`, `boundary` and
`aggregation`. `measured`/`derived` amounts are finite and nonnegative;
`unknown`/`not_applicable` amounts are null. Derived rows require a derivation;
state why a quantity is inapplicable. Record source/version/locator when extracting
published evidence. Numerical model examples must be labeled synthetic.

| Quantity | Unit examples | Aggregation / required distinction |
|---|---|---|
| New real interaction | transitions, robot-seconds | Sum; separate inherited demonstrations |
| Human work | interventions; person-hours | Separate rows; counts do not measure labor |
| Simulated / imagined work | transitions | Separate simulator/model rows and phases |
| Optimization | steps; GPU-seconds | Separate update count, measured device time and hardware |
| Deployment work | selections; chunks; internal evaluations | Separate rows; cached actions are not new generations |
| Host wall time | seconds | Outer interval; loading, warm-up and rollout boundaries separate |
| Device elapsed time | milliseconds | Declare stream/profiler scope; no overlap double counting |
| Memory / storage | bytes | Peak maximum versus retained snapshot at a named time |

State the required rows before collection. A missing required row or an unknown
applicable row makes the card incomplete. A syntactically valid card can still
omit a relevant requirement; independent review is needed.

## Context needed for allocation studies

Record the following when they affect the named comparison:

- **Experience and representation:** underlying observation-action pairs or
  transitions, represented transformer inputs/tokens, repeated training exposures,
  tokenizer revision, trainable versus frozen parameters, and training objective.
  These are distinct quantities, not interchangeable meanings of “data size.”
- **Candidate production and selection:** policy generations, proposals drawn
  from those predictions, action-verifier calls, verifier training/adaptation,
  and elapsed deployment cost. An action verifier scores candidate actions;
  an execution trace verifier checks the consistency of measurement records.
- **Imagined experience:** world-model execution can consume resources even
  with frozen weights. Record generation work separately from model updates,
  and distinguish new work from the inherited model's unknown training cost.

These are reporting requirements for the selected claim. The current SmolVLA
execution trace does not implement a world-model or action-verifier pipeline. Its checks
do not certify that an unknown cost fits a budget or estimate a scaling law.
The [allocation literature notes](audit/ALLOCATION_LITERATURE.md) document the
source examples motivating these fields.

## Outcomes and validity

- Attempted/completed/successful episodes, task-level outcomes and terminal causes:
- Steps and time to termination; per-attempt cost and all-attempt cost per success:
- Paired initial-state hashes and keyed randomness; same-condition replay:
- Independent terminal outcome and dispatch records:
- Model-entry versus internal-solver counters, warm-up and instrumentation overhead:
- Adverse outcomes/recovery: definition and validation, or explicitly unmeasured:
- Structure valid / declared fields complete / execution reconciled (separate verdicts):
- Missing measurements, contamination, retries and exclusions:

## Completed illustration: archived v2 h=1 versus h=5

This card summarizes recorded deployment evidence, not an adaptation improvement.

| Field | Evidence and limit |
|---|---|
| Baseline / intervention | SmolVLA/LIBERO, d=10; execute h=1 versus h=5 |
| Evaluation | Spatial task 0, state 0, seed 1000; one episode per cell |
| Selected actions / generations | 92/92 versus 66/14, reconciled from records |
| Warm-up | 20 extra generations per job, separate from rollout generations |
| Task success | 1/1 each in evaluator summaries; independent terminal stream absent |
| Pairing | Same initial-state hash and seed records; no quality inference |
| Timing | Per-generation CUDA-event stream elapsed recorded; outer eval_s includes warm-up and pipeline work |
| Checkpoint provenance | Revision declared in project manifest, not embedded/hash-verified in each archived run |
| Training / real acquisition | No new training or real interaction in the declared deployment intervention; inherited checkpoint costs unknown |
| Safety / reactivity | Unmeasured; queued actions do not revise from intermediate visual observations |
| Defensible conclusion | Schedule/count reconciliation on this record; no preserved-quality or speedup claim |

Source: [generated archive audit](experiments/real_smolvla/ARCHIVE_AUDIT.md).
The generic trace validator does not automatically generate this card from the
deployment execution trace; this is a documented summary of the audited records.
