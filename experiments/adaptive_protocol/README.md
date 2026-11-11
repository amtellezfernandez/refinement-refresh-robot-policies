# Extensible evaluation framework for closed-loop compute allocation

This framework studies how closed-loop robot policies allocate inference
computation between generating actions and obtaining new
observation-conditioned predictions. Each study maps the controls exposed by
its policy architecture onto this question and reports the applicable
evaluation and cost boundaries. Earlier software stacks, checkpoints, and
simulators provide reference cases rather than mandatory templates.

A study reports the following fields when they apply:

1. the internal generation or refinement control;
2. the predicted action-chunk length `K`;
3. the executed prefix `h` before observing and replanning;
4. cost per control step and per attempt under an explicit counting boundary;
5. the success rule and evaluation population; and
6. whether each comparison is discovery, frozen confirmation, or amended
   descriptive analysis.

## Evidence stages

- **Runtime baseline:** first test model loading, then run representative
  inference and measure latency and memory. A load-only result establishes only
  model-load feasibility; inference and closed-loop feasibility remain
  untested.
- **Adapter parity:** feed identical frozen observations through the native and
  adapted paths, then compare their action outputs.
- **Closed-loop preflight:** use a small set of identities excluded from
  confirmation to validate reset, stepping, termination, and outcome capture.
- **Confirmation:** for confirmatory studies, freeze the primary within-model
  comparison and its attempt count before evaluation.
- **Descriptive extensions:** report later contrasts separately from the
  prespecified primary analysis.

## Run records

Auditable executions should emit a compact run record conforming to
`run_record.schema.json`. The record describes what happened in that execution;
it is not a permanent recipe for future studies. Fields that do not apply to an
evidence stage remain explicitly null rather than implying an operation that
did not occur.

The first local study uses SmolVLA as a worked case. Its initial order is:

1. checkpoint download and load test;
2. batch-one inference and resource measurement;
3. native-adapter parity;
4. closed-loop preflight; and
5. a frozen paired confirmation comparison.
