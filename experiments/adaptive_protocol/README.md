# Adaptive RoboPAD evaluation

RoboPAD is an evolving evaluation protocol for allocating inference compute in
closed-loop robot policies. A study instantiates the protocol using controls
native to its policy architecture; it does not need to reproduce the software,
checkpoint, or simulator used by an earlier study.

Each study must define:

1. the internal generation or refinement control;
2. the predicted action-chunk length `K`;
3. the executed prefix `h` before observing and replanning;
4. complete cost per control step and per attempt;
5. the success rule and evaluation population; and
6. whether each comparison is discovery, frozen confirmation, or amended
   descriptive analysis.

## Evidence stages

- **Runtime baseline:** load the policy, run representative inference, and
  measure latency and memory. This establishes feasibility, not task quality.
- **Adapter parity:** compare the study adapter with the policy's native path on
  frozen observations and actions.
- **Closed-loop preflight:** use a small set of identities excluded from
  confirmation to validate reset, stepping, termination, and outcome capture.
- **Confirmation:** freeze one primary within-model comparison and its attempt
  count before execution.
- **Descriptive extensions:** label later contrasts explicitly and keep them out
  of the frozen primary claim.

## Run records

Every execution writes a compact run record conforming to
`run_record.schema.json`. The record describes what happened in that execution;
it is not a permanent recipe for future RoboPAD studies.

The first local study uses SmolVLA as a worked case. Its initial order is:

1. checkpoint download and load test;
2. batch-one inference and resource measurement;
3. native-adapter parity;
4. closed-loop preflight; and
5. a frozen paired confirmation comparison.

