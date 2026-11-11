# Studying compute allocation in closed-loop robot policies

The scientific question is how a closed-loop robot policy should divide
inference computation between generating actions and obtaining a new
observation-conditioned prediction. Different policy architectures expose
different controls for this choice, so each study must define what varies, what
is held fixed, how cost is counted, and which conclusions its evidence supports.
Earlier checkpoints and simulators are worked cases rather than templates that
later studies must reproduce.

A study reports the following fields when they apply:

1. the internal generation or refinement control;
2. the predicted action-chunk length `K`;
3. the executed prefix `h` before observing and replanning;
4. cost per control step and per attempt under an explicit counting boundary;
5. the success rule and evaluation population; and
6. whether each comparison is discovery, frozen confirmation, or amended
   descriptive analysis.

## What each stage can establish

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

## Evidence retained for each experiment

Each experiment retains the model and data versions, comparison being tested,
outcome rule, measurement boundary, observed resources, and evidence status.
These details make the result auditable without prescribing how later studies
must be conducted.

The first local study uses SmolVLA as a worked case. Its initial order is:

1. checkpoint download and load test;
2. batch-one inference and resource measurement;
3. native-adapter parity;
4. closed-loop preflight; and
5. a frozen paired confirmation comparison.
