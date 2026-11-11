# Studying compute allocation in closed-loop robot policies

The scientific question is how a closed-loop robot policy should divide
inference computation between generating actions and obtaining a new
observation-conditioned prediction. Different policy architectures expose
different controls for this choice, so each study must define what varies, what
is held fixed, how cost is counted, and which conclusions its evidence supports.
Earlier checkpoints and simulators are worked cases rather than templates that
later studies must reproduce.

To interpret a comparison, report the following quantities when they apply:

1. the internal generation or refinement control;
2. the predicted action-chunk length `K`;
3. the executed prefix `h` before observing and replanning;
4. cost per control step and per attempt under an explicit counting boundary;
5. the success rule and evaluation population; and
6. whether each comparison is discovery, frozen confirmation, or amended
   descriptive analysis.

## What each stage can establish

- **Feasibility:** establish separately whether the model can be loaded, whether
  it can generate actions, and whether it can run in closed loop. Loading alone
  does not establish the latter two.
- **Agreement with the reference implementation:** give the native and modified
  evaluation paths identical observations and compare their predicted actions.
- **Evaluation check:** use a small set of cases excluded from the main comparison
  to verify initialization, action execution, termination, and outcome capture.
- **Held-out comparison:** specify the primary within-model comparison and sample
  size before observing its outcomes.
- **Additional analyses:** distinguish later comparisons from the prespecified
  analysis.

## Evidence retained for each experiment

Each experiment retains the model and data versions, comparison being tested,
outcome rule, measurement boundary, observed resources, and evidence status.
These details make the result auditable without prescribing how later studies
must be conducted.

The first local study uses SmolVLA as a worked case. It moves from feasibility
and agreement checks to an excluded-case evaluation check, followed by a
prespecified paired comparison.
