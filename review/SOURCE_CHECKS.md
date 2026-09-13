# Model-assisted source verification

Status: in progress, not independent human annotation. These checks do not
retroactively validate all 128 cells or replace the historical CSV silently.
Priority: evidence correctness, comparison boundaries, sensitivity, then breadth.

## VLA-RL — checked 2026-09-10

Source/version: https://arxiv.org/html/2505.18719v1, section 4.1,
Experimental Setup, Performance Comparisons and Table 1.

Boundary: RL adaptation of the released suite-specific OpenVLA SFT checkpoints
versus OpenVLA SFT in the four LIBERO suites; inherited SFT is not newly acquired
interaction. Other Table 1 models are not assumed resource-matched.

- Evaluation denominator: section 4.1 specifies 500 episodes **per suite**.
  The old note's unqualified “500 evaluation episodes” loses that denominator.
- Training quantity: the text reports 48 GPU-hours, but the inspected passage
  does not resolve per-suite versus total allocation or the hardware identity.
  Preserve the stated quantity; do not call this a fully reconstructed per-recipe
  training budget. The historical Q needs boundary review, not automatic approval.
- This check does not establish absence of other costs in all code/supplements.

## SAM2Act+ — checked 2026-09-10

Source/version: https://arxiv.org/html/2501.18564v1, sections 5.1, 8, 8.1,
and 8.2; Table 5.

Proposed boundary for further checking: SAM2Act+ memory-dependent evaluation
on MemoryBench, not every experiment under the SAM2Act paper title.

- Section 5.1 gives 100 training demonstrations per task and 25 held-out
  demonstrations; quantities are demonstrations, not known-length transitions.
- Section 8 gives 32 H100/A100 GPUs, with 16/8-device alternatives at equal
  total batch size. Table 5 gives 56.25K steps for **SAM2Act**, while section 8.2
  separately addresses SAM2Act+. Do not transfer the base-model training budget
  to the memory variant without its own evidence.
- The historical row combines real-robot, simulation and memory claims. Revisit
  applicability under the chosen comparison before changing labels or counts.

## Remaining work

Other 14 systems and the remaining cells are not verified by this note. Retain
Q/P/U/NA as preliminary coding and the existing 0/16, 1/16, 2/16, 3/16 sensitivity
as calculations on unchanged labels—not updated evidence-backed results.
For each subsequent cell record version, exact locator, comparison, value/unit/
denominator, interpretation and unresolved ambiguity. Missing source access must
remain unresolved, not be converted into U or NA.
