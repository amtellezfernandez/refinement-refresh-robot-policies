# Transfer analysis amendment - 2026-09-13, after 09:18 CEST

This amendment was written after transfer collection began. At the last remote
check (09:10 CEST), 581/1120 attempts were recorded and individual success
outcomes were visible in progress logs. It is not originally prespecified or
fully outcome-blind. No aggregate transfer outcomes were used for these additions.
The diagnostics read only discovery, calibration, and the original frozen model
artifact. No remote runner, schedule, model, or primary contrast is modified.

A later operational check at 09:25:29 CEST recorded 601/1120 confirmation
attempts (53.7%), with attempt 602 active. No confirmation summary, analysis,
completion marker, or failure marker existed. This progress count is retained
for provenance and is not a manuscript result; no interim success rate was
computed or reported.

## Preserved primary and original comparisons

The half-native (2,2) minus (2,5) contrast remains primary. Original quarter-
and full-native contrasts and all seven confirmation configurations are retained.
No sample-size, stopping, or non-inferiority-margin change is made.

## Additional descriptive secondary contrasts

- Recipe transfer: (2,2) minus native (1,10). This is a quality-cost comparison,
  not an equal-budget comparison. No quality-preservation/non-inferiority claim.
- Prefix allocation at half-native ceiling: (2,2) minus (5,2).
- Prefix allocation at full-native ceiling: (1,2) minus (2,2).
- Full-budget learned versus compute-only selection: (1,2) minus native (1,10).

Use all planned 160 attempts per configuration, preserving both seeds within
each of 80 task-state blocks. Average seed differences within state and equally
across states/tasks. Use 20,000 bootstrap resamples of states within each fixed
task, retaining both seeds and all configurations; analysis seed 20260913.
Report two-sided descriptive 95% intervals, without multiplicity-adjusted
superiority, non-inferiority, or benchmark-wide claims. Report per-configuration
all-attempt arithmetic and elapsed means; do not equate rate with realized cost.
Do not select or prioritize contrasts based on transfer outcomes.

## Selector references

Derived from original discovery records and stored model coefficients:

| Budget/native | Learned | Empirical grid | Minimum d, shortest feasible h | Compute-only |
| --- | --- | --- | --- | --- |
| .25 | (5,2) | (5,2) | (5,2) | (5,20) |
| .50 | (2,2) | (2,2) | (2,2) | (2,10) |
| 1.00 | (1,2) | (1,5) | (1,2) | (1,10) |

Empirical selection uses the original grid; model and simple-rule selection use
the original full candidate set. Model ties use lower cost then lexicographic
order, as in the original protocol. The compute-only model is ridge 10, selected
by the original discovery held-task loss. These additional choices are recorded
now, not claimed to have been frozen before collection. Its reduced-budget
choices are outside the existing transfer plan and will not be evaluated by
this campaign. No substitute configuration is treated as that selector.

Identical choices are reported as identical; successful transfer cannot establish
learned-selector advantage over the simple rule at any of the three budgets.
The original full-budget (1,2) minus (1,5) contrast tests learned versus empirical
selection, but does not distinguish learned from simple-rule selection.

## Reproducibility and timing scope

`experiments/tooling/selector_diagnostics.py` checks discovery hashes, reproduces
all four original held-task losses, computes constant and fold-fitted intercept
references, derives selector choices, and summarizes all 360 fixed-input timing
calls, preserving the three block means. Its JSON output records input hashes.
The logs contain full-generation host/CUDA intervals, not component-attributed
timings or component-attributed FLOP counts. Do not infer a component-level
causal decomposition. No further rollout campaign is authorized by this amendment.

## Resource-coordinate sensitivity addition

The manuscript additionally applies the existing minimum-denoising,
shortest-feasible-prefix rule to the fixed-input host-generation-time coordinate.
At ceilings .25, .50, and 1.00 relative to the native value of each resource,
the arithmetic choices are (5,2), (2,2), and (1,2), while host-time choices are
(2,2), (1,2), and (1,2). This is a descriptive calculation from calibration,
not a replacement of the frozen arithmetic-budget experiment or a quality-optimal
claim. The profiled tensor shapes are recorded in the manuscript. Because the
rollout artifacts do not establish identical instruction-token length on all
transfer tasks, using one arithmetic value is disclosed as a transport assumption.
