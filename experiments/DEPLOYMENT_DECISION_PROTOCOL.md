# Configuration decision protocol

2026-09-12; follows `c89a898`. **Design only.** No new learned-policy outcomes,
matched-cost pairs, quality certificates, or validated selector are reported.
The existing runner and analyzer collect and describe the four-cell comparison;
they do not implement the confirmation decision procedure below.

## Question and stages

At comparable measured inference expenditure, does refining a prediction or
refreshing it from a new observation produce better closed-loop outcomes?
The checkpoint always predicts 50 actions. `h` is the number executed before
refresh; `d` is denoising work per generation. Changing either leaves weights fixed.

| Stage | Work | Evidence it can supply |
|---|---|---|
| Runtime and development gates | Strict loading, source/runtime lock, reset/noise checks, upstream parity, connectivity, profiler reconciliation, probe overhead, timed development rollout | Whether the evaluator is usable and a balanced evaluation fits |
| Initial four-cell study | `(1,5), (1,10), (5,5), (5,10)` on the frozen paired manifest | Local measured cost–quality differences, including agreement or disagreement with call-count summaries |
| Matched-cost extension | Development-only timing calibration; freeze new candidates and matching tolerance before evaluation | Refinement-versus-refresh comparison at a specified measured cost, if the match holds |
| Decision confirmation | Freeze selection rule and evaluate selected alternatives on identities/conditions not used to choose them | Whether a configuration choice meets the declared quality and resource requirements; a separate condition can test transfer |

The first grid is not automatically iso-compute and does not establish the last
two results. A transfer claim must name the changed task, condition, or hardware
and its held-out identities; it is not established by another seed on a selected
development observation. Historical overlap remains unverified for the existing
fixed task set.

## Concrete first comparison

Use native `(h=1,d=10)` as the reference. The proposed principal resource-reduction
comparison is `(5,5)` versus that reference, chosen before new outcomes because it
reduces both nominal work controls. This is a joint change, not an isolated test
of refinement versus refresh and not a prediction of preserved quality.
Within-`h` and within-`d` contrasts describe the two factors; report every cell.

The broader plan has 8 tasks × 5 states × 2 seeds × 4 configurations = 320 attempts,
or 80 per configuration. The focused plan has 20 per configuration. Runtime
measurement determines which complete allocation is feasible before reserved
outcomes are collected. Neither guarantees a narrow quality interval.

The initial study reports separate resource coordinates: outer rollout wall time
per attempt, summed host prediction intervals, CUDA-event stream time, work
counts, and peak memory. The proposed principal elapsed-cost coordinate is
task-weighted mean outer rollout wall seconds per attempt, including failed
attempts within the runner's recorded rollout boundary. Initialization and warm-up
are separately reported. This is simulator throughput expenditure; it is neither
allocated GPU time nor physical response latency. Any later inference-budget
claim must name its own clock and denominator.

## Requirements to freeze before a quality-constrained recommendation

The following fields are **unresolved**, not zero or default values:

- Application quality requirement: an absolute success threshold, or a justified
  noninferiority margin `delta` relative to the native reference (or both).
- Resource cap, unit, clock boundary, aggregation, and denominator. Fixed volume
  `N` applies only to compatible additive resources and a stable workload.
- One-sided confidence level and a paired inference method justified for the
  fixed-task/state/seed design; prospective precision assessed using plausible
  discordant-pair frequencies, including sensitivity to small state counts.
- Responsiveness requirement, when applicable: metric, threshold, observation
  timestamps and validation. If outside scope, explicitly limit the claim to
  synchronous simulator outcomes and expenditure.
- Candidate set, principal contrast, selection rule, multiplicity handling,
  confirmation identities, and minimum decision-relevant cost difference.

For candidate `c` and reference `0`, the relative-quality criterion is
`LCB(p_c - p_0) >= -delta`. An absolute threshold also requires appropriate evidence
for `p_c >= q_min`; relative preservation alone cannot establish it. Equal observed
success or a degenerate bootstrap interval does not prove preservation. The current
descriptive bootstrap keeps both seeds and all conditions within each state and
weights tasks equally; it is not a substitute for a planned confirmatory method.

Choosing the lowest observed cost across three alternatives requires simultaneous
inference covering the relevant quality and cost comparisons, or a separate
confirmation evaluation with the choice frozen first. A prespecified principal
contrast does not authorize an uncorrected post-hoc winner among the other cells.

## Decision output and evaluation

1. Check that candidate measurements answer the same contract. Missing applicable
   measurements leave eligibility unresolved; incompatible boundaries cannot be
   pooled. A failed record check is a reason to repair evidence, not a bad policy
   outcome.
2. Assess quality, resource caps, and any responsiveness constraints with the
   prespecified uncertainty method. Distinguish evidence of failure from lack of
   evidence of feasibility. Keep a feasible set and an unresolved set.
3. Compare paired costs among supported candidates. Return a preferred candidate
   only if the decision rule supports its advantage; otherwise return unresolved
   alternatives or insufficient evidence. Do not infer a tie from failure to
   detect a difference. Unresolved candidates prevent a claim of global optimality
   over the declared set, even when one feasible option can be recommended.
4. Apply the same quality information and selection rule to generation-count and
   measured-cost summaries. The count-based procedure may rank a proxy, but cannot
   certify an elapsed-time budget without a separately validated conversion.
   Evaluate the resulting choices on confirmation records using actual cost and
   quality. Report agreement, disagreement, or an unresolved decision; do not
   require a reversal or a predetermined saving. Human reader benefit is a separate
   endpoint requiring actual readers.

## Matching refinement and refresh

A planning approximation is `c_gen(d) ≈ c_encode + d*c_denoise`, so for full executed
prefixes `rho(h,d) ≈ c_gen(d)/h`. Here `c_encode` groups work independent of `d`;
it is not an isolated measured encoder kernel time. Use development measurements
to assess the approximation and screen candidate pairs. Two `d` values determine
an affine line but do not validate linearity or justify extrapolation.

Fix the clock, matching tolerance, resource denominator, candidate-selection
procedure, and evaluation identities before observing their task quality. Do not
call matching `d/h` a cost match. If matching per executed step, also report cost
per attempt and task quality: trajectories, early failure, and partial final
prefixes change episode expenditure. If the measured match fails, retain and
report a cost–quality comparison without an equal-budget claim.

Longer prefixes alter when observations revise queued actions. Neither nominal
`h*delta_t` nor synchronous simulator overhead measures physical responsiveness,
adverse contact, or recovery. Those require a separate validated evaluation.

## Submission boundary

Until these stages produce results, keep the measurement-foundation title and
claims. The archive motivates the resource axis; the constructed-defect suite
checks record consistency. Neither validates a deployment decision or adaptation
crossover. The original synthetic example remains unchanged, and no percentage
saving or labor-to-compute exchange rate is assumed.
