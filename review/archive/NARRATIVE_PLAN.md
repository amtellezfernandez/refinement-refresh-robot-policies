# Proposal 2: narrative and motivation revision

Date: 2026-09-12. Starting revision: `bf55389`. Scope: Proposal 2.

## Central decision

For a required task quality and an expected number of deployments, which
adaptation recipe and deployment configuration fit the available resources?

The paper should make the cost of an ambiguous comparison concrete: a reader
cannot allocate a compute budget from generation counts, infer correction labor
from demonstration counts, or choose a lifetime budget from adaptation cost
alone. Unknown measurements can change which question is answerable.

## What to learn from Chinchilla

[Hoffmann et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html)
asks how to divide a training-compute budget between model size and data, then
tests the allocation empirically. Use that decision-first structure. It does
not establish an allocation rule for robot post-training. This paper supplies
measurement requirements for studying that question; it has no fitted robot
scaling law or measured adaptation/deployment optimum.

## Revision sequence

1. Lead the title, abstract and introduction with the resource decision and why
   scores or single counters cannot settle it. Introduce the contract as the
   response to that problem. Present the software components afterwards.
2. Bring deployment volume into the main argument through an accounting
   identity for one compatible additive resource. Preserve the original synthetic
   table: A uses less training GPU time and deployment elapsed time, while B
   uses less labor. It supports no crossover. Never change illustrative values
   to manufacture support for the motivation or add unlike clocks.
3. Explain the contract in reader language: what is compared, what is counted,
   and what supports the numbers. Connect the card and the three verification
   questions to those decisions. Remove the unvalidated legacy checklist
   digression and unsupported contrasts with narrative reporting.
4. Present archive evidence before the mutation suite. Equal generation counts
   with different recorded time illustrate proxy insufficiency. The mutation
   suite then explains why checking field presence alone misses recorded
   inconsistencies. Keep the distinct sample sizes and evidence limits.
5. Ask whether deployment compute is better spent on prediction refinement or
   observation-conditioned refresh. The initial grid compares measured cost and
   success with counts; a matched-cost test needs separate calibration. The 80-attempt pilot and
   320-attempt broader plan are allocations for this question, not results or
   automatically powered tests of equivalent quality. Keep execution details
   in the appendix.
6. Consolidate limitations, repeated archive paragraphs and submission notes.
   The conclusion should answer why the reporting instrument is needed for
   resource decisions, without implying that the current checks optimize them.
7. Recompute cited archive values, verify the hypothetical arithmetic, build the
   PDF, inspect all rendered pages, and check cross-references and page limits.

## Claim and evidence contract

| Statement | Evidence available | What remains open |
|---|---|---|
| Generation counts do not uniquely recover recorded elapsed cost | Three archived h=10 runs: 19 generations each; distinct host-time sums | Replicated causal effects of deployment settings |
| Labor, training GPU time, and deployment elapsed time impose different constraints | Original synthetic resource vector; no aggregate GPU total or crossover | Measured adaptation costs and quality for real recipes |
| Event reconciliation detects specified defects missed by field/totals checks | Frozen 52-defect suite, four controls and 12 external-evidence probes | Natural-failure detection, independent review, reader benefit |
| The instrument can support a prospective comparison | Pinned manifests, runners and analysis code | Completed paired rollouts, overhead/parity gates, uncertainty |

The necessity claim is conditional and practical: a decision about elapsed
budget requires elapsed measurements with compatible boundaries. This does not
assert that every paper needs every field, that all existing rankings are wrong,
or that an author-built validator certifies external truth.

## Correction during revision

The initial revision substituted hypothetical deployment costs of 12 and 9
GPU-seconds per attempt for the original 1.20 and 1.35 elapsed seconds. The
resulting 1,800-attempt crossover was arithmetic for that replacement, not
evidence from the original example or archive. The user rejected this change.
The replacement values and crossover claim are removed from the manuscript;
the original values and their distinct units are retained. The original
120/135-second quantities are totals for 100 attempts.

## Empirical work needed to answer the motivating question

1. Verify the runtime, checkpoint, upstream parity, reset/noise identities, and
   probe overhead before collecting comparison outcomes.
2. Select and freeze the paired allocation before viewing results: the 80-attempt
   pilot tests collection on two tasks; the 320-attempt plan expands the fixed
   task set. Neither is automatically a powered quality-equivalence study.
3. Report generation counts, internal work, rollout time, success, cost per
   attempt, and all-attempt cost per observed success for the same records.
   Compare orderings with paired uncertainty; retain agreement, disagreement,
   and inconclusive results. Do not require a reversal for the study to succeed.
4. For a real adaptation comparison, collect two actual recipes' adaptation
   expenditure and evaluate their deployment under compatible boundaries.
   Prespecify the quality requirement and relevant resource constraints.
   Claim a use-volume crossover only if those measured quantities support one.

## Embodied allocation literature refinement

The follow-up revision starting at `a51a4af` keeps the title and empirical scope.
It adds Pearce, Baniodeh, Kwok (RoboMonkey), and Sardana as targeted precedents;
[the source notes](../ALLOCATION_LITERATURE.md) map each to reporting requirements.
The central distinction is that scaling studies estimate resource-performance
relationships, while this contribution specifies and checks selected evidence
behind a stated allocation comparison.

The manuscript defines the finite-candidate allocation target under separate
resource limits. This is a decision definition, not a measured frontier or fitted
law. The archive motivates choosing the resource axis; the prospective h/d
comparison addresses local configuration choices. Before claiming budget
feasibility or preserved quality, the application budgets, quality threshold,
and required precision must be declared. No numerical threshold or new outcome
is invented in this revision.

The Resource Card template now prompts for candidate sets, resource caps, use
volume, quality requirements, representation/exposure units, and auxiliary-model
work when applicable. These are reporting prompts, not new implemented world-model
or action-verifier pipelines.

## Deployment decision refinement

The revision starting at `c89a898` makes refinement versus refresh the next
scientific question. The current grid remains a first cost–quality comparison;
the title and completed empirical scope are retained. VLA-Perf and SmolVLA's
asynchronous inference work are acknowledged directly. Predicted chunk size
remains 50; `h` changes its executed prefix.

[The decision protocol](../../experiments/DEPLOYMENT_DECISION_PROTOCOL.md) gives the
execution sequence: development validity and timing, the paired grid, a separately
calibrated matched-cost extension, and decision confirmation. It proposes `(5,5)`
versus native `(1,10)` as the principal joint reduction contrast without predicting
its result. Application quality margins, budgets, responsiveness requirements,
and confirmatory precision remain unset. The protocol explicitly handles unknown
eligibility, selection across candidates, unresolved choices, and proxy agreement.
It is a design; the existing descriptive analyzer is not a validated selector.

The appendix uses an affine generation-cost approximation only to explain
candidate screening. Two timing points cannot validate its functional form.
No universal allocation law, prescribed percentage saving, physical responsiveness
result, or synthetic crossover is introduced.

## Current completion status

The deployment decision revision is implemented in the manuscript and protocol.
The executed 80-attempt pilot now supplies the focused count--quality result on
two fixed tasks. All 20 paired blocks completed after the checkpoint, runtime,
instrumentation, reset, connectivity, and upstream-parity gates passed. The
principal joint work reduction lost 25 percentage points of success relative to
native; a one-action prefix outperformed a five-action prefix at both denoising
levels. Halving denoising at per-action refresh remains quality-inconclusive.
Concurrent GPU use excludes the retained durations from clean elapsed-efficiency
claims. The final PDF has eight main-content pages and 15 pages total. Offline
verification runs 79 tests with two environment-dependent skips. See
[the current audit](../REVISION_AUDIT.md).

## Earlier completion status

The narrative revision and correction above are implemented. The final PDF was
built and visually reviewed (six main-content pages, 13 pages total). Offline
verification completed with 77 passed tests and two skips; archived figures
reconcile. The empirical work listed above remains unexecuted.
