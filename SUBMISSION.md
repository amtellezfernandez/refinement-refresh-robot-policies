# RoboPAD 2026 submission notes

Title: **Refinement versus Refresh: Compute Allocation for Closed-Loop Robot
Policies**.

Positioning: a fixed-checkpoint empirical deployment study. The manuscript leads
with the completed primary transfer result: at a fixed two-action prefix, two
denoising steps record 139/160 successes versus 122/160 for five steps on eight
tasks excluded from configuration selection. It then reports the 12-configuration
discovery study, calibrated generation costs, and the selector limitation. Pilot
and record-verification tests are supplementary methods evidence. The obsolete
320-attempt plan, synthetic adaptation example, and generic lifecycle framework
are excluded from the paper. No universal allocation law or adaptation result is
claimed.

The [official call](https://robotpad2026.github.io/call-for-papers.html), checked
2026-09-10, welcomes position/discussion papers, early and negative results, and
reporting practices. It permits up to nine main-content pages (4--7 welcome),
with unlimited references/appendix, using anonymous NeurIPS 2026 workshop style.
The stated deadline is September 10, 2026 AoE. Submission is not performed by
this revision.

`main.tex` uses `[dblblindworkshop]`, anonymous authors, no identifying repository
link or acknowledgements, and a brief disclosure of LLM-assisted revision/source
checking/code review. Human authors remain responsible for research correctness.

## Build

```bash
latexmk -pdf -outdir=build main.tex
# Alternative: tectonic --keep-logs --outdir build main.tex
cp build/main.pdf output/resource_optimal_post_training.pdf
```

The stable output is `output/resource_optimal_post_training.pdf`. Build artifacts
are ignored. Render and inspect the final PDF, check reference resolution and
page count, and retain the validation record in `audit/REVISION_AUDIT.md`.

The manuscript evaluates prediction refinement and execution-prefix length across
twelve configurations on four development tasks, then reports the completed
primary frozen comparison on eight transfer tasks. The two-task pilot and
measurement verification appear in the appendix; no universal configuration
selector or cross-policy law is claimed.

The current PDF has ten pages: six main-content pages, one reference page,
and three appendix pages. The main text remains within the nine-page limit. It was rebuilt from the current source and
visually checked.
The integrated offline verifier now passes; see the revision audit for provenance
and limitations.

## Scientific status

- The discovery study adds 480 attempts across 12 configurations and four
  tasks, a dense-neural-arithmetic cost calibration, a quality-versus-compute
  figure, and model-based allocation predictions. All 87,364 dispatched actions
  reconcile with the episode and runtime records. See
  [the port audit](audit/DISCOVERY_PORT_20260913.md).
- Transfer validation is not inferred from discovery. The originally planned
  primary pair is complete: all 320 identities are present, and the paired
  result is reported in the manuscript. Remaining secondary confirmation
  settings are not used for claims. All recommendations are existing grid
  points, so this is a task-transfer test rather than prediction of an untested
  configuration.
- Generic trace v0.2 tests check structure and declared completeness, not whole
  event coverage, empirical accuracy, unit conversion, or causality.
- The archive audit reconciles 28 results: 22 v1, four v2, and two v3, with one
  v2 task/state and an 86-action same-condition replay. Historical v1 pairing is
  unverified.
- The executed offline experiment comprises 68 constructed trials / 272 checks:
  cumulative syntax, structure, aggregate, and full execution trace checks reject 4, 12,
  28, and 52 of 52 observable corruptions. All accept four controls and 12 probes
  whose requested claims remain unknown without external evidence. Unknowns are
  excluded from rejection denominators; constructed cases do not estimate
  natural error rates or robot performance.
- The mutation operators are validator-aware unit-test cases, so their rejection
  counts describe specified mechanism coverage rather than defect prevalence.
- The three-run timing proxy is illustrative and retrospective; it does not
  identify a solver effect or provide replicated deployment-cost evidence.
- The controlled four-cell, 80-attempt pilot completed with independent terminal
  outcomes, full runtime/checkpoint provenance, profiler reconciliation, and
  upstream parity. It is conditional on two tasks and five states per task.
- Its wall and CUDA timings overlap another GPU process and cannot support clean
  elapsed-efficiency claims; the paper reports measured work counts and outcomes.
- The 320-attempt broader protocol and matched-cost confirmation remain
  prospective. No matched-quality efficiency result is claimed.
- The bibliography is corrected; the 16-paper resource coding is not validated.
- The proposed learned-policy evaluation is descriptive on a fixed task set and cannot promise
  noninferiority, safety, population-wide performance, or a scaling law.
