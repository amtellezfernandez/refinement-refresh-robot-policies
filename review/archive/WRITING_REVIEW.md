# Scientific writing revision

Revision: 2026-09-10. Scope: `paper/main.tex`; no empirical records, bibliography entries,
protocols, or generated result tables changed. This revision improves the argument
and organization; it does not claim NeurIPS acceptance or upgrade the evidence.
The PDF requires rebuilding and visual inspection after experiment integration.

## Editorial decisions

- The opening now starts with a concrete research decision: comparing adaptation
  methods requires a defined expenditure boundary as well as outcomes. Chunk
  refresh and solver effort give a precise example of ambiguity in call counts.
- The central question is evidence needed for a named resource claim. The
  contributions now distinguish the reporting contract, executable instruments,
  and the evidence evaluating those instruments.
- Related work explains how the contract connects a comparison to checkable
  records, while crediting Model Cards, measurement-context reporting,
  embodiment-specific reporting, taxonomies, and compute-allocation work.
  No unsupported first-of-its-kind or comprehensive-novelty claim was added.
- The contract table was corrected to avoid another universal-checklist error:
  a scoped outer-wall-cost comparison need not also measure internal solver
  stages. Evidence requirements follow the stated cost coordinate and claim.
- Cost per attempt and all-attempt cost per observed success are explicitly
  distinguished from mean cost conditional on success. Undefined and nonadditive
  cases remain explicit.
- The mechanism experiment follows a design/results/boundary-test structure.
  Its decisive limitation is visible next to its positive result: all validators
  accept the 12 probes whose requested claims need external evidence. The label
  `unknown` belongs to the experiment's interpretation, not a newly implemented
  native validator return value.
- Most archival smoke-test and quality details moved to an appendix. The main
  text retains the call-count/timing ambiguity and its confounding limitations.
- The preliminary 16-paper corpus no longer occupies a standalone appendix or
  drives a long list of otherwise unused citations. Its provisional status is
  disclosed in artifact transparency; its files and the bibliography audit remain
  preserved. Unsupported prevalence claims remain absent.
- The prospective section is compact and self-contained, ready to be replaced
  with actual execution methods/results. Its tasks are correctly described as a
  fixed set whose independence from unarchived historical runs is uncertified.
- Discussion now explains an adoption path and the connection to post-training.
  Repetitive disclaimers were consolidated into the relevant evidence boundary
  and a focused limitations paragraph.

## Evidence still needed for a stronger submission

1. **Live measurement validity.** Strict checkpoint identity, controlled resets,
   action parity, instrumentation overhead, and independent counter checks are
   more important than adding many success-rate episodes to an unverified runner.
2. **An executed information-recovery comparison.** Reduce the same completed
   records to success-only and success-plus-calls views. Score prespecified
   queries against the reconciled records, with unknowns and inconsistencies
   explicit. This tests the instrument's added information; it is not a reader
   study or a proof of better scientific decisions.
3. **Paired cost and outcome evidence.** Report every fixed condition with actual
   NFE and measured boundaries. Predictable `ceil(T/h)` reductions establish
   scheduling correctness, not a new efficiency method. Preserve differences
   and uncertainty even when no cost/quality ranking is supported.
4. **Scope of post-training relevance.** A frozen-policy study evaluates
   deployment accounting. An adaptation intervention is still needed for claims
   about adaptation effectiveness, tuning expenditure, or lifecycle break-even.
5. **Breadth and novelty.** Constructed operators were selected with knowledge of
   the validator and applied to related source records. They demonstrate
   mechanisms, not robustness across naturally occurring failures or multiple
   evaluators. A broader reporting contribution needs independent adoption or
   additional implementations; a reader-benefit claim needs actual readers.

## Handoff for the GPU-result integration

Update the abstract, evaluation overview, Section 5 (`sec:prospective`), the
limitations paragraph, and the conclusion together when completed GPU results
are available. These currently and correctly say that live overhead and broader
task behavior remain unmeasured. Keep the archived-data and new-run populations
separate. The template, title, anonymity, generated-table inputs, and existing
result labels remain intact.

Static checks cover balanced LaTeX environments/braces, existing included files,
and citation/reference resolution from source. They do not replace compilation
or page rendering.

Executed checks: 11 distinct cited keys resolve in the bibliography; all labels,
table inputs, and graphics resolve; environment nesting balances; 120 unescaped
opening and closing braces match in the combined manuscript/table sources.
`git diff --check` reports no whitespace error in the owned files. The abstract
has 190 whitespace-delimited words; main source before references has 2,694,
including LaTeX markup. No PDF build was performed in this writing subtask.
