# Revision audit

## Current revision: 2026-09-13 ICLR-first development plan

### Remote instrumentation repair

The historical stale-input run remains immutable. Its runner was patched locally
to log `input_refreshed`, current and supplied observation hashes, recursive
field hashes, supplied-observation tick, and supplied-observation age on every
step. The patch passes the offline runner tests (56 run, one environment skip)
and was deployed as a separate remote file with SHA-256
`09fbef924edbf075ea245db6d59ba4e181785e0d3a2f66d632f264d82970458f`.

The remote launch initially failed before inference because the shell command
did not expose LeRobot, then a second isolated launch reached an installed
numba/coverage incompatibility. Both failed output directories are preserved.
An isolated fourth run is active at
`/mnt/NVME1/projectx-robopad-proposal2/results/refinement_refresh_final_001/stale_instrumented_004`,
using the verified `lerobot312` environment and an isolated `sitecustomize`
compatibility shim. The first journal records the expected 0,1,2,3,4,0 age
sequence and matching/differing current-versus-supplied hashes at refresh and
held ticks. No completed result from this rerun is reported yet; it must finish
and be audited separately.

Adopted ICLR 2027 as the main-conference target. Renamed the single active
roadmap from `NEURIPS_DEVELOPMENT.md` to `ROADMAP.md` and updated active links;
no additional Markdown report was created. The plan prioritizes second-policy
feasibility and the existing observation-age diagnostic, followed by a separately
frozen targeted protocol. A new quality-preservation campaign is conditional
follow-up rather than a prerequisite for cross-policy evidence. Official venue
dates, Paris conversions, format requirements, AI disclosure, and author-side
submission tasks are recorded with primary-source links in `SUBMISSION.md`.

PDF review identified two distribution issues for the future ICLR rewrite:
Figure 1(b) gives main-text space to host-time sensitivity rather than the
measured denoising-independent floor, and a forced break isolates the short
discussion on page 7. The roadmap proposes a nine-page content budget and
revised figure/appendix placement. These structural changes and conversion to
the ICLR template are planned, not represented as implemented. No new-result
sections or stronger empirical claims were inserted in the ICLR plan; the
working manuscript now adds the completed stale-input diagnostic below.

Remote inspection found the diagnostic complete: 40/40 planned attempts, 20/40
successes with held inputs versus 28/40 on the matched fresh-input discovery
condition. The paired difference is -20.0 percentage points, with a descriptive
task-conditional 95% state-bootstrap interval of [-32.5, -7.5]. All 40 initial
observation hashes match their fresh-input partners, and initial state pairing
is verified. The intervention refreshes images and proprioception every five
ticks while generating every tick at h=1,d=5. However, step records contain
action age rather than supplied-observation age and lack per-tick supplied-input
hashes. The result therefore supports an observation-freshness contribution but
does not isolate every implementation detail or establish a clean causal
decomposition. No remote process was running at inspection; no historical
records or frozen protocols were modified.

The working manuscript now has an expanded AI Use Statement covering conceptual
framing, hypotheses, design discussion, interpretation, source searches, writing,
code implementation/review, and figures/tables. It describes recorded automated
checks without claiming independent human validation. Authors still need to
review this disclosure and complete the matching submission form.

Validation: all 11 publication assets reproduce bytewise; the confirmation
analyzer reproduces its five outputs, 1,120 unique attempts, two excluded
repeats, and 171,805 canonical actions. Tests: 55 run, 54 passed, one environment
skip. Generated visuals and main-campaign evidence are unchanged by hash; the
working manuscript includes only the audited stale-diagnostic result and
disclosure additions. Active documentation links resolve. The rebuilt
14-page PDF was rendered; pages 1--13 are pixel-identical to the prior reviewed
version, and the changed disclosure page was visually checked. No overfull
boxes or undefined references/citations; the pre-existing underfull-page notice
and style-path warning remain. The legacy integrated-verifier limitation is
unchanged and was not represented as repaired.

Working PDF SHA-256:
`9b0b081e81575fdb835de59d6470cc8e6d82cc219dcd08d37ae649fb5f611fae`.
Both submitted PDFs and the primary-only supplement remain unchanged. No
checkpoint was selected, inference launched, or conference submission made.

## Previous revision: 2026-09-13 print-size and numerical-graphics preflight

A second visual pass checks the current manuscript, including the approved
abstract, at its actual print dimensions. Both vector figures now use 10-point
text at a 5.5-inch width, matching the caption size. Figure 1 labels the plotted
choice as the rule-selected prefix rather than implying that it shows every
feasible prefix. All twelve discovery configurations and seven confirmation
contrasts remain visible, with the original uncertainty and analysis grouping.

Table headers no longer repeat denominators already shown in the cells.
The prediction table has grouped configuration and percentage-point headers.
The task table includes the previously reported configuration totals, separated
from its 20-attempt task rows; no new statistical estimate was computed.
The runtime table uses deliberate line breaks and keeps version numbers intact.
The approved abstract and all pre-existing numerical table rows are unchanged.

Five added tests verify every plotted discovery estimate and interval, every
confirmation effect and interval, the task totals, and rejection of clipped or
undersized figure text. Results: 55 tests, 54 passed and one environment skip.
All publication assets reproduce bytewise. The complete confirmation audit
again reproduces its five evidence outputs and reconciles 1,120 unique attempts,
two excluded repeats, and 171,805 actions. Evidence, calibrations, frozen
predictions, both submitted PDFs, and the submitted supplement are byte-identical.

All fourteen pages were visually reviewed, including grayscale figure checks.
Independent PDF inspection confirms 10-point figure text, no clipped figure
words, embedded fonts, no Type 3 fonts or raster images, and US Letter pages.
The build has no overfull boxes or undefined references/citations; one underfull
page notice and the existing style-path warning remain. The template, margins,
and body font sizes are unchanged. This is a presentation preflight, not a claim
of full main-track submission readiness; the known legacy verifier limitation
remains separate. No new Markdown document was created.

Working PDF SHA-256:
`08a9cdb8d7a3623f53d5f423d4de167e6072cceabee181ed51be8d23bdd562b0`.

## Previous revision: 2026-09-13 abstract precision

Applied the user-supplied abstract. The opening now describes two distinct
uses of compute exposed by chunked policies, without implying an exhaustive
taxonomy. Profiling establishes a denoising-independent cost per generation,
not a component-level attribution of every intercept FLOP to freshness.
The abstract retains the primary refinement result, reports the longer prefix's
60.7% lower arithmetic per attempt with quality preservation unresolved, and
ends with the question of when a fresh prediction is worth its fixed cost.

Checksums confirm that the manuscript outside the abstract, all publication
assets, both submitted PDFs, and the submitted supplement are unchanged.
The presentation reproducibility check passes. The rebuilt 14-page PDF was
visually reviewed; it retains seven main-content pages, one reference page,
and six appendix pages. No overfull boxes or undefined references/citations;
one underfull-page notice and the existing style-path warning remain.

Working PDF SHA-256:
`696bebe181da11254d71848b680ab2464beb1539c82333dec5c55c359059e4f4`.

## Previous revision: 2026-09-13 publication tables and figures

Both figures and all eleven tables received a presentation pass. No experiments,
statistical estimates, bootstrap draws, or frozen predictions were changed.

- Figure 1 now shows every discovery configuration on a categorical denoising
  axis with separated paired-state intervals, plus the resource-dependent
  feasible-prefix comparison. The success axis spans 0--100%; shapes supplement
  color. This replaces the crowded cost--success scatter, not its evidence.
- Figure 2 shows all seven confirmation contrasts in documented order. Frozen
  and amended analyses form separate groups; the filled primary diamond denotes
  prespecification, not a significance threshold. Exact effects and intervals
  remain visible, including every interval spanning zero.
- Figures are vector PDFs designed at the actual 5.5-inch text width, with
  9--10 point labels and embedded fonts. Both were checked in color and grayscale.
  The renderer rejects undersized or out-of-canvas text; independent PDF text
  geometry checks find no clipped words. No Type 3 fonts are present.
- Tables use booktabs, decimal alignment, consistent rounding, explicit units,
  and grouping by prefix or task suite. Selector choices and numerical frozen
  predictions are separate appendix tables, eliminating the cramped eight-column
  layout. No outcome-dependent winner highlighting was introduced.

`render_paper_visuals.py` owns `paper/generated/`: eight tables, two figures,
and a source/output hash manifest. It reads the retained reports without fitting
or resampling. Its `--check` mode regenerates every asset in a temporary directory
and verifies exact bytes. Nine presentation tests cover rounding, complete
contrast coverage, primary evidence, and malformed inputs. The original
evidence reports and their analysis-generated tables/figures remain byte-identical.

Validation: 50 runner/analysis/presentation tests, 49 passed and one
environment-dependent skip. The full confirmation audit reproduces its five
saved outputs: 1,120 unique attempts, two excluded repeats, and 171,805 actions.
Publication assets reproduce exactly. All 14 rendered pages were reviewed;
the final changed pages were re-rendered and checked. The working draft has
seven main-content pages, one reference page, and six appendix pages. There
are no overfull boxes, undefined references, or undefined citations. Two
underfull-page notices and the existing style-path warning remain.

The existing anonymous workshop template, margins, and body fonts are unchanged,
consistent with the [NeurIPS formatting constraints](https://neurips.cc/Conferences/2026/MainTrackHandbook).
This is a visual-quality pass, not a full main-track submission-readiness audit.
The legacy integrated verifier's frozen-runner mismatch remains a separate
known limitation. No new Markdown report or roadmap was created.

Working PDF SHA-256:
`a9ecbaf7cc51348ea3d4e7993552c7d5dbebbe4660ce4dfd71cfdb50723c705f`.
Both submitted PDFs retain SHA-256
`9a23d4e8b603f20b2b0aed45ab68f55300a975d979d9814259ff5679499c19f5`;
the submitted primary-only supplement is unchanged.

## Previous revision: 2026-09-13 argument-flow revision

The paper now follows allocation principle, fixed-generation cost, controlled
study, frozen refinement results, descriptive refresh results, then synthesis
and limits. This is an editorial restructuring, not a new experiment or analysis.

- The introduction puts the fixed-cost consequence before the result preview
  and states the two empirical questions separately.
- A dedicated allocation section presents the compute-rate objective, realized
  episode expenditure, and measured arithmetic floor before study details.
- Methods describe discovery, freezing, and confirmation together. All three
  original comparisons are explicitly fixed-prefix denoising tests under common
  ceilings, not equal-rate allocation experiments.
- The first main-text table is now the full confirmation overview. The primary
  counts, interval, and nominal cost reduction stay together on page 5; the
  descriptive refresh comparisons follow on page 6.
- Detailed fitting, selector diagnostics, runtime history, language shapes,
  timing sensitivity, and primary task sensitivity are in the appendices.
  Figure 1 follows its introductory paragraph. The discussion leads with the
  feasibility-versus-usefulness conclusion, then limits and next experiments.
- Original-versus-amended analysis status, the simple-selector limitation,
  action-position confounding, and the absence of an equivalence or
  non-inferiority result remain explicit. The stale-input cutoff is unchanged.

Validation: generated-table inputs, inline numerical tables, figure assets, and
citation sets match the previous revision. The full confirmation analyzer
reproduces all five saved artifacts and reconciles 1,120 unique attempts,
two excluded repeats, and 171,805 canonical actions. Runner/analysis tests:
41 run, 40 passed, one environment-dependent skip. No experiment inputs,
analysis code, frozen predictions, or submitted PDFs were changed.

The reviewed working PDF has six main-content pages, one reference page, and
six appendix pages (13 total), versus eight main-content pages before this
revision. All pages were visually checked; changed pages were re-rendered and
reviewed after the final edits. Cross-references resolve, with no undefined
citations/references or overfull boxes. The existing style-path warning and
two underfull-page notices remain. Fonts, margins, and the workshop style are
unchanged. The legacy integrated verifier's frozen-source mismatch recorded
below is not repaired or represented as passing.

Working PDF SHA-256:
`3027582d789d82e2e6b16e4b3668b5d20ecf0ccf9477ccb6a6568b9b65df87ad`.
Both submitted PDF copies retain SHA-256
`9a23d4e8b603f20b2b0aed45ab68f55300a975d979d9814259ff5679499c19f5`.

## Previous revision: 2026-09-13 completed confirmation

The working manuscript now includes all 1,120 unique confirmation attempts
alongside 480 discovery attempts. Its abstract and introduction retain the
refinement-versus-refresh allocation principle, full-generation arithmetic
floor, and distinction between resource feasibility and useful allocation.
The submitted PDFs are not modified.

The frozen primary result is unchanged: `(2,2)` versus `(2,5)` gives 139/160
versus 122/160 successes, a +10.625 percentage-point difference with descriptive
95% interval [+4.375, +16.875]. All three original comparisons are now reported;
the quarter- and full-native intervals span zero. Four later-amended descriptive
contrasts are separately identified. No confirmation outcomes refit the selector.
The `(5,2)` setting gives 138/160 successes at 16.1 reference TFLOPs per attempt,
versus 41.1 for `(2,2)`; their quality difference is unresolved, not equivalent.

### Evidence and reproducibility

- The completed snapshot contains 1,122 raw completed records across four
  phases: 799 + 92 + 6 + 225. Documented precedence excludes two repeats with
  identical outcomes and action-stream hashes.
- All 1,120 planned identities and 160 complete seven-condition paired blocks
  reconcile. Initial states/observations, horizons, seeds, checkpoint/runtime
  controls, and recorded schedule hashes agree.
- Full journals reconcile all raw completed attempts, including the repeats;
  the canonical outcome sample contains 171,805 dispatched actions. One trailing
  incomplete attempt with ten actions is explicitly recorded and excluded.
- The primary subset remains 320 attempts / 50,451 actions. The continuation is
  completion of the original schedule, not an independent replication cohort.
- `experiments/tooling/analyze_confirmation.py --check` reproduces the report
  and four LaTeX tables from the compressed snapshot. Original comparisons use
  seed 20260912; amended contrasts use 20260913. The submitted primary interval
  also reproduces with its original independent RNG reset.
- Numerical results, runtime comparisons, input hashes, and journal audits
  exactly match the earlier independent full-results study. Post-hoc interaction
  analyses from that exploratory study are not promoted into the manuscript.
- New analyzer/helper code passes Ruff checks. Runner/analysis unit discovery
  runs 41 tests: 40 pass, one skips because the pinned source checkout is absent.

Snapshot SHA-256:
`36c1d5f2dcaef85664ec7a1bfe7409c849b3be17831566e783d24a00ea4e8df0`.
The generated confirmation report retains source, validator, and input hashes.

### Limits and remaining work

The stale-input diagnostic was incomplete at the analysis cutoff and supplies no
outcome claim here. Timing-monitoring gaps, reference language-shape calibration,
fixed-task uncertainty, unadjusted secondary intervals, and action-position
confounding are retained. No non-inferiority margin is chosen retrospectively.
The existing README and roadmap now distinguish completed confirmation from
the next mechanism and quality-preservation studies; no new Markdown files were
added for this revision.

The legacy integrated verifier is **not passing**: 79 tests run, with 76 passes,
two skips, and one error, `Frozen input drift: runner_sha256_lf_normalized`, in
instrument validation. This pre-existing frozen-source mismatch is separate
from the completed-confirmation audit. Its protocol hashes are left unchanged.

### PDF review

The working build has eight main-content pages, one reference page, and four
appendix pages. All 13 rendered pages were visually checked for readable tables,
equations, references, and page transitions; the two pages changed in the final
pass were re-rendered and inspected again, and the other images match exactly.
The build has no undefined citations,
undefined references, or overfull boxes. The existing style-path warning and
two underfull-page notices remain; no template geometry or font sizes were
changed.

The reviewed build was copied to root `paper.pdf` and matches it byte for byte.
Working PDF SHA-256:
`f2fc3e02dce765e05172b32e7c353cf3dc552e4ea532cbaafe722d4c3adfe198`.

Both frozen submitted PDF copies retain SHA-256
`9a23d4e8b603f20b2b0aed45ab68f55300a975d979d9814259ff5679499c19f5`.

Earlier sections below are historical validation records, not current status.

## Previous revision: 2026-09-12 executed paired pilot

Starting commit: `622b353`. The verified SmolVLA checkpoint and pinned LeRobot
source were evaluated on the frozen 80-attempt focused plan. The design contains
two LIBERO Spatial tasks, five states per task, two policy seeds per state, and
four `(h,d)` conditions. All 80 attempts and 20 paired blocks completed; the
retained journal reconciles every initial state, observation, generation, noise
key, dispatched action, outcome, and episode total.

Development checks passed deterministic replay, image connectivity, independent
work counters, profiler agreement, a zero-action environment control, and exact
controlled-noise parity with the pinned upstream rollout loop. The paired analysis
averages seed repeats within state, resamples the five paired states within each
task, and weights both tasks equally.

Results: native `(1,10)` succeeds on 13/20 attempts; `(1,5)` on 12/20; `(5,5)`
on 8/20; and `(5,10)` on 6/20. The principal `(5,5)` minus native success
difference is `-0.25`, with descriptive paired state-bootstrap 95% interval
`[-0.40,-0.10]`. One-action minus five-action execution is `0.20 [0.05,0.35]`
at `d=5` and `0.35 [0.15,0.55]` at `d=10`. The `(1,5)` minus native difference
is `-0.05 [-0.20,0.10]`, which does not establish equivalence or degradation.
Another GPU process overlapped execution, so retained clock values are excluded
from clean efficiency claims; the paper uses reconciled generation and denoising
counts as its resource measures.

The manuscript also adds a compact main-text Resource Card, one worked
aggregate-preserving mutation, the closed-loop cost--outcome motivation, and a
pinned MLPerf Inference Rules reference. The current PDF has eight main-content
pages, two reference pages, and five appendix pages. Integrated verification runs
79 tests with two environment-dependent skips, and the numerical audit matches
all reported cell totals and paired contrasts to the retained journals.

Final PDF SHA-256: `b7f23b28e665916122a5aa308943203accda577d5f6083d53154eed7e0a41b26`.

## Previous revision: 2026-09-12 deployment allocation and decision protocol

Starting commit: `c89a898`. The title and measurement-foundation contribution are
retained. The abstract, introduction, and Section 5 now connect deployment cost
to prediction refinement versus observation-conditioned refresh. Redundant prose
was removed; the main text still fits six pages.

The appendix separates the four-cell cost–quality comparison from a matched-cost
extension. Its affine cost expression is an unvalidated planning approximation.
The planned principal joint reduction contrast is `(5,5)` versus native `(1,10)`;
no result or quality margin is assumed. A relative-quality lower-bound criterion,
unknown eligibility, selection across candidates, and independent confirmation
are specified. Generation-based and measured-cost choices may agree, differ, or
remain unresolved. Query recovery is not presented as decision validation.

The new [decision protocol](../experiments/DEPLOYMENT_DECISION_PROTOCOL.md) makes
the execution stages and missing application thresholds explicit. The Resource
Card and experiment design point to those requirements. Existing runner/analyzer
code remains descriptive; a validated selector has not been implemented or tested.
VLA-Perf is added as direct systems prior work, and SmolVLA's asynchronous inference
is acknowledged; see [source notes](ALLOCATION_LITERATURE.md).

Validation:

- 37 unique bibliography entries, 14 cited keys, all resolved. The VLA-Perf title,
  four authors, version, and methodological scope were checked against the primary
  record and manuscript.
- Integrated offline verification: **79 tests run, 77 passed, two Torch-dependent
  skips**; 28 archived results and 68 mutation trials / 272 layer checks reconcile.
- LaTeX/BibTeX build completed with no undefined citations/references or overfull
  boxes. The remaining package-path warning is unchanged. All 13 final pages were
  visually inspected: six main-content pages, two references, five appendix.
- Original archive data, synthetic recipe values, and executed results are
  unchanged. The PDF copied to the stable output path matches the build.

Final PDF SHA-256: `5b8c26c154ab084e95fcd5b5d4375caf06999bd5e723a9cab15769fc45a04572`.

### Runtime follow-through and actual blocker

The base VLM tokenizer/configuration download completed; both checkpoint
normalization files match publisher SHA-256
`7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e`.
The model file obtained after an interrupted/resumed transfer failed integrity
verification: its local SHA-256 was
`8b4d3a17ca79eb8786946943dffb25733e0af254c02c2f12c1145cc076e607f0`,
whereas the pinned Hugging Face revision's API reports
`71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`.
The bad file was quarantined. The failure establishes that those local bytes are
unusable; it does not diagnose the precise cause or implicate the publisher's file.

A fresh HTTP checkpoint transfer and a PyPI dependency-install retry each reached
their 300-second process limit after network stalls/timeouts. Torch is still
absent from the isolated Python 3.12 runtime, and the verified checkpoint revision
marker is absent. Setup logs, the quarantined file, and partial transfers remain
under ignored `build/runtime/`. No download/install process from these retries
is left running. Strict policy loading, development validity gates, timed rollouts,
and the prospective 80/320-attempt evaluations have not run. These setup failures
are not new policy outcomes or mutation-suite evidence.

## Previous revision: 2026-09-12 embodied allocation literature

Starting commit: `a51a4af`. Title and empirical scope retained. The introduction
now states the measurement role alongside direct embodied allocation studies.
Four references were added: Pearce (ICML 2025), Baniodeh (versioned 2025 technical
report), Kwok/RoboMonkey (CoRL 2025), and Sardana (ICML 2024). See
[the source notes](ALLOCATION_LITERATURE.md) for metadata, methodological locators,
and the distinction between source findings and our reporting requirements.

Section 2 defines the budget-constrained decision over a finite declared candidate
set, with resource caps compared componentwise. It handles an empty feasible set
and unknown measurements explicitly. The existing accounting identity retains its
compatible-unit scope and now cites prior inference-demand-aware allocation.
The archive and prospective h/d comparison point back to this decision target.
Neither is claimed to establish a scaling law, an empirical frontier, iso-compute
conditions, or a quality-preserving ranking. The Resource Card template adds the
corresponding reporting prompts without claiming new pipeline implementations.

Validation:

- 36 unique bibliography entries; 13 keys cited in the current main text; all
  four added references are used and every citation key resolves.
- Integrated offline check: **79 tests run, 77 passed, two Torch-dependent skips**;
  28 archived results reconcile; 68 mutation trials / 272 layer checks reproduce.
- LaTeX/BibTeX build completed without undefined citations/references or overfull
  boxes. All final pages were visually inspected (unchanged pages were checked
  against previously inspected renders).
- Final PDF: **six main-content pages, two reference pages, five appendix pages**
  (13 total). Conclusion and its text fit on page 6.
- Original synthetic quantities and empirical artifacts remain unchanged. No
  replacement crossover or new performance result was added.

Final PDF SHA-256: `7f6d907bb6534c3f8d83ac4492c2eb0f1289e57a5147842b64b453c363ba19dd`.

### Runtime prerequisite work in this turn

The current host has an NVIDIA RTX 2000 Ada Generation Laptop GPU with 8 GB VRAM
and driver 580.173.02. CUDA matrix multiplication passed in the existing Python
3.10 / Torch 2.7.1 / CUDA 12.6 environment. This is not the experiment runtime.
An isolated Python 3.12.11 environment was created under `build/runtime/venv`;
the extracted LeRobot source matches all seven pinned source digests. The source
archive SHA-256 is
`429a869b02b3fd8b9745f390951148f5185fce42d7bd7678337981d4510324d7`.

Installing Torch 2.8.0, torchvision 0.23.0, torchcodec 0.7.0 and the pinned
LeRobot extras encountered network timeouts. A retry limits concurrent downloads
to three. The checkpoint and base tokenizer/configuration downloads are also
incomplete. Logs and partial downloads are retained in `build/runtime/`; neither
strict model loading nor policy gates nor the prospective study has run. Runtime
availability must not be inferred from the completed source/GPU checks.

## Previous revision: 2026-09-12 narrative and motivation

Starting commit: `bf55389`. Scope: Proposal 2 manuscript and supporting narrative
notes. See [the revision plan](archive/NARRATIVE_PLAN.md).

The title is now *Accounting for Robot Post-Training: Comparing Policies under
Resource Budgets*. The argument starts with the decision about required task
quality and available adaptation/deployment resources, introduces the needed
measurements, and then explains the card and validation checks. Archived timing
observations precede the constructed-defect suite. The next study asks whether
generation counts and measured cost give consistent comparisons on the same
attempts. Detailed execution gates are in the appendix. No new policy evaluation
or adaptation results were collected for this revision.

The original synthetic recipe values are retained: training GPU-hours 2.0/3.5,
correction person-hours 6.0/1.0, and deployment elapsed seconds per attempt
1.20/1.35. At 100 attempts, deployment totals are 120/135 seconds. These values
support no time-cost crossover or common GPU-time total. An intermediate rewrite
substituted different hypothetical quantities and calculated a 1,800-attempt
crossover. That replacement was rejected and removed; it is not evidence from
the original table or archive. No labor-to-compute exchange rate is used.

Validation:

- Integrated offline verification completed: **79 tests run, 77 passed, two
  Torch-dependent skips**; all 28 archived results reconcile; 68 mutation trials
  and 272 layer checks reproduce. This supersedes the intermediate protocol
  incompatibility described in the historical audit below.
- Recomputed archived timing observations match the paper: 19 generations in
  each of three records, with host-time sums 4.88, 8.14, and 15.31 seconds.
- Final LaTeX build completed with no undefined references or overfull boxes.
  The PDF has **six main-content pages, one reference page, five appendix pages**
  (12 total). Remaining package-path and underfull-box warnings have no visible
  clipping or missing text.
- All pages were rendered and inspected. After the final appendix layout fix,
  pages 9--11 were inspected again; the other rendered pages were byte-identical
  to the inspected versions. The Resource Card now shares a page with its
  introduction. The final PDF contains no 1,800-deployment claim.
- Raw archived results, validators, evaluation manifests, and generated empirical
  tables remain unchanged. The runtime guide now describes the pilot as a fixed
  task set, consistent with its unverified holdout status.

Final PDF SHA-256: `ec3560a7df8ae1560777970e5af8a8f3ef253733056976a06b28a6b19b9c20e3`.

## Historical audit: 2026-09-10

Date: 2026-09-10. Scope: Proposal 2 only. Starting commit: `dfc300a`.
This revision includes a research/code/source audit and an executed offline
mutation experiment. It is not an independent replication of the GPU experiments
or independent human annotation. New GPU results are not reported here.

## Main assessment

The old manuscript combined a broad lifecycle/scaling proposal, a weak disclosure
headline and tiny deployment pilots. None of the experiments tested an adaptation
method or justified quality-preserving efficiency. Its strongest workshop-sized
contribution is a **comparison contract plus a tested reporting instrument**,
supported by a finite constructed-error experiment and an audit of existing runs.

The revised title is *Accounting for Robot Post-Training: Comparison Contracts
and an Auditable Deployment Ledger*. The paper now leads with an explicit
comparison boundary, separates trace typing from measured execution, and treats
the small deployment records as instrumentation evidence. The newly executed
offline experiment distinguishes structural, aggregate, and full ledger checks
without claiming a natural-error detection rate. The unvalidated
literature codes move to an appendix/corpus-development role.

## Flaws and concrete corrections

| Finding | Why it mattered | Correction |
|---|---|---|
| Frozen-policy settings were presented under lifecycle scaling | No weights/data/adaptation budget changed | Narrowed empirical claim to deployment measurement; adaptation extension explicitly future work |
| 0/16 checklist headline lacked comparison-specific evidence | Different quantities/phases and inherited costs were mixed | Removed headline/figure/full matrix and unsupported VLA-RL card; preserved legacy CSV with a new extraction template |
| Historical runs called paired or deterministic | Equal binary outcomes/aggregate counts do not establish shared starts or actions | Corrected design/README/manifest; v1 remains unverified; v2 replay is limited to one recorded stack |
| Chunk reduction was treated as an efficiency result | Queue schedule implies ceil(T/h), and T changes with failures | Stated identity and open-loop policy feedback tradeoff; removed equivalence/safety implications |
| Model calls substituted for actual cost | Same 19 chunks in three solver cells have different recorded host intervals | Added a generated descriptive proxy counterexample, with clock/confounding limits |
| Summaries counted only successes[0] per task | Additional episodes could be silently discarded | Centralized all-episode outcome accounting; regression with two successes among three episodes |
| Ledger acceptance did not require a completed evaluator result | Partial journals could appear eligible for reporting | Added final-wrapper and task/group/overall reconciliation; incomplete artifacts rejected |
| Generic validator did not validate resource numbers/status | Negative/nonfinite costs and unsupported operators could pass; unknowns could not be encoded | Trace v0.2 schema and semantic validation, null unknowns, explicit units/boundaries/aggregation/status |
| Trace was called causal and promised omitted-cost detection | Types cannot establish causal effects or detect an omitted whole event | Renamed provenance trace; separate structural validity, declared completeness and independent execution reconciliation |
| Instrument benefits were described without an executed comparison of validation layers | Passing isolated unit tests did not expose which evidence each layer adds | Froze and executed 68 constructed trials across four cumulative layers, with observable-defect, control, and unknown categories kept separate |
| Timing/storage aggregation was underspecified | Nested durations double count; peaks and retained snapshots are not additive | Added explicit aggregation contract and distinct host/device/warm-up boundaries |
| References contained author/title/venue/version errors | Incorrect attribution and source-version drift weaken claims | Corrected metadata with primary-source register; added SmolVLA, LIBERO, Model Cards and energy-reporting precedents |
| Proposed 600+4,800-run study was broad and underjustified | Large compute burden without promised precision; many selection/quality claims | Four locked cells, 320 proposed attempts, runtime and holdout gates; descriptive fixed-task inference |

## Recomputed evidence

[The machine-generated archive audit](../experiments/real_smolvla/ARCHIVE_AUDIT.md)
reconciles all 28 completed records: 22 v1, four v2, and two v3. V2 verifies shared starts
across three comparisons and exact same-condition replay of 86 dispatched actions.
Counts 92/92, 66/14 and 86/9 are selected actions/neural generations; twenty
additional warm-up generations occur per job. All raw result files are preserved.
Input digests normalize CRLF to LF to remain reproducible across Git checkouts.

The v1 broad sweep's 14/20 versus 12/20 successes remain descriptive. No completed
broad h=1 baseline is present. Historical v1 pairing, independent terminal outcomes,
per-run checkpoint/hardware/software provenance and matched instrumented/upstream
execution cannot be established from the archived ledgers.

The pinned public checkpoint configuration was fetched directly from
[the exact revision](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json)
and confirms `chunk_size=50`, `n_action_steps=1`, `num_steps=10`. This verifies
the model configuration statement, not the identity of weights loaded in old runs.

## Newly executed offline validation experiment

[Frozen protocol and methodology](../experiments/instrument_validation/README.md)
record 17 prespecified cases applied to all four archived v2 sources. Protocol,
source, schema, and validator/runner digests were saved before the first trial.
The experiment executed **68 constructed trials and 272 validation checks**:
52 observable corruptions, four untouched controls, and 12 unobservable probes.
All raw sources remain unchanged.

| Cumulative validation layer | Observable defects rejected | Controls accepted | Accepted probes outside the evidence |
|---|---:|---:|---:|
| Strict JSON syntax | 4/52 | 4/4 | 12 unknown |
| + Structural schema | 12/52 | 4/4 | 12 unknown |
| + Aggregate reconciliation | 28/52 | 4/4 | 12 unknown |
| + Full ledger reconciliation | 52/52 | 4/4 | 12 unknown |

There were zero validator crashes. Every source/case/layer verdict and rejection
reason is preserved in [trials.json](../experiments/instrument_validation/trials.json).
The summary, LaTeX table, and vector PDF/SVG heatmap derive from those executed
checks. At the time of the recorded experiment, re-execution with
`run_experiment.py --check` reproduced the results and three experiment-integrity
tests passed. An intermediate checkout rejected the frozen protocol after a validator revision
for v2 archive compatibility. The current integrated verification is reported in
the 2026-09-12 entry above. The heatmap was visually
inspected.
The main manuscript includes the concise table and an appendix heatmap.

The 12 unknowns are **technical acceptances**, not newly implemented native
validator outputs. Four change both action streams and recompute consistent
hashes; an external expected-action reference is needed to identify the rewrite.
Eight leave the source bytes unchanged and query preprocessing/rendering coverage.
They do not invent missing measurements or establish that a real stage was
omitted. The protocol annotates their requested claim as unknown and excludes
them from the observable-defect denominator.

The mutation operators were chosen with knowledge of the validator contract.
The four source records share one task/state and include a repeated condition;
they are not independent robot samples. These counts establish specified
mechanism checks, not real-world error prevalence, exhaustive robustness,
human-reader benefit, policy quality, or lifecycle completeness. No new learned
policy inference or simulator rollout was used in this experiment.

## Redesigned experiment

[The protocol](../experiments/EXPERIMENT_DESIGN.md) and
[proposed manifest](../experiments/planned_evaluation.json) specify four
conditions (h=1/5 crossed with d=5/10), eight reserved tasks, five states per task,
and two keyed policy seeds: 320 attempted episodes in complete blocks. Schedule
seed is 20260910. Runtime compatibility and holdout status are explicitly unverified;
the generator does not launch experiments or certify those external facts.

Primary tests address counter correctness, instrument overhead, and recovery of
resource queries from the same rollouts. The offline mutation experiment executes
the recorded-evidence mechanism checks. The separate GPU experiments require upstream parity,
independent terminal records/internal-work counters, reset/RNG gates and measured
budget before collection. No quality-equivalence, safety, population-wide LIBERO,
or scaling-law claim is promised. A null utility result is publishable.

## Bibliography and corpus

[Bibliography audit](BIBLIOGRAPHY_AUDIT.md): all 28 original entries checked;
32 final entries after four necessary additions. Corrected author identities,
RoboFAC versioned authorship, WoVR and published Chinchilla titles, and several
publication venues/years. Sources and publisher-access limits are recorded.

[Replacement extraction protocol](COMPARISON_EXTRACTION.md): comparison-level
source/version/locator/value/unit/phase/denominator and applicability, with
independent annotations. The blank template contains no invented extraction.
Bibliographic corrections do not validate the legacy Q/P/U/NA codes.

## Validation record

`python verify_artifacts.py` runs the integrated offline checks and compares
generated audit artifacts with recomputed results. Local validation uses an
isolated temporary CPython 3.13.7 runtime with NumPy and jsonschema. The original
Torch RNG-isolation test remains skipped because Torch is absent; its historical
server success is not reasserted as a newly executed check.

Earlier integrated run: **42 tests, 41 passed, one Torch-dependent skip**; all 28
archived evaluator artifacts and three comparison checks reconcile. Generated
JSON, Markdown and LaTeX table match fresh recomputation. The proposed evaluation
schedule is balanced, deterministic, and protected against accidental overwrite.
Independent JSON Schema parity is included in the trace tests.

The new offline mutation experiment and its three integrity tests are recorded
above. Separately, ten focused execution-analysis tests passed with constructed
records and outputs confined to temporary directories. Those tests cover declared
schedule completeness, paired hashes, instrumentation parity, seed grouping,
task-weighted versus pooled estimands, and null cost-per-success when there are
no successes. They do not contribute synthetic observations to empirical tables.

The preceding PDF build, before integration of the new mutation experiment, used
Tectonic 0.17.0 and PyMuPDF 1.28.2.
All nine rendered pages were visually inspected: **five main-content pages,
three reference pages, one appendix page**. No missing citations, replacement
characters, overflowing text or clipped tables were found. The PDF has no author
metadata; the anonymous block is generated by the unchanged official template.
The build retains benign fontconfig/lineno-source warnings and two underfull
table-cell spacing warnings; these do not appear as visible glyph/layout defects.

Preceding PDF SHA-256 (not the updated manuscript source):
`bf589c97e41176b89f6bc2efc9ed1d35d2f8a371cfff3f126ffd0b039251caf3`.
That PDF was copied to `paper.pdf` only after
rendering. The manuscript source now includes the offline mutation results; a
current 9-page PDF has since been rebuilt and visually inspected. Raw archived
result files are unchanged. No manuscript was submitted or branch pushed; this
audit does not yet report a new GPU experiment.
