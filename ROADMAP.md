# ICLR 2027 roadmap

[Working paper](paper.pdf) · [Frozen submission](submitted.pdf)

Continue in [paper/main.tex](paper/main.tex). Never overwrite either
`submitted.pdf` or `submissions/robopad_2026_20260913/submitted.pdf`.

Target: **ICLR 2027 main conference**. This is a research and submission plan,
not a claim of readiness or an authorization to launch new campaigns.
The working PDF still uses the RoboPAD/NeurIPS workshop template.
Venue requirements and author-side tasks are in [SUBMISSION.md](SUBMISSION.md).

## Scientific objective

Determine when additional prediction computation helps closed-loop control,
and when it is unnecessary or harmful. Preserve the allocation framework:
`b(h,d) = F_gen(d) / h`, with a measured denoising-independent component.
Do not claim that refresh universally beats refinement or that a fitted
selector adds value: the completed results establish neither.

The main strengthening priorities are **a second-policy evaluation and a
controlled observation-age diagnostic**. A new quality-preservation campaign
is conditional follow-up, not a prerequisite that delays those two priorities.

## Completed

- 480-attempt discovery study and 1,120-attempt confirmation study.
- Exact coverage of all seven confirmation configurations and 160 paired
  task/state/seed identities; two repeat records excluded.
- Full action-journal and runtime checks, original frozen comparisons, amended
  descriptive contrasts, and task/resource tables reproduced locally.
- Working manuscript revised around refinement versus refresh, including the
  unchanged primary result and the cheaper five-action setting.
- Current workshop flow: the allocation framework precedes the study design;
  frozen refinement tests and descriptive refresh comparisons are separate,
  with detailed selector and timing diagnostics moved to the appendices.
- Fitted-selector limitation retained: no demonstrated advantage over the simple
  cost-aware rule. Similar quality point estimates are not equivalence evidence.
- Publication visuals: two print-sized vector figures, all eleven tables
  standardized, and deterministic presentation assets generated from unchanged
  evidence reports. Frozen and amended comparisons remain visibly separate.

The primary result is 139/160 versus 122/160 successes for `(h,d)=(2,2)`
versus `(2,5)`. The cheaper `(5,2)` setting records 138/160, but its quality
difference remains unresolved. The other frozen denoising comparisons do not
establish a uniform benefit across prefixes. Existing confirmation coverage is
balanced: seven configurations, each on the same 160 identities. Do not
redistribute or relabel these completed attempts.

## Next research actions

1. **Second-policy feasibility gate, alongside the existing diagnostic audit.**
   Choose a runnable chunked iterative policy, preferably from a different
   family. Record checkpoint/source revisions, access and license, hardware
   fit, preprocessing, action units and positions, native prefix and solver,
   supported refinement settings, and parity with its reference evaluator.
   Measure full-generation cost at multiple settings and report fit residuals;
   do not assume SmolVLA's affine model or numerical denoising choices transfer.
   Keep hardware feasibility checks separate from outcome-based selection.
   No second checkpoint or compute allocation has yet been selected here.
2. **Audit the completed observation-age diagnostic and its instrumentation.**
   The remote run is now complete: 40/40 planned attempts, with 20/40 stale
   successes versus 28/40 matched fresh-input successes. The paired difference
   is -20.0 percentage points with a descriptive task-conditional 95% interval
   [-32.5, -7.5]. This is evidence for an observation-freshness contribution,
   not a complete causal attribution. The frozen design uses 40 discovery
   identities, generates every tick at `h=1,d=5`, and refreshes both images and
   proprioception every five ticks. Thus supplied observation age cycles through
   0--4 ticks; generation frequency and the first executed action position stay
   fixed. The task instruction is unchanged. Use the matched fresh `h=1,d=5`
   condition as the direct comparison; `h=5,d=5` is contextual, not an isolated
   freshness contrast. Audit training targets/masks and action-queue indexing.
   The completed historical journals still lack per-tick supplied-input hashes
   or observation ages. This has now been fixed in a separate runner snapshot
   and an isolated rerun is active on the remote host at
   `stale_instrumented_004`; its live journal has verified ages 0--4 and hash
   changes between refreshes. The prepared environment required a compatibility
   shim for the installed numba/coverage type API; installed packages and the
   frozen runtime were not modified. Do not pool this rerun with the historical
   result until all 40 attempts, pairing, hashes, and source manifests are
   audited. This does not by itself explain the denoising benefit.
3. **Freeze a targeted second-policy protocol before confirmation.**
   A compact candidate design is two refinement settings crossed with two
   prefixes, plus the native configuration if distinct (at most five settings).
   This supplies fixed-prefix refinement and fixed-refinement prefix contrasts
   and permits a declared interaction analysis; it is not a chosen run plan.
   Specify development versus evaluation identities, primary/secondary
   contrasts, pairing, task weighting, uncertainty, sample size, resource limits,
   failure handling, and stopping rules before collecting confirmation outcomes.
   Record which earlier results informed the design. Include all planned
   outcomes, including disagreement between policies; do not tune for `d=2` to
   win. Keep the new campaign separate from the completed 1,120 attempts.
4. **Use the accounting model to compare feasibility, not predict success.**
   Where the measurements support `F_gen(d) ≈ F0 + d F1`, report
   `rho = F0 / F_gen(d_native)` under a matched counting boundary. Larger `rho`
   leaves less fractional arithmetic to save by reducing refinement alone.
   It does not predict the quality-optimal prefix. The intercept is not a
   component-level measurement of freshness. A different policy preference is
   useful evidence about the framework's boundaries, not a failed replication
   to hide.
5. **Defer a new quality-preservation claim unless there is time to test it.**
   A new `(5,2)` versus `(2,2)` study needs an application-relevant margin,
   sample-size justification, new evaluation identities, and a frozen protocol.
   Do not choose a non-inferiority margin from the existing interval. Testing
   `d=1` likewise requires a separately declared experiment.

## Main-paper flow and space allocation

Proposed flow: **allocation question → measured compute floor → controlled
comparisons → cross-policy agreement or disagreement → observation sensitivity
→ scope and limits**. New-result sections remain conditional on audited data.

The current 14-page workshop PDF has seven main-content pages, one reference
page, and six appendix pages. Its confirmation section starts on page 5;
Figure 1(b) emphasizes host-time sensitivity, and a forced break isolates the
short discussion on page 7. For the ICLR rewrite:

- Put a measured cost-decomposition/feasibility visual early; move the
  host-time sensitivity panel to the measurement appendix. Do not label the
  entire intercept as the cost of observation acquisition.
- Keep protocol and freezing together, then report discovery and confirmation
  in chronological order. Retain the complete confirmation overview and all
  frozen/amended contrasts, with their analysis status visibly separated.
- Compress repeated selector discussion to its negative conclusion and an
  appendix reference. Group full confirmation tables together; keep runtime,
  provenance, and selector details in clearly separated appendix sections.
- Reserve main-text space for second-policy and mechanism findings, including
  inconclusive results. Reflow floats and the discussion in the official ICLR
  template; do not preserve workshop page breaks or shrink typography.

An initial nine-page budget is: motivation/related work 1.25; cost framework 1;
protocol 1; completed SmolVLA evidence 2; second policy 1.5; mechanism 1;
discussion/limits 0.75; layout buffer 0.5. This is an editorial allocation,
not permission to fill missing results with promises. The current abstract
and numerical evidence stay unchanged until there is something new to report.

## Deadline plan

Official deadlines: abstract **18 September 2026, 23:59 AoE**; paper
**25 September 2026, 23:59 AoE**. In Paris these are 19 and 26 September,
respectively, at 13:59 CEST. Use 18 and 25 September at 18:00 CEST as internal
submission cutoffs. [Official dates](https://iclr.cc/Conferences/2027/CallForPapers).

| Dates (September 2026) | Deliverable / decision |
| --- | --- |
| 13--15 | Preserve completed evidence; audit diagnostic status and instrumentation; establish second-policy feasibility |
| 16--18 | If feasible and authorized, collect development evidence and freeze targeted comparisons; authors register a genuine abstract and complete author list |
| 19--22 | Run the frozen second-policy confirmation if authorized; audit all outcomes and diagnostic evidence; narrow scope if evidence is incomplete |
| 23--25 | Freeze results; complete ICLR format, accurate disclosure, anonymous artifact, author review, and submission with a buffer |

Second-policy selection, resource budget, and the new protocol are unresolved
launch decisions. Author registration, authorship, and submission require the
authors' action; no conference submission has been made by this plan. ICRA is
an alternative venue decision, not a concurrent submission or a fallback after
the ICLR deadline. Do not conceal the one-policy limit if new evidence fails
to materialize.

## Reproducibility maintenance

The completed-confirmation analyzer and runner/analysis tests pass locally.
The older integrated verifier reports `Frozen input drift:
runner_sha256_lf_normalized` in the instrument-validation experiment.
Investigate against preserved source snapshots; do not replace frozen hashes
just to obtain a passing test.

```bash
python3 experiments/tooling/analyze_confirmation.py --check
python3 -m unittest discover -s experiments/tooling -p 'test_*.py'
python3 experiments/tooling/render_paper_visuals.py --check
latexmk -pdf -outdir=build paper/main.tex
```

Render and inspect the resulting PDF before copying it to `paper.pdf`.
Record substantive revisions and verification in
[the existing audit](review/REVISION_AUDIT.md), not another roadmap file.
