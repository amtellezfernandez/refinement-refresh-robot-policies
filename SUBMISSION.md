# Submission and working-version notes

Title: **Refinement versus Refresh: Compute Allocation for Closed-Loop Robot
Policies**.

## Version policy

- [release/submitted.pdf](release/submitted.pdf) is the frozen submitted manuscript.
- The dated [submission archive](release/archive/2026-09-13/) retains
  another copy and the submission materials.
- [paper/main.tex](paper/main.tex) and [paper/working.pdf](paper/working.pdf) are the continuing
  working version. Post-submission results belong here.

Both submitted PDF copies must retain SHA-256:

```text
9a23d4e8b603f20b2b0aed45ab68f55300a975d979d9814259ff5679499c19f5
```

The working manuscript uses the existing anonymous NeurIPS 2026 workshop style
with the RoboPAD workshop title. It now includes the full 1,120-attempt
confirmation study, in addition to 480 discovery attempts. This revision is not
a new submission and does not change the submitted PDF.

## Next target: ICLR 2027

The main-conference development plan is in [ROADMAP.md](ROADMAP.md).
Official requirements checked on 13 September 2026:

- Abstract: 18 September 2026, 23:59 AoE (19 September, 13:59 CEST in Paris).
  Paper: 25 September, 23:59 AoE (26 September, 13:59 CEST).
  Internal cutoffs: 18 and 25 September at 18:00 CEST.
  [Call for papers](https://iclr.cc/Conferences/2027/CallForPapers).
- Convert the active manuscript to the official ICLR 2027 template; at most
  nine main-text pages, excluding references and appendices. Preserve the
  submitted workshop version. The current PDF is not yet ICLR-formatted.
  [Author guidelines and template](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- Authors must settle the author list, verify OpenReview profiles, register a
  genuine abstract, and check reciprocal-reviewer eligibility or exemption.
  Do not assume these steps are complete. Additions to the author list are not
  permitted after the abstract deadline.
  [Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- Include an accurate AI-use section and matching submission-form disclosure,
  covering conceptual work, hypotheses, design, interpretation, implementation,
  and writing/figures where applicable. Authors must review the account and
  describe only verification actually performed; automated checks are not
  evidence of independent human review. The section is outside the page limit.
  [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).
- Audit anonymity in the manuscript and any released code or supplementary
  archive. Keep the frozen primary-only supplement unchanged; prepare any new
  full-evidence artifact separately. Include a reproducibility statement.
- Do not submit substantially the same work concurrently to ICLR and ICRA.
  ICLR permits relevant prior non-proceedings workshop presentations; RoboPAD
  describes itself as non-archival. This does not certify submission status.
  [ICLR policy](https://iclr.cc/Conferences/2027/AuthorGuidelines),
  [RoboPAD policy](https://robotpad2026.github.io/call-for-papers.html).

ICRA 2027 remains an alternative requiring an explicit venue switch: its
15 September 2026, 23:59 deadline is labeled PST, and its two-column eight-page
limit includes references. It is not a later fallback after ICLR.
[ICRA call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).

## Build and review

```bash
python3 experiments/tooling/analyze_confirmation.py --check
python3 experiments/tooling/render_paper_visuals.py
python3 experiments/tooling/render_paper_visuals.py --check
latexmk -pdf -outdir=build paper/main.tex
# Render and review build/main.pdf before copying:
cp build/main.pdf paper/working.pdf
sha256sum release/submitted.pdf release/archive/2026-09-13/submitted.pdf
```

The reviewed working build has seven main-content pages, one reference page,
and six appendix pages. The final PDF and evidence checks are recorded in
[review/REVISION_AUDIT.md](review/REVISION_AUDIT.md).
The frozen primary-only supplement remains in
[output/primary_transfer_supplement.zip](output/primary_transfer_supplement.zip);
the new full-confirmation evidence is indexed in
[experiments/runs/README.md](experiments/runs/README.md).

## Claim boundaries

The paper studies one frozen checkpoint in simulation. The primary comparison
supports lower denoising at a fixed two-action prefix; it does not establish a
universal refinement rule, a learned-selector advantage, or equivalent quality
for the cheaper longer prefix. Secondary intervals are descriptive,
task-conditional, and unadjusted. The stale-input diagnostic reports 20/40
successes with five-tick held-input refresh versus 28/40 matched fresh inputs,
with descriptive paired difference -20.0 points and interval [-32.5, -7.5].
Its journals lack per-tick supplied-input hashes and observation ages, so it does
not establish a complete causal decomposition. Aggregate transfer timings are
not controlled latency evidence.

The new confirmation audit passes. The legacy integrated verifier still reports
a frozen runner-hash mismatch in instrument validation; no frozen protocol
hashes were changed. Future work is tracked in
[ROADMAP.md](ROADMAP.md).
