# Submission and public-version notes

Title: **Refinement vs. Replanning: Inference Compute Allocation in Robot Policies**.

## Version policy

- [release/submitted.pdf](release/submitted.pdf) is the frozen submitted manuscript.
- The dated [submission archive](release/archive/2026-09-13/) retains
  another copy and the submission materials.
- [output/pdf/camera-ready-submission-2026-11-02.pdf](output/pdf/camera-ready-submission-2026-11-02.pdf)
  is the optional de-anonymized accepted-paper edition; its source is
  [paper/camera-ready-2026-11-02.tex](paper/camera-ready-2026-11-02.tex).
- [paper/main.tex](paper/main.tex) and the dated accepted-paper source are
  synchronized. A future anonymous archival submission must be prepared as a
  separate release rather than by changing the frozen submitted manuscript.

Both submitted PDF copies must retain SHA-256:

```text
9a23d4e8b603f20b2b0aed45ab68f55300a975d979d9814259ff5679499c19f5
```

The accepted-paper edition uses the final NeurIPS 2026 workshop style with the
workshop title and author information. It includes the full 1,120-attempt
confirmation study, in addition to 480 discovery attempts. It is maintained
separately and does not change the frozen submitted PDF.

Current accepted-paper artifact digests:

```text
PDF:    b97811bd68ca78fbffa4cef37fe42f856e2e525f4647d0b49e72aa218f1794b9
Source: 5e51e2c53b82edb36eedb70ae73198c196a70a702b7bcb312e1c0bb0ba675fb2
```

Compliance was checked on 11 November 2026 against the workshop
[call for papers](https://robotpad2026.github.io/call-for-papers.html). The
accepted-paper edition has eight main-content pages, within the nine-page cap;
references begin on page 9 and the appendix on page 10, both outside that cap.
It uses the official workshop style. The workshop does not require the
main-conference paper checklist or a formal camera-ready submission. This
de-anonymized edition is maintained as an optional updated public version.

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
  ICLR permits relevant prior non-proceedings workshop presentations; this workshop
  describes itself as non-archival. This does not certify submission status.
  [ICLR policy](https://iclr.cc/Conferences/2027/AuthorGuidelines),
  [workshop policy](https://robotpad2026.github.io/call-for-papers.html).

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

# Build the accepted-paper edition from the paper directory:
cd paper
latexmk -pdf -outdir=../build/camera-ready camera-ready-2026-11-02.tex
cd ..
cp build/camera-ready/camera-ready-2026-11-02.pdf \
  output/pdf/camera-ready-submission-2026-11-02.pdf

sha256sum release/submitted.pdf release/archive/2026-09-13/submitted.pdf
sha256sum output/pdf/camera-ready-submission-2026-11-02.pdf \
  paper/camera-ready-2026-11-02.tex
```

The reviewed accepted-paper build has eight main-content pages, one reference
page, and six appendix pages. The final PDF and evidence checks are recorded in
[review/REVISION_AUDIT.md](review/REVISION_AUDIT.md).
The frozen primary-only supplement remains in
[release/primary_transfer_supplement.zip](release/primary_transfer_supplement.zip);
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
