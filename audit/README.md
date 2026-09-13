# Proposal 2 audit artifacts

The manuscript no longer uses the old 0/16 disclosure-completeness headline.
`pilot_audit.csv` is **provisional legacy coding**, retained without rewriting
historical evidence. Its `fully_reconstructable` column is an old checklist label,
not an established scientific reproducibility verdict.
Current scope: no second human annotator is available. Human agreement and kappa
are outside this submission. Prioritize [source verification](SOURCE_CHECKS.md),
comparison boundaries and sensitivity; do not substitute two model passes for
human reliability. The unused second-annotation preparation is retained:
[coordinator protocol](SECOND_ANNOTATION.md),
[blind coding guide](ANNOTATOR_GUIDE.md), and [blank 128-cell packet](blind_annotation.json).
`agreement.py` calculates pre-adjudication agreement only after two completed,
evidence-backed packets are supplied. No second annotation or kappa result has
been collected. Keep the existing labels and headline counts preliminary.

Audit status, 2026-09-10: the existing labels are preliminary. They have not been
independently recoded with comparison-specific boundaries and exact source
locators. "0/16" means none satisfies this draft's maximal checklist; it does
not establish that the original scientific comparisons are irreproducible.
Dropping storage changes this to 1/16; dropping storage, human labor, and wall
time gives 2/16; considering training and deployment alone gives 3/16, using the
unchanged labels. See `../EXPERIMENT_AUDIT.md` for remaining validity issues.

The CSV lacks per-field source versions, passages, units, phases and comparison
boundaries. Claims that every resource label was source-verified or that missing
fields invalidate the original comparisons are not supported by this artifact.
No independent annotation was performed during this revision.

- [Revision audit](REVISION_AUDIT.md): flaws, corrections, and remaining evidence.
- [Bibliography audit](BIBLIOGRAPHY_AUDIT.md): verified bibliographic identities,
  metadata corrections, version choices, and access limitations.
- [Comparison extraction protocol](COMPARISON_EXTRACTION.md) and
  [blank CSV](comparison_extraction_template.csv): replacement measurement unit.
- [Archive audit](../experiments/real_smolvla/ARCHIVE_AUDIT.md): executed offline
  checks on actual experiment records.

`summarize_audit.py` checks the consistency of the old labels and reports their
boundary sensitivity. Its numerical output is a calculation on those labels,
not a new source-verified literature result. Historical Q/P/U/NA codes mean
quantitative/proxy/unknown/not-applicable under the old coding; applicability
itself must be re-extracted before using them as evidence.

Status codes remain `Q` (quantitative), `P` (proxy), `U` (applicable but
unreported), and `NA` (not applicable). The strict `fully_reconstructable` label
requires every applicable cost for the named comparison and is retained only as a
provisional legacy calculation. Source lists remain in `SOURCE_CHECKS.md`.
