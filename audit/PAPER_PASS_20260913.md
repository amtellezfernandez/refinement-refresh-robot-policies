# Manuscript pass, 2026-09-13

Scope: the current allocation paper, including its compiled PDF. No experimental
configurations, selections, outcomes, or analysis code were changed.

## Corrections

- Defined executed prefix `h` and denoising steps `d` in the abstract; specified
  the ten-to-five comparison and per-attempt cost denominators.
- Defined `F_gen(d)` as reference-calibrated throughout the resource equations.
- Replaced the overbroad statement that the shape audit required no cost-model
  correction with its supported conclusion: eligibility is unchanged, while
  task-shape arithmetic is lower than reference arithmetic.
- Specified the 24-element selection candidate set explicitly.
- Explained the discovery tie between the primary comparator `(2,5)` and
  `(2,10)` and its lower-compute tie-break.
- Defined Table 2's normalized rate and clarified seed-preserving bootstrap
  resampling; retained every reported success and expenditure value.
- Corrected the verification hyperlink, which had inherited the language-shape
  subsection number after the appendix was compressed.
- Named the later-documented baselines and distinguished fixed policy execution
  settings from GPU-process monitoring, which did not cover the final segment.
- Distinguished the discovery bootstrap seed (20260913) from the transfer
  bootstrap seed (20260912).
- Scoped the conclusion to two versus five denoising steps at `h=2`.
- Compressed repeated resource-sensitivity prose and corrected excessive spacing
  in the checkpoint paragraph without changing fonts or page geometry.

## Evidence checked

- All 480 discovery episode identities are unique. Each of the 12 configurations
  has 40 attempts. Recomputed success counts and rollout means agree with the
  discovery text and tables.
- Table 2 and the abstract agree with `primary_deadline_analysis.json`: 139/160
  versus 122/160, paired difference 0.10625, interval [0.04375, 0.16875], and the
  reported arithmetic and rollout means.
- Calibration values, frozen predictions, candidate configurations, model scores,
  and selector choices agree with their retained JSON records.
- The primary analysis script multiplies generation counts by reference
  calibration and uses a task-conditional, seed-preserving bootstrap. This pass
  checked the existing primary analysis and summary; it did not rerun the full
  remote primary journal audit.
- All eight cited bibliography keys resolve. This was a manuscript consistency
  check, not a fresh literature search or verification of every external source.

## Outstanding submission item

The previous availability statement claimed an accompanying anonymous
supplementary artifact. No such packaged artifact or anonymous access URL was
found in the workspace. That unsupported claim was removed; the LLM-assistance
disclosure remains. Retained local and remote research records are not evidence
that a supplement has been uploaded. Packaging, anonymization, upload, and the
corresponding access statement still need verification before claiming access.

The README and SUBMISSION notes also contain historical status descriptions;
they were not treated as evidence of the current result or submission status.

## Final PDF checks

The revised PDF has nine pages: six main pages, one references page, and two
appendix pages. All pages were rendered and visually inspected, with changed
pages inspected again after the final edit. No clipping, overlapping text,
overfull or underfull boxes, or unresolved citations/references were found.
All fonts are embedded and none is Type 3. The verification hyperlink resolves
to its paragraph on page 9. The remaining build warning is the existing
relative package-path/name mismatch for the shared NeurIPS style file.
