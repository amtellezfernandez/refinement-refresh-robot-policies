# Replay repair qualification (5 November 2026)

## Qualification boundary for the current PR head

The local results below apply to the source set identified by the earlier
`SOURCE-MANIFEST.sha256`, before the runtime provenance edits now on this
branch. Commits after that qualification changed
`experiments/tooling/run_paired.py` and `replay/governance.py` to capture
imported module versions/paths, expose duplicate distribution records, hash the
LIBERO asset tree for future experiment manifests, and verify imported replay
modules against distribution metadata. **Those edits have not been tested or
requalified.** The earlier counts must not be presented as tests of the current
head. No CI workflow or run is being used for these changes.


PR #3 was fetched at `e8861f2b89ff57f1eece91646e5f2c33c5d157b8`. That head did not contain the three verified protections: complete ordered result-row accounting, independent repeat success/tick/hash comparison, and `ARCHIVE_DEFECT` classification for frozen-input defects.

The repaired source set is identified by `SOURCE-MANIFEST.sha256` (manifest SHA-256 `508049894b25e02065c8dae746c4e083daee50401a89ce0dbb9b64547a77e4e8`). All checks were rerun against a fresh copy matching all 21 manifest entries. The test processes ran under a macOS write-denial sandbox; write, chmod and rename probes were denied, and post-test hashes still match.

Results: integration controls 15/15; synthetic fixture checks 40/40; governance 20/20; safety 13/13; CPU stage drills 5/5; stage logic 4/4; focused repair regressions 6/6. The synthetic integrated preflight passed its controls and refused the archived sample-first path pending the stop-versus-perturbation protocol decision. The original frozen protocol SHA-256 remains `ede3eb8570487b7e2eda8e47f39c6afc1d12e40bece590e8c4590d34341ea737`; the separate sample-first protocol SHA-256 is `2c66efd5677c90e7a3307aa3ed7215a509341356a51a01f5e8b41cf2115805c0`.

No archived replay, sample reset, experimental attempt or real stage ran. Archived replay remains blocked by the unresolved sample-first stopping-rule clarification and Alba's separate stage-scoped approval. Synthetic qualification does not grant execution approval.
