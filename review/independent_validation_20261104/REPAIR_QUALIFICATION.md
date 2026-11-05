# Replay repair qualification (5 November 2026)

PR #3 was fetched at `e8861f2b89ff57f1eece91646e5f2c33c5d157b8`. That head did not contain the three verified protections: complete ordered result-row accounting, independent repeat success/tick/hash comparison, and `ARCHIVE_DEFECT` classification for frozen-input defects.

The repaired source set is identified by `SOURCE-MANIFEST.sha256` (manifest SHA-256 `508049894b25e02065c8dae746c4e083daee50401a89ce0dbb9b64547a77e4e8`). All checks were rerun against a fresh copy matching all 21 manifest entries. The test processes ran under a macOS write-denial sandbox; write, chmod and rename probes were denied, and post-test hashes still match.

Results: integration controls 15/15; synthetic fixture checks 40/40; governance 20/20; safety 13/13; CPU stage drills 5/5; stage logic 4/4; focused repair regressions 6/6. The synthetic integrated preflight passed its controls and refused the archived sample-first path pending the stop-versus-perturbation protocol decision. The original frozen protocol SHA-256 remains `ede3eb8570487b7e2eda8e47f39c6afc1d12e40bece590e8c4590d34341ea737`; the separate sample-first protocol SHA-256 is `2c66efd5677c90e7a3307aa3ed7215a509341356a51a01f5e8b41cf2115805c0`.

No archived replay, sample reset, experimental attempt or real stage ran. Archived replay remains blocked by the unresolved sample-first stopping-rule clarification and Alba's separate stage-scoped approval. Synthetic qualification does not grant execution approval.
