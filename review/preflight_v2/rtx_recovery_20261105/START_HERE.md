# Preflight v2 handoff, 5 November 2026

**Purpose.** This continues your preflight v2 (`experiments/protocols/cross_policy_preflight_v2.json`, with R4, R6, R9 and N1 still open). It reports non-counting checks run on the RTX after it was repaired.

**Recommendation.** Do not launch run001 as frozen. Its pi0.5 step-count check cannot complete on this host, and `run_preflight_v2.sh` runs SmolVLA parity before it, so the counted run would spend parity episodes and then fail.

**Done.**
- SmolVLA real-weight profile at d=2, 5, 10: denoise steps = generations x d, 100 generations each. Parity not run.
- pi0.5 real-weight loaded normalization equals `norm_stats.json` exactly.
- pi0.5 frozen eager step count: d=5 fails at the default JAX cap, both after d=2 and alone. Live JAX memory rose about 4.6 GiB per language-model pass. After the d=2 call and garbage collection, 0.05 GiB was freed and about 16.1 GiB above the post-load level stayed allocated. The holder is unidentified.
- Also failed: at the full-card cap, and with the non-pooling allocator.
- EGL on this host is Mesa llvmpipe, software only.

**Limits.**
- Nothing here counts toward run001, or covers parity, action units, matched resets, smoke tests or videos.
- Physical VRAM exhaustion was not established.
- d=10 has never been counted with real weights.

**Requests, in priority order.**
1. The pi0.5 step-count method: find and release the retained memory; count on the compiled path with independent validation; or run elsewhere. Any change needs a new protocol version and run ID.
2. R9 (parity and action-unit acceptance criteria), R6 (renderer), R4 (server wrapper as official path), N1 (state-generation stack).
3. The G0 release bound to the protocol hash.

**Navigation.**
- [REPORT.md](REPORT.md): detailed results, memory tables, options and reproduction.
- [REVIEW.md](REVIEW.md): independent review outcome and corrections.
- [evidence/](evidence/): per-run commands, logs and outputs. The index is in [REPORT.md section 11](REPORT.md#11-evidence-index).
- `MANIFEST.sha256`: every file.
