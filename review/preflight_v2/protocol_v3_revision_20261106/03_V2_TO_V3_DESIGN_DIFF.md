# v2 → v3 (revision 2) design diff

The baseline is v2: `cross_policy_preflight_v2.json` `4fc520ec...`, `PREFLIGHT_V2_SPEC.md`, and the confirmation design it gates, `cross_policy_confirmation_v1.json` `9310639c...` with Alba's PR #1 primary-only derivation. The intermediate step is v3 revision 1, `diagnostic_recovery_20261105/PROTOCOL_V3_DRAFT.md`. This diff includes Alba's eight decisions of 6 November (D1–D8, `12_DECISION_REGISTER_20261106.md`). No frozen file is edited.

## Structure

| Aspect | v2 / confirmation_v1 | v3 rev 2 | Reason |
|---|---|---|---|
| Protocols | one preflight gating a five-condition confirmation, primary subset derived | **two separate protocols**: A (engineering preflight) and B (`cross_policy_confirmation_v2`, not authorised) | Alba, Part 1 |
| Preflight scope | mechanics plus native-parity episodes plus condition smoke | mechanics only: synthetic fixed actions, fresh envs, diagnostic observations; no outcome values | Part 1A, Part 8 |

## Removed

| Removed item | Was in | Note |
|---|---|---|
| Five-condition design (NATIVE, R22, R25, R52, R55) | confirmation_v1 `conditions` | Protocol B has R22 and R25 only |
| 2,000 attempts per model, 4,000 total | confirmation_v1 `expected_attempts` | replaced by 800 per model, 1,600 total |
| "800 attempts across both models" | Alba's 2 Nov email wording; `cross_policy_preflight_v2.json` `alba_decisions_2026_11_02.attempt_count_conflict` | already corrected in PR #1; now removed from current text |
| R52 − R22 secondary contrast; −5-point quality-preservation margin | confirmation_v1 | no h=5 outcome claims |
| R22 = R52 and R25 = R55 first-action identities; R52/R55 smoke | v2 checks and negative control 3 | no R52/R55 anywhere |
| Native parity as 4 × 10 × 10 episodes per model, reading success | v2 check "native evaluator parity" | replaced by R9 action-level bitwise parity on DO1–DO3 (π0.5) and by Protocol E, a small model-free evaluator check for both stacks (D3) |
| Per-task video contact-sheet inspection of parity episodes | v2 | no parity episodes |
| Eager `count_denoise_loop` for pi0.5 as the primary count | v2 | compiled-path CNT (G02 design) primary; eager count kept as the CNT-IND cross-check |
| Hierarchical state-slot percentile bootstrap as the primary interval | confirmation_v1 `primary_analysis` | undercovered (R3). Replaced by the exact cohort contrast with worst-case missingness bounds (D1, D2); a secondary fresh-state interval is reported only if the R3 coverage check passes |
| Task-clustered t₃₉ interval; McNemar test; Tango and bootstrap sensitivity analyses | v3 rev 2 as first published (PROPOSED) | superseded by D1 before release; never used |
| Optional per-request count attestation; attestation mismatch as INFRA_POST | v3 rev 2 as first published | superseded by D5: required; mismatch is an instrument incident that stops the run |
| Single shared generated state set | v1 R7 rule; `generate_states.py` keyed by cohort | two model-native populations (N1) |

## Added

| Added item | Where |
|---|---|
| Delta_m estimand; exact cohort contrast and four paired counts; worst-case bounds over 400; secondary fresh-state interval with R3 coverage check (D1, D2, D6) | `02` B2, B6 |
| INSTRUMENT_INCIDENT attempt class; SmolVLA non-finite action under valid inputs as SCI_FAILURE (D5, D7) | `02` B5 |
| Protocol E: model-free evaluator validation for both stacks (D3) | `13` |
| Per-model qualification record; single planned host RTX 5090 WSL OSMesa (D4) | `01` A3a; `07` |
| Per-request executed-step attestation with request and tick binding (D5) | `09` C4 |
| Preprocessing comparison; matched noise; stop-if-infeasible rule (D8) | `05` P2, P2a, P6 |
| SmolVLA qualification items S-QUAL (D7) | `01` A5 |
| Decision register for D1–D8 | `12` |
| Attempt classes SCI / INFRA_PRE / INFRA_POST / PROTOCOL_STOP; retry and missingness rules; worst-case bounds | `02` B5–B7 |
| Balanced keyed condition order; adjacent pairs | `02` B8 |
| Independent stock-native reference (no wrapper, separate process); bitwise d=10 parity on frozen observations; h=5 used only for the chunk-use check | `05` |
| Full action-dimension, normalization, clipping, frame and unit specification; commanded ≠ measured | `05` P3 |
| Cross-host tolerance procedure (specified, not run) | `05` P4 |
| R4 approval-evidence list; `KeyedPolicy` = candidate | `05` P5 |
| Per-model environment register; renderer kinds distinguished; 15 gaps | `07` |
| CNT-IND (independent count), independent step-size check, CNT-T timing, fire drills, review requirements | `09` |
| Minimal diagnostic population (SC1, SC2, DO1–DO3, mechanics fixtures) | `06` |
| Revised earlier-campaign interpretation | `08` |

## Unchanged

- The SmolVLA path.
- The checkpoints and their hashes.
- OpenPI `215abfb`.
- The fresh env per attempt.
- The policy seeds 810001 and 810002.
- The identity factorisation 4 × 10 × 5 × 2.
- The exclusions: the invalid campaign, the stale 20/40 vs 28/40 result, and the submitted PDF.
- The 1 Nov PREFLIGHT_FAILED verdict.
- v2 PREFLIGHT_INCOMPLETE.
- The hypothesis (the d=2 vs d=5 at h=2 contrast).

## Tooling

**Written as code in the separate implementation draft PR, never run** (`experiments/tooling/cross_policy_v3/`):
- the Protocol B schedule generator emitting 800 rows directly;
- the Protocol B runner, attempt classifier and release check;
- the π0.5 attesting server (startup G02 gate, per-request attestation) and the client-side verifier;
- the Protocol E evaluator-validation harness;
- the analysis module implementing B6, and the R3 coverage check;
- machine-readable qualification records, all items UNVERIFIED.

**Still not written:**
- `run_preflight_v3.sh`;
- `parity_stock_reference.py` and the P2a preprocessing comparison script;
- `make_diagnostic_population.py`;
- a model-keyed `generate_states.py`;
- Protocol E fixtures.

`make_primary_schedule.py` and `analyze.py` encode the old design: derivation from a 2,000-row source, and a bootstrap primary.
