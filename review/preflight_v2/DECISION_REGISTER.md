# Decision register, 2 November 2026

**Alba's latest email, relayed by Warren on 2 November, overrides what follows where they differ.** Current position: `run001_cpu/PREFLIGHT_V2_STATUS.md`.

| Item | Her email | Status |
|---|---|---|
| Invalid campaign | Exclusively the earlier pi0.5 campaign; official normalization could not be established and requested denoising steps were not reliably verified. "Exclude that campaign permanently" | RESOLVED |
| SmolVLA | ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING; the pi0.5 failure does not by itself invalidate it; shared evaluator and journaling need independent validation | RESOLVED as a label; validation still owed |
| Submitted PDF | Frozen; later work is post-submission | RESOLVED |
| Stale result | Excluded unless original journals are recovered and audited | RESOLVED |
| Preflight | 1 November stays PREFLIGHT_FAILED; new versioned protocol and run ID, run from the beginning | Protocol exists; run incomplete |
| First confirmation | R22 vs R25 only, after a passed preflight. **"Total: 800 attempts across both models"** | **CONFLICT: her factors and PR #1 give 1,600. Reopened; frozen schedules unchanged** |
| Asynchronous and others | Deferred | RESOLVED |
| R9, R6, N1, R4 | **Not stated in her email** | OPEN, listed in the status document |
| Push | Harness and specification to `warren/cross-policy-confirmation` | Warren's action |

---

Sources: Alba's PR #1 (`review/POST_SUBMISSION_CROSS_POLICY_REPAIRS_20261102.md`, commit `d8d9421`, her GitHub account), which is the text we hold in her own words; and her decisions as relayed by Warren. The relayed message itself was not found in Gmail, local Mail, local documents or other transcripts. **Where they differ, the PR overrides.**

## Current status

| Item | Her wording | Status |
|---|---|---|
| Attempt count | "400 identities x 2 conditions = 800 planned attempts; total across both models is 1,600, correcting the 800-across-both wording in Alba's email" | RESOLVED |
| R7 | "requires freshly generated state slots, rather than an unapproved substitution of official slots" | RESOLVED |
| R3 | "Preregister a calibrated method with a synthetic coverage study while the primary outcomes remain unseen" | Direction RESOLVED; method not implemented, OPEN |
| R5, pi0.5 XLA arithmetic | "explicit measurement gaps, not numbers inferred from host time" | RESOLVED |
| R6 | EGL "if an equivalent reference stack is required, or explicitly preregister OSMesa as a separate evaluation environment" | **OPEN, blocks preflight** |
| R9 | "preregister native-parity acceptance criteria against the reference evaluator before looking at outcomes" | **OPEN, blocks preflight**: form given, criterion not |
| R4 | "The response echo alone is insufficient" | **OPEN, blocks preflight**: whether the wrapper is official. Execution evidence is engineering |
| N1 | Not addressed | **OPEN, blocks preflight** |
| N2 | Pin the client's LIBERO source | Engineering; done on the reconciliation branch (`N2_PROBE_20261102.md`) |
| N6 | Confirmation release format given | Implemented from her PR. Preflight release format is our proposal |
| R12 | Not addressed | Engineering default from pinned source, unless she objects |
| R10, N4, N5, R8, R13 for SmolVLA | Not addressed | OPEN, before confirmation only |
| Invalidation reason | "must stay conditional" | **OPEN, unknown**; SmolVLA stays ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING |
| Frozen script | "Do not patch the frozen `run_gpu_preflight.sh` retrospectively" | Restored on the reconciliation branch; see `RECONCILIATION_20261102.md` |

## Alba's decisions and where each is implemented

| # | Decision | Implementation |
|---|---|---|
| 1 | Permanently exclude the earlier invalid pi0.5 160-attempt campaign | `exclusions` in the preflight protocol |
| 2 | SmolVLA: ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING | Her PR's label; recorded in this register |
| 3 | Submitted NeurIPS RoboPAD PDF unchanged | `release/submitted.pdf` untouched |
| 4 | Stale 20/40 vs 28/40 and its contribution claim excluded from future versions unless original journals are recovered and audited | `exclusions`. `paper/main.tex` still contains it and was not edited; it is Alba's manuscript |
| 5 | 1 November preflight stays PREFLIGHT_FAILED | `prior_preflight`; raw evidence untouched |
| 6 | New versioned preflight and unique run ID | `cross_policy_preflight_v2`, `cross_policy_preflight_v2.run001` |
| 7 | All Alba's checks for both models | `checks` in the protocol; `preflight_verdict.py` |
| 8 | Only permitted confirmation design: matched R22 vs R25 | Her `make_primary_schedule.py` and `PRIMARY_R22_R25_ONLY` release; 1,600 attempts per her PR |
| 9 | Append-only journals; failed attempts kept | `journal.py`, `attempt_loop.py` |
| 10 | No confirmation until Alba has the preflight commit, verdict, manifest and checksums | Guard requires her release; `launch_condition` |
| 11 | R52, R55, observation age, asynchronous, third policy deferred beyond mandatory smoke checks | Refused by guard; R52 and R55 only in the `conditions` smoke check |

## Measured facts for Alba's rulings

From CPU-only, simulator-only jobs on the RTX on 2 November (`alba-v2-probe-*`). No policy ran, and the GPU was not used.

| Finding | Evidence |
|---|---|
| **N2:** all 585 OpenPI asset files are byte-identical in the tree the client uses. That tree has 1,176 further files | `alba-v2-probe-3` |
| **N1:** a state generated in the SmolVLA stack restores exactly in the client stack (`libero_object` 1, `libero_10` 3) | `alba-v2-probe-3`, one state per task |
| **N1, not established:** whether such states settle the same way physically across stacks. The two stacks' hashes use different formats and cannot be compared | |
| Reused pi0.5 client environment: `libero_object` settled state varies by about 1e-14 between resets, for official and generated states | `alba-v2-probe-4`, 6 resets on two tasks |
| Fresh pi0.5 client environment: identical in 5 of 5 builds on 6 combinations across all four suites; about 0.8 s per build | `alba-v2-probe-5` |
| SmolVLA stack: identical over 6 resets on 4 combinations, with hard reset | `alba-v2-probe-4` |
| The generator's seed does not reproduce `libero_10` task 3, so the saved state and hash are the identity, not the seed | `alba-v2-probe-3` |
| The three model environments have no pip module | `alba-v2-probe-env2` |

**Limit:** these cover the tasks named, not all 40.
