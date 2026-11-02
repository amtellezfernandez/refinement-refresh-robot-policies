# Decision register, 2 November 2026

Source: Alba's written decisions, as relayed by Warren on 2 November. Her message itself was not found in the mailbox searched, so this register rests on that relay. If her text has more detail, for example on any item marked OPEN, it overrides this file.

## Alba's decisions and where each is implemented

| # | Decision | Implementation |
|---|---|---|
| 1 | Permanently exclude the earlier invalid pi0.5 160-attempt campaign | `exclusions` in both new protocols |
| 2 | SmolVLA: ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING | `smolvla_historical_label` in the matched protocol |
| 3 | Submitted NeurIPS RoboPAD PDF unchanged | `release/submitted.pdf` untouched |
| 4 | Stale 20/40 vs 28/40 and its contribution claim excluded from future versions unless original journals are recovered and audited | `exclusions`. `paper/main.tex` still contains it and was not edited; it is Alba's manuscript |
| 5 | 1 November preflight stays PREFLIGHT_FAILED | `prior_preflight`; raw evidence untouched |
| 6 | New versioned preflight and unique run ID | `cross_policy_preflight_v2`, `cross_policy_preflight_v2.run001` |
| 7 | All Alba's checks for both models | `checks` in the protocol; `preflight_verdict.py` |
| 8 | Only permitted confirmation design: matched R22 vs R25 | `matched_r22_r25_confirmation_v1`; other cohorts refused by `launch_guard.py`. **See the count conflict** |
| 9 | Append-only journals; failed attempts kept | `journal.py`, `attempt_loop.py` |
| 10 | No confirmation until Alba has the preflight commit, verdict, manifest and checksums | Guard requires her release; `launch_condition` |
| 11 | R52, R55, observation age, asynchronous, third policy deferred beyond mandatory smoke checks | Refused by guard; R52 and R55 only in the `conditions` smoke check |

## Attempt-count conflict: Alba must rule

Decision 8 lists 2 models x 4 suites x 10 tasks x 5 states x 2 seeds x 2 conditions, and "800 total attempts".

**Those factors multiply to 1,600. 800 is the count per model.**

The handover contributed to this: `07_REMAINING_WORK_AND_ESTIMATES.md` labelled R22 vs R25 on both models as "800 attempts", while its 6-28 hour estimate had counted 800 for each model. So the figure may come from that error. It has been corrected in the repository.

To reach 800 in total, some factor must change, and which one is a scientific choice. Schedules follow the stated factors, so 1,600 rows. `matched_r22_r25_confirmation_v1` is `BLOCKED_ATTEMPT_COUNT_CONFLICT`, and the guard refuses it until this is settled. This does not block the preflight.

## Status of every open question

| ID | Question | Status after Alba's decisions | Blocks |
|---|---|---|---|
| R1 | Asynchronous runtime | DEFERRED (decision 11) | Nothing now |
| R2 | Observation-age seeds | DEFERRED | Nothing now |
| R3 | Bootstrap undercoverage | OPEN. R22 vs R25 still has an interval; the -5 point test is not in this design | Confirmation release |
| R4 | Noise and `num_steps` wrapper as official path | OPEN. Decision 7 asks for official normalization and receipt of d, and does not rule on the wrapper | **Preflight** |
| R5 | No CUDA-event time for pi0.5 | OPEN. Recorded as unavailable | Confirmation release |
| R6 | OSMesa or EGL | OPEN | **Preflight** |
| R7 | Newly generated or unused official states | OPEN. Check list names "generated-state hashes", which suggests generated; not read as a ruling | **Preflight** |
| R8 | Identity overlap | Decision 1 excludes the campaign. Its records are still unavailable, so overlap stays uncertified | Confirmation release |
| R9 | Native-parity criterion and action-unit acceptance | OPEN. The check is required with no criterion given | **Preflight** |
| R10 | How failed attempts enter tests | PARTLY. Decision 9 preserves them; entry into tests is not stated | Confirmation release |
| R12 | pi0.5 NATIVE h=5, d=10 | OPEN | **Preflight** |
| R13 | Arithmetic across frameworks | OPEN | Confirmation release |
| R16 | First action divergence | DEFERRED | Nothing now |
| N1 | State stack across MuJoCo 3.8.1 and 3.2.3 | OPEN; facts below | **Preflight** |
| N2 | LIBERO source for the client | OPEN; facts below | **Preflight** |
| N3 | R22 and R25 first, or all five | RESOLVED by decision 8 | |
| N4 | Independent outcome verification | OPEN. Not in the decisions | Confirmation release |
| N5 | Stop on first failed attempt | OPEN. Compatible with decision 9, not confirmed | Confirmation release |
| N6 | Release format | OPEN; proposed format is implemented | Preflight release |
| New | 800 or 1,600 attempts | OPEN | Confirmation release |
| New | Fresh environment per pi0.5 attempt | Technical; implemented because exact reset reuse requires it. Reported to Alba | |
| — | Why the pi0.5 campaign was invalid | Decision 2 keeps validity pending; reason still unknown | SmolVLA validity |

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
