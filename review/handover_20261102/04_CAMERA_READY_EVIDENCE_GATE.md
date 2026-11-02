# Camera-ready evidence gate, 2 November 2026

Deadline: 11:59 UK, 12:59 Paris. The manuscript and PDFs were not changed. No camera-ready PDF was available, so this covers `release/submitted.pdf` and `paper/main.tex` with `paper/working.pdf`. Citations, figures, selector fitting and timing tables were not audited.

Classes: **A** supported; **B** conditional or requires qualification; **C** exclude. Class A means supported on the records for what it states. No SmolVLA result has VALIDATED status.

| # | Claim | Class | Basis and required action |
|---|---|---|---|
| 1 | SmolVLA 139/160 vs 122/160, +10.6, [+4.4, +16.9], 24 wins, 7 losses | **B** | ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING ALBA CONFIRMATION. Keep "descriptive" and "task-conditional". Alba decides after the three confirmations in `03_160_ATTEMPT_PROVENANCE_AUDIT.md` |
| 2 | Earlier invalid pi0.5 campaign | **C** | No figure from it may appear. The two 160s must not be conflated |
| 3 | Any pi0.5 result or second-policy claim | **C** | No attempt exists |
| 4a | Existing SmolVLA (2,2) vs (2,5) | **B** | Same as item 1 |
| 4b | New matched R22 vs R25 across models | **C** | NOT STARTED |
| 5 | -5 point quality-preservation criterion | **C** | Belongs to the new protocol; no data. The paper correctly says no margin was prespecified |
| 6 | Asynchronous execution claims | **C** | NOT STARTED; blocked |
| 7a | Paper's RTX 4060 Ti timing tables | **B** | Keep existing monitoring-gap wording; not audited here |
| 7b | Any physical or real-time latency claim | **C** | No evidence |
| 8 | November smoke tests and RTX 5090 timings | **C** | Engineering only: 3 episodes per model on one task. PREFLIGHT FAILED |
| 9 | Stale-input 20/40 vs 28/40, [-32.5, -7.5] | **C** | Provenance insufficient. The 28/40 matches discovery (1,5). No raw record of the 20/40 arm is available. This does not show the number is false. May return only if the journals are recovered and verified |
| 10 | "This supports an observation-freshness contribution" | **C** | Depends on item 9 |
| 11 | "An outcome-blind, deadline-driven scheduling amendment" | **B** | Reword, below. The amendment records `outcomes_used_for_scheduling: false`, but 114 attempts per primary setting were already journaled and cannot show that outcomes went unseen |
| 12 | Any incomplete preflight or confirmation result | **C** | Incomplete work stays out |
| 13 | Other frozen counts: 138 vs 133 at h=5; 137 vs 136 at h=1; native 135 | **B** | Counts reproduced. Same condition as item 1 |
| 14 | 1,120 attempts; 7 settings, 8 tasks, 10 states, 2 seeds; 2 repeats excluded | **A** | Recounted. This states what the records contain |
| 15 | 171,805 actions; 50,451 primary | **A** | Recounted |
| 16 | Discovery 25, 28, 26 of 40; at h=5 21, 18, 16, 14 | **B** | Counts reproduced; shares the LeRobot instrument, so same condition as item 1 |
| 17 | Arithmetic: 5.9% per step; 41.1, 48.7, 16.1, 83.3 TFLOPs per attempt | **B** | Reproduced from step counts with the paper's rounded F(d). Keep "reference-calibrated". Not latency |
| 18 | 82.1% denoising-independent; affine F(d) | **B** | Keep "implementation-specific". Not re-derived from raw calibration |
| 19 | (5,2) 138/160 with 60.7% less arithmetic; prefix contrasts | **B** | Keep "unresolved" and "descriptive". Never "equivalent" or "non-inferior" |
| 20 | "Frozen before transfer outcomes" | **B** | Rests on host file times and records. GitHub witnesses only from 13 September |
| 21 | Journals "reconcile independently" | **B** | Internal consistency only; the paper already says shared upstream errors are out of scope |

## Recommended edits for Alba to consider

**Line references are to `paper/main.tex`. None has been applied.**

1. **Stale-input result (lines 423-432, 684-688; not in `submitted.pdf`).** Remove the result and its interval. If a sentence is wanted, this one states only what is planned: "A stale-input diagnostic with complete supplied-input instrumentation is planned." Keep the reference to completing the instrumentation only if accurate on her records.
2. **Observation-freshness (line 428).** Remove "This supports an observation-freshness contribution".
3. **Amendment (line 725; also in `submitted.pdf`).** Replace "An outcome-blind, deadline-driven scheduling amendment changed execution order" with: "A deadline-driven scheduling amendment changed execution order. Its record states that outcomes were not used for scheduling; 114 attempts per primary setting had been journaled when it was written."
4. **Figures 139/160 and 122/160 (lines 37-38, 79-80, 327-328 and tables).** No change proposed. Their use depends on Alba's confirmation.

If the camera-ready base is `submitted.pdf`, items 1 and 2 do not apply. Whether post-submission results are allowed in camera-ready was not checked.
