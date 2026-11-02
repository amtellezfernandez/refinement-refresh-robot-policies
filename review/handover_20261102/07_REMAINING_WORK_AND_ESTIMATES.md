# Remaining work and estimates, 2 November 2026

**These are machine-time extrapolations from limited observations, and they are not commitments.** They use timings from one easy task on the RTX 5090, with low-end episodes of 76 steps and high-end 330. Resets, state generation, defect repair, reruns and review time are excluded. No new evidence since 1 November justifies changing them.

| Work | Estimate |
|---|---|
| Full preflight | 6-26 hours |
| R22 vs R25, both models, 1,600 attempts (800 per model; corrected 2 November, first written as 800 attempts) | 6-28 hours after preflight |
| Full confirmation, 4,000 attempts (includes R22 and R25) | 17-76 hours after preflight |
| Observation-age study, 1,600 attempts | 10-43 hours |
| Asynchronous study | BLOCKED pending runtime clarification |

## Why R22 vs R25 cannot be valid before today's deadline

Preflight plus R22 vs R25 is at least 12 hours of machine time on these estimates, and fewer than 5 hours remained when this was written. The steps before it also need Alba's decisions. Running part of the design without them would be a protocol change.

## Required sequence

1. Resolve protocol decisions: R1 asynchronous runtime, R2 seeds, R3 bootstrap, R4 wrapper, R6 rendering, R7 states, R9 parity threshold, R10 failed attempts, R12 pi0.5 native, R16 divergence, and the deadline.
2. Fix known code defects (`06_PREFLIGHT_STATUS_AND_CORRECTIONS.md`).
3. Generate and hash initial states, on the host that will run the campaign, since state hashes do not transfer across machines (R15).
4. Complete the official-interface preflight for both models.
5. Obtain Alba's protocol release.
6. Execute matched R22/R25 comparisons.
7. Independently validate results.

No new run is authorised by this handover. The asynchronous study can follow submission.
