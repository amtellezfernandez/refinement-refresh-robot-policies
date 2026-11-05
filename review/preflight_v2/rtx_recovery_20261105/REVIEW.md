# Independent review

Each stage of this work was reviewed by a separate read-only reviewer. It received the frozen code, the raw evidence and the acceptance criteria, and had to grade the runs before reading the author's write-up. There were four reviews: host recovery, P0-P4, P5-P6 and P7.

**Outcome.**
- In every review, the reviewer's grading of each run matched the author's.
- None found a breach of the run constraints: no `run_preflight_v2.sh`, no run001, no parity, no frozen-file change. Lock lapses were found and are disclosed below.
- Every review found reporting errors in the write-up, and every substantiated one was corrected before this package. The ones that changed a conclusion:

| Stage | Correction |
|---|---|
| P0-P4 | P1 and P2 fail at different stages (image encoding against the denoising loop); an earlier "same place, raising the cap changed nothing" was wrong |
| P0-P4 | "Amendment necessary" was downgraded to "likely" while allocator settings remained untested |
| P0-P4 | The EGL teardown error was marked unresolved; the follow-up runs had skipped the destructor |
| P0-P4 | Added: SmolVLA parity runs before the pi0.5 profile in `run_preflight_v2.sh` |
| P5-P6 | Peak device use was 31.4 GiB, below the 31.84 GiB card ("GB" figures had been MiB/1000); "demand exceeds the card" was withdrawn |
| P5-P6 | Two disclosures: the fraction 1.0 acceptance check ran without the GPU lock, and the reservation ID differs between runs |
| P7 | The retained increment does not scale with sequence length, so it is not KV-cache or activation memory, as first written |
| P7 | The verdict is bound to the 23.88 GiB cap; "not fixable by settings or process isolation" was withdrawn; the compiled-path option was downgraded from "best route" to "candidate to validate" |
| P7 | Disclosure: the "prefix done" mark is not synchronised on the KV cache |

**Lock lapses.** Three runs did not take the GPU lock:
- the health check, including a 20 GB allocation, within a 21-second job; no other GPU process was listed;
- the EGL runs (P4), under 1 GiB for a few seconds;
- the fraction-1.0 acceptance check, under 1 GiB for a few seconds.

**Open reviewer points, not acted on (they need new runs):**
- list live arrays at call end and clear caches, to identify the holder;
- re-run the EGL check with a normal exit;
- run d=10 alone.
