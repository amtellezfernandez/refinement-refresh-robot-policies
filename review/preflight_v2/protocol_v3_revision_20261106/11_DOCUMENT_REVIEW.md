# Independent document-level review (one round, AI reviewer, 6 Nov)

**Reviewer.** A read-only blind reviewer. It received the documents, the primary sources and Alba's ten requirements, but not the author's reasoning. It spot-checked about 18 facts (hashes, line refs, versions, timings, checksums) against sources and found them correct.

**Not a human review.**

## Checklist

| Check | Result |
|---|---|
| All five decisions (R6, R9, R4, N1, CNT) implemented | PASS, with gaps (findings 7–9 below, now addressed or disclosed) |
| No contradictory attempt counts | PASS. Every use of 2,000, 4,000 or "800 total" is marked as removed. 800 per model and 1,600 total are used throughout |
| No unauthorized claims | PASS |
| No invented parity results | PASS |
| No cross-model physical-state equivalence | PASS |
| No unsupported renderer equivalence | PASS. The "probably not" wording was softened (finding 10) |
| No unverified instrument described as a proven defect | PASS |
| Cross-document consistency | FAIL in round 1 (findings 1–3), fixed below |
| Arithmetic in `04` | PASS |

## Findings and dispositions

| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | MAJOR | CHUNK ran a policy-generated chunk through the frozen client, which computes `success`. That contradicts A2 and A8. It also used the name "NATIVE", which Protocol B removes | **Fixed.** The chunk is frozen and replayed as synthetic fixed actions in a new success-free harness; "native setting, not a Protocol B arm" (`01` A5, `05` P2). X16 added |
| 2 | MAJOR | Deterministic flags are certified in A but only conditional in B | **Fixed.** "Certified configuration = executed configuration" (`01` A5). The decline branch is put to Alba (cover note, item 8) |
| 3 | MAJOR | A SmolVLA PREFLIGHT_PASS gates Protocol B while barely testing SmolVLA | **Fixed.** Per-model applicability table; S-SRC source read added; sufficiency put to Alba (cover note, item 7) |
| 4 | MINOR | Retry and continue semantics conflict with frozen `attempt_loop.py:7-9` | **Fixed.** Noted in `02` B5; X15 added; a new runner is required |
| 5 | MINOR | "Stop at first success" was unverified for SmolVLA | **Fixed.** `02` B4 now gives the per-model termination and marks SmolVLA UNVERIFIED |
| 6 | MINOR | Timeout is outcome-correlated; classifier unspecified; two thresholds count different units | **Fixed.** Timeout defined as a ≥3× watchdog; classifier fields listed (not written); units clarified |
| 7 | MINOR | "ABSENT" overstated, since the RTX was unsearched | **Fixed** in `01`, `09` and `00` |
| 8 | MINOR | Dtype column missing; rotation-dimension training ranges unmentioned | **Fixed.** Dtype stages and q01/q99 added (`05` P3). Clipping and frame stay UNVERIFIED (owed source reads) |
| 9 | MINOR | Distribution and architecture "inferred" rather than UNKNOWN | **Fixed:** now UNKNOWN, with the inference noted |
| 10 | MINOR | "Probably not equivalent" overstates one confounded comparison | **Fixed:** "suggests they may differ" |
| 11 | MINOR | CNT terms undefined in this package | **Fixed.** Terms section added to `09` |
| 12 | MINOR | Coverage grid has infeasible cells; MC SE unstated | **Fixed.** Feasibility rule and MC SE added (`02` B6.4) |
| ref | MINOR | `run_paired.py` path | **Fixed** |

**Review budget:** one round was used; the frozen maximum is 2. A second round was not sought. The residuals listed below stay open.

## Residuals (open)

- A human review has not been done.
- The source reads are owed (G10, G11, G12).
- All tooling is unwritten.
- The PROPOSED statistical choices await Alba.
