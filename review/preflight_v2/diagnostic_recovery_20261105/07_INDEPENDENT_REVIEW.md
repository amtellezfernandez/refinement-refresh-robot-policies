# Independent review and how it was reconciled

**Reviewer:** a separate read-only AI reviewer (file reading and search only), not given the author's reasoning. It received:
- the frozen code, the pinned OpenPI source, the drivers and the raw `runs/` outputs;
- the six acceptance criteria from the authorising brief;
- the instruction to grade raw evidence **before** reading any write-up, then to attempt falsification, then to list contradicted sentences.

It did **not** receive the author's conclusions. Rounds: 1 (`review_rounds_max` 2). It is an AI review, not a human one.

At the time of review, E03, E05, T01, D01, G01 and the files `06` and `07` were not in the package. The E03/E05/T01 raw outputs existed only in job logs; they have since been copied into `runs/`.

## Findings and disposition

| # | Finding (reviewer) | Class | Disposition |
|---|---|---|---|
| 1 | "Allocation retention, not physical exhaustion" is wrong: retained bytes are live buffers, and the projected demand exceeds the card | actionable | **Corrected** in `01` (live-buffer retention causing real over-demand) |
| 2 | A01 recorded no `bytes_limit`, reproduced no OOM, and the growth with d is projected | actionable | **Disclosed** in `01`; not re-run (P7 holds the OOM evidence) |
| 3 | The within-call peak under the hook is bounded at 22.80 GiB, which is stronger than stated | actionable | **Corrected** |
| 4 | "Eager is numerically different" is not shown to exceed cross-process noise | actionable | **Corrected**. D01 then measured cross-process drift of about 5e-3 (default flags), the same order as eager-vs-compiled. The claim was narrowed to "count the served program" |
| 5 | I1's callback takes no operand, so it counts call-site visits, not evaluations | actionable | **Fixed and re-run (G01)**: data-dependent operand. Mutant K (reviewer's) is now rejected |
| 6 | Near-miss mutant W1 (stale backbone on one step) could pass under a tolerance | actionable | **Tested in G01**: I1 is blind to it (count d); bitwise I2 rejects it (0.06-0.51) |
| 7 | Mutants were split across processes; no coded verdict; d+/-1 inequality never tested | actionable | **Fixed (G01)**: one process, coded gate, non-zero exit, d+/-1 tested bitwise. GATE_OK = True |
| 8 | A re-jit was certified, not the served object | actionable | **Fixed (G01)**: the callback is compiled into `loaded._sample_actions` at its first trace |
| 9 | F01 vs F02 drift across processes | actionable | **Measured (D01)**. Deterministic XLA flags give bitwise cross-process identity, 2/2. G01 ran with them |
| 10 | The F01 reference uses traced `num_steps`, so it is not independent of plumbing | actionable | **Fixed (G01)**: static-k reference, bitwise equal at 2/5/10 |
| 11 | Probe-input specificity (W4) and serving-process identity (W5) | actionable, not run | **Open.** Proposed: per-request attestation (v3 section 4) and the gate in the server process. Timing cost unmeasured |
| 12 | Bitwise gates fail closed on a toolchain change; a tolerance would reopen W1-W3 | trade-off | **Recorded** in `03` Limits. Bitwise kept |
| 13 | Stock bakes `num_steps=10` as a constant; the HLO differs from the wrapper's | actionable | **Corrected** in `04` |
| 14 | Stock `main.py` also calls `np.random.seed(7)` | actionable | **Added** to `04` as unresolved |
| 15 | "No implementation errors" was unqualified; observation equivalence rests on source reading | actionable | **Qualified** in `04` |
| 16 | E01 did not log `GL_RENDERER`; the renderer negative control is weak | actionable | **Disclosed** in `05` (llvmpipe rests on P4). Not re-run |
| 17 | E02's `reset_goal` comparison is order-confounded | actionable | **Answered by E03** (order-balanced; `runs/E03/result_e3.json`). The partial effect holds at matched reuse positions |
| 18 | "0.05 m per step" is the command range, not the displacement | actionable | **Corrected** in `05` and v3 section 7 |
| 19 | E01: 4/5 distinct renders at sub-ulp state differences are undiscussed | actionable | **Added** to `05` as not explained |
| 20 | The rz representation effect is an untested explanation | actionable | **Relabelled** |
| 21 | The seed-7 isolation argument is an argument; task-level overlap is unaddressed | actionable | **Corrected** in v3 section 6 |
| 22 | "Validated" overreaches in `02`; normalization evidence is outside the package and from the same author | actionable | **Corrected** in `02` |
| 23 | The 1..200 trip-count check has no saved artefact | misread of my packaging | **Fixed**: `drivers/pdr_t_tripcount.py`, `runs/T01/output.txt` (re-run, 0 mismatches) |
| 24 | E03 and E05 raw outputs absent | packaging defect | **Fixed**: `runs/E03`, `runs/E05` (also C01, C02) |
| 25 | `margin_ratio` of about 1e29 is meaningless; I2 identification has no tolerance; B02 margins down to 2.0 | actionable | **Corrected** in `03`. B01's I2 is superseded by G01's bitwise CNT-2 |

## What would change the reviewer's mind (its list) and status

| Condition | Status |
|---|---|
| A data-dependent I1, with bitwise I2 rejecting the W1 and K mutants | **Met (G01)** |
| Evidence that the F01/F02 divergence is a seed or logging artefact | **Not met, and not expected.** Same input hash in both; D01 confirms process-level non-determinism under default flags, removed by flags |
| A measured W1 error well above the residual | **Met:** 0.061-0.51 against a residual of 0 under deterministic flags |
| An order-balanced E03 | **Met** (E03 design) |
| A `GL_RENDERER` log for EGL | Only P4's (outside the package). Not re-run |
| `bytes_limit` plus an uncleaned d=5 failure | Not re-run. P7 holds it |

## Not reviewed

The G01 run itself, D01 and the revised documents were written after the review. No second review round was run. One round remains available under `review_rounds_max`.

## Round 2 (G01 only; the last round under `review_rounds_max: 2`)

A fresh read-only reviewer was given only `pdr_g_gate.py`, `runs/G01/`, the frozen wrapper and the pinned upstream files.

| # | Finding | Class | Disposition |
|---|---|---|---|
| R2-1 | The echo is equal by construction (the wrapper sets and compares it) | actionable | Labelled tautological in G02 (`echo_tautological`); never gated as evidence |
| R2-2 | `served_eq_plain` only shows the instrumentation is numerically neutral; it is not a correctness check | actionable | Kept for that purpose only, and stated so |
| R2-3 | No positive control through the mutant harness | actionable | **G02** runs a correct program P through the same harness |
| R2-4 | Mutants never went through the served path; stock and mutants used different predicates | actionable | **G02**: one predicate; every variant is installed as `Policy` installs its sampler and driven through `KeyedPolicy.infer` |
| R2-5 | K was confounded by a prologue head call (count d+1) | actionable | **G02**: K rewritten (count d, wrong output); only equality can reject it |
| R2-6 | W1 is a value mutant, not skipped work; count alone is insufficient | trade-off | Stated in `03` |
| R2-7 | M5 is functionally correct; rejecting it is a design choice | trade-off | Recorded: the gate counts head evaluations, so redundant evaluations fail closed |
| R2-8 | One input, one noise, d in {2, 5, 10}, batch 1; the rng/no-noise path never exercised; memoisation or early-exit keyed to the probe would pass | actionable / limit | **G02** adds a second input and noise case. The serving path is always batch 1 with noise supplied by the wrapper. Probe-keyed cheating stays **NOT ESTABLISHED**; per-request attestation (v3 section 4) is the proposed answer |
| R2-9 | The deterministic flags are not the production configuration; bitwise equality is empirical | actionable | **G02** runs one process with default flags. v3 makes the flags part of the served configuration, not only of the gate |
| R2-10 | `example_sha` hashed only the state | actionable | G02 hashes every input field |
| R2-11 | Pinned-source check of the local copy of `pi0.py` used for review | checked | Packed from the RTX checkout in the same job that printed `215abfb` and a clean `git status` (source-copy job log; not included) |
| R2-12 | The gate checks the loop controller against a transcription of the same model code; shared-method bugs pass | limit | **NOT ESTABLISHED**: the gate verifies the step count and step size, not model correctness |
