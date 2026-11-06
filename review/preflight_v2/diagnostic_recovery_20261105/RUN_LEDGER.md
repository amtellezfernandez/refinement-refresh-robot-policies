# Diagnostic run ledger: PDR, 5-6 November 2026

All runs are **NON-COUNTING diagnostics** under Alba's authorisation as relayed by Warren. No `run_preflight_v2.sh`, no run001, no parity, no episode, no confirmation identity, no outcome read, no frozen file changed (27/27 `run001_cpu/SHA256SUMS` OK).

GPU runs: RTX 5090 `DESKTOP-J7AGHMV`, exclusive GPU reservation **R018**, `gpu.lock`, RAM watchdog (never fired: no `watchdog.txt` in any run). Frozen code on the RTX: `code-078e166`, hashes equal to local (`runs/*/code_sha256.txt`). OpenPI `215abfb`, clean (`openpi_rev.txt`). Env: `runs/*/env.txt`, identical across GPU runs except `PDR_RUN_ID` and, in D01, `XLA_FLAGS`.

Exact commands: `runs/<ID>/cmd`, or the job log head for E runs. Driver hashes: `06_REPRODUCIBILITY.md`, plus the `driver_sha256` line in each log.

| ID | Purpose / hypothesis | Input | Exit | Key measurements | Result | Interpretation | Remaining uncertainty |
|---|---|---|---|---|---|---|---|
| C01, C02 | Locate the invalid campaign; read LeRobot pi05 config (read-only) | source trees | 0 | `~/lerobot` `c441ebb`: QUANTILES, `num_inference_steps` 10 | context only | cannot be bound to the campaign | campaign records absent |
| A01 | H1: retained memory is held by a JAX cache | unseeded synthetic | 0 | +15.75 GiB after eager d=2; gc 0; `clear_caches` to baseline; hook counts 2/5/10; restore +15.75; 99.99995 % of bytes are per-layer weight slices | **H1 supported, causal** (remove, then restore) | allocation retention, not physical exhaustion | exact cache object unnamed |
| A02 | reproduce A01 | unseeded synthetic (new) | 0 | identical to A01 (15.7476 GiB) | reproduced | | |
| B01 | compiled-path counter I1, identifier I2, mutants M1/M2, stock vs wrapper | unseeded synthetic | 0 | I1: plain 0, cb 2/5/10, M1 10/10/10, M2 3/6/11; cb = plain bitwise; stock = wrapper bitwise at d=10; echo = d for M1 and M2; I2 12/12 correct, margin 3.4-7.4 | PASS with mutants rejected | I1 valid on this input; I2 margin thin | I2 reference construction (superseded by F01) |
| B02 | reproduce B01 | unseeded synthetic (new) | 0 | every verdict identical | reproduced | | |
| X01 | eager vs compiled on one seeded input (corrects a void comparison) | seeded 20261105 | 0 | eager counts 2/5/10; eager vs compiled max 0.023 / 0.0086 / 0.0097; compiled repeat bitwise | eager is a numerically different program | count at the deployed path | |
| F01 | falsify I1/I2: M3 (wrong dt), M5 (head x2), M6 (skipped NFEs), P (faithful reference) | seeded 20261105 | 0 | I1 P 2/5/10, M3 2/5/10, M5 4/10/20, M6 1/1/1; P bitwise = plain; M3 differs 2.40/1.49 | gate = I1 + I2 rejects M1/M2/M3/M6; M5 false reject (safe) | I1 counts NFEs; I2 needed for M3 | the reference shares the loop body with `pi0.py` |
| F02 | reproduce F01 | seeded 20261105 | 0 | verdicts identical; distances differ at about 1e-2 | verdicts reproduced, numbers not | led to D01 | |
| D01 | H: cross-process differences come from XLA kernel selection | seeded 20261105, 4 processes | 0 | default flags: 2 processes differ (about 5e-3); deterministic flags: 2 processes bitwise equal | H supported (n=2/arm) | adopt flags after timing | two flags changed together; speed unknown |
| T01 | float32 trip count of `time >= -dt/2` equals d? | arithmetic, CPU | 0 | 0 mismatches for d=1..200 (numpy float32, JAX 0.5.3 jit CPU) | holds | source arithmetic, not a GPU count | |
| G01 | **coded single-process gate after review**: data-dependent I1 compiled into the served object; static-k bitwise reference; d+/-1; mutants M1/M2/M3/M5/M6/K/W1; deterministic XLA flags | seeded 20261105 | 0 | **GATE_OK = True**: stock passes all conditions at 2/5/10; 19/19 scored mutant cells rejected; K counts d+1; W1 counts d (I1 blind) and differs 0.06-0.51 (I2 rejects); smallest neighbour gap 0.0175 | PASS | the recommended method | one input; fire drill of the exit-1 path owed; probe-input specificity |
| G02 | gate after review round 2: one predicate; served-path install for every sampler; positive control P; K fixed; 2 cases; 3 processes (det, det, default) | seeds 20261105 / 20261106, noise keys fixed-noise / fixed-noise-2 | 0, 0, 0 | GATE_OK in 3/3; 0/162 wrong verdicts; det1 = det2 stock hashes; default flags: no false rejects | PASS | current prototype | probe-keyed cheating; shared-body bugs; fire drill |
| E01 | EGL vs OSMesa; restore; determinism; units | diagnostic state S0, CPU | 0 / 0 | renderers byte-identical (both Mesa software); reused env diverges up to 0.79; zero-action drift 5 cm | renderer: no difference **on this host**; env history-dependent | GPU EGL not testable here | |
| E02 | carried state; fresh determinism; units | S0, CPU | 0 | fresh 3/3 bitwise; carried: timestep, controller goal; `reset_goal` insufficient; units linear, signs right; render repeat 0 | fresh env validated | | carried state partly unknown |
| E03 | which reset removes carried state (order-balanced: fresh env + one dirtying roll per condition) | S0, CPU | 0 | warm-start none; goal partial; timestep none | bounded explanation | | root cause unnamed |
| E05 | norm stats read | file | 0 | hash `b3a44bb2...`; action q01/q99 in [-1, 1] | consistent | | |

## Failed approaches and own defects (preserved, not rewritten)

1. **The wait loop matched its own command text.** My first background wait for A01 matched `python_rc=` inside the echoed job command and reported "done" while the job was still running. No result was read from it. Replaced by a wait on `rc.txt`.
2. **The B01 launch failed: "The command line is too long"** (Windows hop, about 8 KB). Replaced by staging the driver on the host in hash-checked chunks (helper not included).
3. **The A01-vs-B01 eager/compiled hash comparison was VOID.** `make_libero_example()` is unseeded, so the inputs differed. I first reported it as a finding and withdrew it in the same session. Replaced by X01.
4. **B01's I2 reference** passed dt as a traced argument and was not bitwise exact. Superseded by F01's reference; B01's numbers are kept.
5. **E01's unit probes ran in a reused env** and are contaminated by carried state (5 cm drift under a zero action). Superseded by E02's fresh-env probes; kept.
6. **E02's hypothesis that the controller goal is the carried state:** confirmed as present, but it does not explain the divergence (E03). Recorded as bounded.
7. **The A01 within-call peak under the hook is unmeasured**, because `peak_bytes_in_use` is cumulative. Bounded by completion under the 23.88 GiB cap.
8. **The operand-free I1 callback (B01/F01) counted call-site visits, not evaluations.** I described it as counting NFEs. The independent review caught it. Replaced by G01's data-dependent callback.
9. **Packaging gaps found by the review:** E03, E05, C01, C02 and T01 raw outputs had not been copied into `runs/`. They were fixed after the review; the content is unchanged.

Independent review: `07_INDEPENDENT_REVIEW.md` (round 1: whole package; round 2: G01 only).
