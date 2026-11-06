# Phase B: counting denoising steps on the compiled path, with negative controls

**Diagnostics B01, X01 and F01** (RTX 5090, R018, synthetic input; B01 unseeded, X01 and F01 seeded `np.random.seed(20261105)`). Reproductions B02 and F02: see `06_REPRODUCIBILITY.md`.

## Recommendation (revised after independent review; prototype G01)

Count on the **served compiled object** with a **coded two-part gate**, in one process, with deterministic XLA flags (`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`) and a seeded input. Every condition below must hold at every d in {2, 5, 10}, or the process exits non-zero:

- **CNT-1 (I1):** a `jax.debug.callback` whose operand is the head output (data-dependent) is compiled into `loaded._sample_actions` itself. Its count after `jax.effects_barrier()` must equal d. The served output must be **bitwise equal** to a separate uninstrumented compile.
- **CNT-2 (I2):** an explicit reference with **k as a Python constant** (static `fori_loop`, dt = -1/k folded) must equal the served output **bitwise**. The references at d-1 and d+1 must **differ**. Because k is static, this reference does not share the traced `num_steps` plumbing.
- **CNT-NC:** in the **same process**, mutants M1, M2, M3, M5, M6, K and W1 must each be rejected. Cells where a mutant is the correct program by construction (M1 and M3 at d=10) are labelled not-a-mutant and are not scored.
- **CNT-0:** the `received_num_steps` echo must equal d. It is necessary, not sufficient.

**G01 result: GATE_OK = True.** Stock passed every condition at 2/5/10. All 19 scored mutant cells were rejected. The smallest bitwise-distinct neighbour gap was 0.0175 (d=10). Raw: `runs/G01/result.json`.

**G02 (after review round 2; `drivers/pdr_g2_gate.py` `e533a605...`) is the current prototype.**
- One gate predicate for every sampler.
- Each sampler is installed as `Policy` installs it and driven through `KeyedPolicy.infer`.
- A positive control P runs through the same harness.
- K was rewritten (count d, wrong output).
- Two input/noise cases, all input fields hashed.
- Three processes: two with the deterministic flags, one with default flags.

**Result: GATE_OK = True in all 3 processes; 0 wrong verdicts out of 162** (2 cases x 9 samplers x 3 d x 3 processes).
- Stock and P pass everywhere.
- M1 and M3 pass only at d=10, where they are the correct program.
- M2, M5, M6, K and W1 fail everywhere.
- **Default flags gave no false rejects.**
- The two deterministic-flag processes gave identical stock hashes (cross-process bitwise). The default-flag hashes differ from them, as expected for a different compiled program.

Raw: `runs/G02/result.json`.

**Why the revision.** The independent review (`07_INDEPENDENT_REVIEW.md`) found these weaknesses in the B01/F01 design:
- the B01/F01 callback took no operand, so it counted call-site visits, not evaluations;
- the controls were split across two processes;
- no verdict was coded;
- the d+/-1 inequality was never tested;
- a re-jit was certified, not the served object;
- references with traced dt were not bitwise (B01);
- outputs drift across processes under default flags (F01/F02, D01).

G01 addresses each of these. B01, F01 and their reproductions stay in the record as the earlier, weaker stage.

## What is counted

| Instrument | Counts | Shown by |
|---|---|---|
| I1 (B01/F01, operand-free) | **visits to the action-head call site** in the compiled program *(corrected after review: no data dependency)* | M5: 2d; M6: 1 |\n| I1 (G01/G02, data-dependent) | **executed action-head evaluations whose output reaches the callback**; it cannot tell an evaluation that is used from one that is discarded | M5: 2d; M6: 1; K (G02: evaluates every step, uses the step-0 velocity): **d, blind**; W1 (stale backbone on one step): **d, blind**. Count alone is insufficient; I2 is required |
| I2 | whether the served output equals d Euler steps of size 1/d, bitwise | G01: static-k reference bitwise equal at 2/5/10; M3 differs 2.39 / 1.49; W1 differs 0.094 / 0.061 at d = 5 / 10 |
| echo | the requested setting reaching the sampler | **reported d for M1 and M2**, which execute 10 and d+1 steps (B01): not evidence of execution |
| frozen eager count | `action_out_proj` calls in an **uncompiled, numerically different** program | X01: eager vs compiled max abs 0.023 / 0.0086 / 0.0097 at d = 2 / 5 / 10, against a 0.074 gap between d=5 and d=10 |

## Negative-control results (deliverable 6)

| Variant | What it does | I1 count at d = 2 / 5 / 10 | Output vs deployed | Gate verdict |
|---|---|---|---|---|
| plain (uninstrumented) | stock | 0 / 0 / 0 (no callback; shows the counter has no other source) | itself | n/a |
| cb (I1 on stock) | stock + callback | **2 / 5 / 10**, both repeats | bitwise equal at all d (B01) | PASS |
| P (faithful re-implementation) | explicit d steps | 2 / 5 / 10 | **bitwise equal** at all d (F01) | PASS |
| **M1** | ignores `num_steps`, always 10 | **10 / 10 / 10** | I2 identifies k=10 | **REJECTED** at d=2, 5. At d=10, M1 is the correct program |
| **M2** | executes `num_steps + 1` | **3 / 6 / 11** | I2 identifies k = d+1 | **REJECTED** at all d |
| **M3** | correct count, wrong step size | 2 / 5 / 10 (**I1 blind**) | differs 2.40 / 1.49; equal at d=10 by construction | **REJECTED by CNT-2** at d=2, 5 |
| M5 | head twice per step, output unchanged | 4 / 10 / 20 | bitwise equal | rejected (false reject, safe direction; documents that I1 counts NFEs) |
| **M6** | skips NFEs (velocity reused) | **1 / 1 / 1** | differs 0.51-0.57 | **REJECTED** at all d |

*(Corrected after review: in B01/F01 the mutants were split across two processes and inputs, and M1/M3 at d=10 are not mutants. G01 is the single-process run that satisfies this sentence.)* In G01 every stock PASS is paired, in the same process and on the same input, with mutants that the coded gate rejects.

## The six Phase B questions

| Question | Answer |
|---|---|
| What is actually counted | I1: action-head evaluations in the compiled program; I2: equality with d faithful steps |
| Does compilation change the observed quantity | **Count: no.** Eager (A01 with release, X01) and compiled (B01, F01) both give 2/5/10. **Numerics: yes.** Eager is not bitwise equal to compiled (X01). The deployed program is the compiled one, so counts are taken there |
| Does `num_steps` control the computation | Yes. `num_steps` is a traced argument (one compile serves every d: first call 4.7 s, then about 0.06 s). The trip count follows the float32 time condition `time >= -dt/2`, which gives exactly d trips for every d in 1..200 (numpy float32 and JAX-jit CPU; `drivers/pdr_t_tripcount.py`, output `runs/T01/output.txt`; source arithmetic, not a GPU measurement). The GPU measurement is I1 |
| Is asynchronous execution synchronised | Yes: `jax.effects_barrier()` before reading the count; outputs are converted to NumPy (blocking) before comparison |
| Are compilation and execution separated | Partly. The first call includes compilation (4.7 s at d=2); counts are identical on the first and the repeat call, so compilation adds no callbacks. Timing is out of scope |
| Are repeated runs reproducible | Within a process: bitwise. Across processes **with default flags: no**. F01 and F02 (same seed, same driver) differ by about 1e-2, and D01 shows about 5e-3 in the first action. **With deterministic flags: 2/2 processes bitwise identical** (D01). Verdicts reproduced in A02, B02 and F02 |

## Failed or superseded approaches (kept)

- **B01's I2 reference** passed dt as a traced argument. It matched the deployed output only to 2.5e-3-5.5e-3 (margin ratio 3.4-7.4 to the neighbouring k). It identified k correctly in 12/12 cases, but the margin is thin. It is **superseded** by F01's reference, which computes dt as `pi0.py` does and is bitwise exact. B01's numbers are preserved.
- **The A01 vs B01 eager-vs-compiled hash comparison is VOID**: different unseeded inputs. It was replaced by X01 (one seeded input, one process).
- **The frozen eager count** works only with the cache-release hook, at 52-138 s per call (A01), and counts a numerically different program. It is not recommended as the primary instrument. It can be kept as a third, structural cross-check.

## Limits

- One host, JAX 0.5.3, OpenPI `215abfb`; synthetic inputs (two seeded, two unseeded).
- I1 identifies `action_out_proj` by signature (in = width, out = action_dim), asserted unique among the model's `nnx.Linear` modules. A model change could break that. The assertion fails closed.
- I1 runs a different compiled program from the plain one (it adds an effect). Its count transfers to the plain program through bitwise output equality at each d on the test input. It is not a direct count of the plain program. CNT-2 covers the plain program directly.
- CNT-2's reference shares source with `pi0.py` (it copies the loop body). It is not independent of a bug in the shared body, such as a wrong velocity sign. It is independent of the loop bound and step size, which is what it is for.
- **Open after review (not run).**
  - **Probe-input specificity (W4):** the gate certifies one seeded synthetic input and one noise key. An implementation with input-dependent control flow, such as an adaptive early exit, could behave differently on real observations. The remedy is **per-request attestation**: return the I1 count with every served chunk and refuse any chunk where it differs from d. Its cost on timing is unmeasured.
  - **Fire drill:** the gate's non-zero exit path (`sys.exit(1)`) was never triggered on a real failing run. Its mutants were scored inside a passing run.
  - **Gate in the serving process (W5):** G01 instrumented the served object in its own process. v3 must run the gate in the same server process that then serves, or use per-request attestation.
  - **Bitwise fragility:** a JAX, XLA or driver change can break bitwise equality and make the gate fail closed. Relaxing it to a tolerance would reopen near-miss mutants. W1's errors (0.06-0.51) and the neighbour gaps (at least 0.0175) are the scale any tolerance would have to respect.
  - The reference copies the loop body, so a shared-body bug (sign, mask, prompt) is invisible to CNT-2.
