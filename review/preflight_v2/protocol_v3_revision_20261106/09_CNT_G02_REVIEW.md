# CNT: review of the proposed G02 compiled-count gate

**Not approved. Not run.** Only Alba can accept compiled-path counting in place of v2's eager count.

## C1. Exact implementation status (read-only inspection, 6 Nov)

| Artefact | Status | Evidence |
|---|---|---|
| `count_compiled_pi05.py` (production gate) | **absent from the repository and the MacBook; RTX UNSEARCHED** | not in the repository (all branches, `git log --all`), and not under `~` to depth 6 on the MacBook. The RTX workspace was not searched (no remote access in this task) |
| `run_preflight_v3.sh` (entry point with G0 and CNT branching) | **absent from the repository and the MacBook; RTX UNSEARCHED** | same search |
| `drivers/pdr_g2_gate.py` (prototype) | **PRESENT**, sha256 `e533a605a577f79677a33959ca99f2f23da4f1f4e138eeaefd1a816e5bfa8fff`, 161 lines; matches `SHA256SUMS` | read-only |
| What the prototype does | Imports the frozen `server_openpi`. Wraps `nnx.Linear` so that the layer matching `action_out_proj`'s signature emits `jax.debug.callback(_tick, sum(head_output))` (data-dependent). Builds stock, P and mutant samplers, installed as `Policy` installs them and driven through `KeyedPolicy.infer`. Two cases: A (seed 20261105, `["fixed-noise"]`) and B (seed 20261106, `["fixed-noise-2"]`). Compares against static-k references. `jax.effects_barrier()` before reading counts. Exits `0 if GATE_OK else 1` (line 161) | lines 6, 30–44, 97–115, 139–161 |
| What the prototype is not | It is not the serving process: it runs as its own diagnostic process. It is not hash-bound to a release. Its non-zero exit path has **never fired** on a real failing run (mutants are scored inside a passing run). It has no CNT-IND, CNT-T or per-request attestation | read-only |
| Recorded result | G02: GATE_OK in 3/3 processes, 0/162 wrong verdicts (2 cases × 9 samplers × 3 d × 3 processes); AI review in 2 rounds only | `runs/G02/result.json`, `07_INDEPENDENT_REVIEW.md` |

### Terms (from `diagnostic_recovery_20261105/03_COUNTING_METHOD.md`)

**Gate components:**
- **CNT-0:** the `received_num_steps` echo equals d. It is reported only, because the wrapper sets it.
- **CNT-1:** the data-dependent callback count in the served object equals d, and the served output is bitwise equal to an uninstrumented compile.
- **CNT-2:** the static-k reference R(d) is bitwise equal to the served output, while R(d−1) and R(d+1) are bitwise different.
- **CNT-NC:** in-run mutants are rejected and the positive control P passes.
- **CNT-REP:** the uninstrumented compile repeats bitwise.

**Samplers and mutants:**
- **P:** a faithful re-implementation.
- **M1:** ignores `num_steps` (always 10).
- **M2:** executes d+1 steps.
- **M3:** correct count, wrong step size.
- **M5:** runs the head twice per step.
- **M6:** skips evaluations by reusing the velocity.
- **K:** evaluates every step but uses the step-0 velocity.
- **W1:** a stale backbone on one step.

## C2. Requirements before Alba can consider approval

### Implementation and hashing

1. `count_compiled_pi05.py` is written from the prototype. It runs **inside the pi0.5 server process before it serves**, on the same `Policy` object that will serve. The served object is the one certified, not a re-jit.
2. Its sha256, and the sha256 of every module it imports from the repository, are entered in Protocol A §A3 before release. A mismatch at runtime aborts.
3. It records the following, and the server refuses to bind its port unless the gate returned 0:
   - `XLA_FLAGS`;
   - JAX, jaxlib and CUDA versions;
   - the GPU UUID;
   - the input-field hashes of SC1 and SC2;
   - every per-cell verdict.
4. The `action_out_proj` signature-uniqueness assertion stays, and fails closed.

### Fail-closed fire drill (required before release; not run)

- **Drill 1:** replace the served sampler with M1 (ignores `num_steps`). The gate must exit non-zero, the server must not bind, and `run_preflight_v3.sh` must not start R9 or any env step. Evidence: exit code, no listening socket, no R9 output file.
- **Drill 2:** corrupt the reference so CNT-2 fails at d=5 only. Same evidence.
- **Drill 3:** run with no release file. G0 must stop before any GPU allocation.
- Each drill is recorded under its own run ID and is never the counted run.

### Independent count verification (CNT-IND)

A second count from a **different instrument on a different program**: the v2 eager count of `action_out_proj` calls under `disable_jit`, with `jax.clear_caches()` after each `ToNNX` call (A01 hook).

- It must equal d at d = 2, 5, 10 on SC1.
- It shares no callback machinery with CNT-1.
- It counts an uncompiled, numerically different program, so it verifies the **loop bound in source**, not the compiled numerics.
- It costs about 52–138 s per call (A01), unmeasured as a whole.

### Independent step-size verification

- CNT-2 already rejects a wrong step size (M3) by bitwise comparison with a static-k reference. That reference **copies the loop body** of `pi0.py`.
- Independence addition: a host-side check, computed outside JAX in float32 numpy, that the time grid visited by the served program is t_k = 1 − k/d for k = 0..d−1, with dt = −1/d.
  - Read it from the callback operand: add the time value to the callback payload.
  - Compare it against T01's arithmetic (exactly d trips for d in 1..200).
- A shared-body bug, such as a velocity sign or mask error, stays invisible to both. That is declared.

### Deterministic XLA timing measurement (CNT-T)

- SC1, d ∈ {2, 5, 10}, 100 warm-up + 100 measured compiled calls.
- Two fresh processes: one with `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`, one with default flags.
- Record host wall time per call, peak device memory, and first-call compile time.
- **No pass threshold.** The numbers inform Alba's choice. Which of the two flags matters is not separated (D01) and is not required.

### Per-request attestation (optional; Alba's choice)

- Each served chunk carries its CNT-1 count. The client refuses a chunk whose count is not d, and the refusal is classed INFRA_POST under Protocol B.
- It closes probe-input specificity (W4).
- Timing cost: unmeasured. It would be measured under CNT-T if adopted.

### Independent review requirements

- **Reviewer:** at least one review by a party who did not write the gate, ideally a human. Both prior rounds were AI reviewers. The reviewer receives the code and the C2 list, not this session's conclusions.
- **Mutation test:** the reviewer supplies at least one mutant not in {M1, M2, M3, M5, M6, K, W1}, and it must be rejected.
- **Budget:** at most 2 rounds. A residual after the last round goes to NOT ESTABLISHED; it does not reopen the design.

## C3. Known limits that remain after all of C2

- Model-code bugs shared between stock and reference are invisible.
- Bitwise equality is fragile to JAX, XLA or driver changes. The gate then fails closed, which is the safe direction.
- One host and toolchain.
- Synthetic inputs, unless attestation is adopted.
