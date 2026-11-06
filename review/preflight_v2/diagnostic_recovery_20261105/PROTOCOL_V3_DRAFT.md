# cross_policy_preflight_v3: DRAFT / NOT RELEASED

> **DRAFT, NOT RELEASED.** Proposed by Warren's diagnostic session, 5-6 November 2026, for Alba's review. Nothing here authorises run001 of v2 or v3, a counted preflight, parity, confirmation or any outcome analysis. v2 (`experiments/protocols/cross_policy_preflight_v2.json`, `4fc520ec...`) stays frozen and unchanged. A release requires Alba's rulings in section 9 and a release file bound to this document's hash.

## 1. Research question

Can the SmolVLA and OpenPI pi0.5 execution paths be run at d in {2, 5, 10} on the RTX 5090 host with the executed denoising computation verified on the deployed path, normalization verified, and the evaluation environment defined, before any confirmation attempt?

## 2. Pinned identities (verify unchanged before and after the counted run)

| Object | Identity |
|---|---|
| Repository base | `4248b58` (`warren/preflight-v2-review`); frozen v2 files unchanged since `078e166` (27/27 `run001_cpu/SHA256SUMS`) |
| pi0.5 checkpoint | `gs://openpi-assets/checkpoints/pi05_libero/`, 16 files, hashes as P0 (`rtx_recovery_20261105/evidence/p0/expected_sha256_local.txt`); `norm_stats.json` `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
| SmolVLA checkpoint | `model.safetensors` `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`; revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479`, clean; JAX 0.5.3; env `envs/openpi` (Python 3.11) |
| Client | `envs/client` (Python 3.8), MuJoCo 3.2.3, LIBERO from `src/openpi/third_party/libero` |
| SmolVLA env | `envs/smolvla`, torch 2.7.1+cu128, MuJoCo 3.8.1 |
| Host | `DESKTOP-J7AGHMV`, WSL2 `6.6.87.2`, RTX 5090 32,607 MiB, driver 610.88 |
| Frozen tools | `server_openpi.py` `c3eca9a4...`, `client_openpi.py` `767cad39...`, `common.py` `afccd5e9...`, `launch_guard.py` `718edbd1...` |
| New v3 tools (to be written, then hashed into this section before release) | `count_compiled_pi05.py`, built from the prototype `drivers/pdr_g2_gate.py` `e533a605...` and run inside the server process |

## 3. Execution paths

- **pi0.5:** stock `create_trained_policy(pi05_libero)` + `KeyedPolicy` (documented `num_steps` and `noise` only; bitwise equal to stock at d=10, `04_STOCK_VS_WRAPPER.md`). Served **compiled** (`module_jit`). The eager path is not used to serve actions.
- **pi0.5 determinism flags:** `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`. With default flags, compiled outputs differ across processes by about 5e-3. With these flags, 2/2 processes were bitwise identical (D01). **Speed cost unmeasured; a timing check must precede adoption.**
- **SmolVLA:** unchanged from v2.
- **Synthetic counting input:** `np.random.seed(20261105)` before `make_libero_example()`; record the input hashes (X01: state `064003f5...`, image `c6fca23a...`, wrist `465a78e9...`). The v2 profile used an unseeded example; v3 does not.

## 4. Counting method (replaces v2's eager `count_denoise_loop` for pi0.5)

Run in the **pi0.5 server process before it serves**, with the determinism flags (section 3), on the seeded synthetic input, with fixed noise key `["fixed-noise"]`. The prototype is `drivers/pdr_g2_gate.py` (`e533a605...`, G02: GATE_OK in 3/3 processes, 0/162 wrong verdicts). The predecessor is `pdr_g_gate.py` (G01). Exit non-zero, and do not serve, unless every row holds at d = 2, 5 and 10:

| Gate | Pass condition |
|---|---|
| CNT-0 | `received_num_steps == d`. Reported only: it is set by the wrapper and equal by construction |
| CNT-1 | the data-dependent callback count in the **served** object equals d, and the served output is bitwise equal to an uninstrumented compile of the same sampler. Two input/noise cases |
| CNT-2 | the static-k reference R(d) is bitwise equal to the served output; R(d-1) and R(d+1) are bitwise **different** |
| CNT-NC | in the same process, under the same predicate: positive control P passes; M1, M2, M3, M5, M6, K and W1 are rejected (M1 and M3 at d=10 are the correct program and must pass) |
| CNT-REP | the uninstrumented compile repeats bitwise |

**Proposed addition, for Alba's choice:** per-request attestation. Each served chunk carries its I1 count, and the client refuses a chunk whose count differs from d. This closes probe-input specificity. Its timing cost is unmeasured.

**Optional structural cross-check:** the v2 eager count with `jax.clear_caches()` after each `ToNNX` call (about 5 min). Reported, not gated.

SmolVLA's count stays as in v2 (P3 passed: denoise steps = 100 x d).

## 5. Simulator and renderer

- **Fresh env per attempt** for every counted rollout (already the case in `attempts` and `conditions`). A reused env is history-dependent (E01-E03).
- **Renderer: R6 decision required** (section 9). This host can only provide Mesa software rendering. P4 logged llvmpipe for EGL, and E01 found OSMesa and EGL byte-identical; E01 did not log the renderer string.
- **Parity mode** reuses one env per task in stock's episode order. That order must be preserved exactly and recorded. Parity results are reported as properties of that reused-env procedure.

## 6. State generation and isolation

- Confirmation states: generated per v2 (`generate_states.py`, keyed seeds; R7). None exist yet (`PENDING_STATE_GENERATION` in all six schedules).
- **Diagnostic isolation rules:**
  - (a) diagnostics use only `make_libero_example` synthetic inputs, or the diagnostic state S0 (`06003c87...`, `libero_spatial` task 0, `env.seed(7)` reset placement);
  - (b) no diagnostic uses policy seeds 810001, 810002, 820001 or 820002, or any generated slot;
  - (c) before release, the generated-state manifest must be checked for hash collision with every diagnostic state hash. That check is the binding one. The supporting argument (no keyed state seed among 40,000 equals 7) assumes `env.seed(7)` fixed S0's placement, which is **unverified**. Task-level overlap (`libero_spatial` task 0 and its prompt) is unavoidable; isolation is at the level of state and seed;
  - (d) diagnostic outputs live under `review/preflight_v2/diagnostic_recovery_20261105/` and are never inputs to analysis.

## 7. Action units

LIBERO delta-EEF OSC: input [-1, 1] maps to a **commanded** goal offset of +/-0.05 m and +/-0.5 rad per step at 20 Hz. The **measured** displacement is about 0.0088 m per step at action 1.0 (E02); gripper -1 open, +1 close. pi0.5 outputs pass through quantile `Unnormalize` (stats in [-1, 1]) and the first 7 dims, with no further conversion (E01, E02, E05). The v2 action-unit acceptance criterion stays **Alba's (R9)**.

## 8. Acceptance, failure and abort

```selection-and-gates
counted_run: the first execution of `run_preflight_v3.sh` under Alba's v3 release whose job start is recorded by the job log and the scheduler-independent `started` file before any output is read
rerun_policy: none for a gate failure; an infrastructure crash before CNT-0 is UNINTERPRETABLE and requires a new run ID, recorded, with the crashed run kept
post_data_free: NONE
gate: G0 | Alba's release file bound to this document's sha256 exists | enforced_at: run_preflight_v3.sh (to be written; must branch before any GPU use) | before: all GPU work | qualified_on: matching release file -> pass; absent or hash-mismatched release -> fail (to be exercised on both before release; v2's G0 at run_preflight_v2.sh:17 is the template)
gate: CNT | CNT-0..CNT-REP all pass at d=2,5,10 | enforced_at: count_compiled_pi05.py (to be written; exits non-zero before parity) | before: SmolVLA parity and any rollout | qualified_on: stock and P -> pass; M1/M2/M3/M5/M6/K/W1 -> fail (G02, 3 processes, 2 cases, 0/162 wrong); fire drill of the non-zero exit still owed
gate: ENV | fresh-env determinism: 2 fresh envs, S0, 20 fixed actions, bitwise equal | enforced_at: run_preflight_v3.sh | before: parity | qualified_on: fresh -> pass (E02 3/3); reused env -> fail (E01, E02)
selection_risk: material
selection_anchor: hash logged before output: before launch, the sha256 of the exact job command and the run ID are sent to Alba (custody outside the operator); the first execution started after that message counts
multiple_executions: any second started execution under the same release = NO VALID RESULT for v3.run001
stop_on_confirmation: if all gates pass, CLOSE the preflight as PASS and hand to Alba; no added probes
verdicts: PREFLIGHT_PASS | PREFLIGHT_FAILED | INSTRUMENT_DEFECT | UNINTERPRETABLE
review_rounds_max: 2
```

**Open lint findings (disclosed):** `prereg_lint.py` reports GATE_POINTER for G0 and CNT because `run_preflight_v3.sh` and `count_compiled_pi05.py` are not written yet. Before release, they must exist, branch at the cited lines, and pass a fire drill (gate forced to fail, so the budgeted step never starts).

Order inside `run_preflight_v3.sh`: G0, then the pi0.5 counting gate (CNT), **before** SmolVLA parity. In v2, parity ran first and would have been spent before the pi0.5 count failed.

Abort conditions: any gate fails; `RESOURCE_EXHAUSTED` anywhere; a frozen-file hash change; a confirmation identity or seed is requested by any step; the GPU reservation is lost.

## 9. Decisions only Alba can make (blocking release)

| ID | Decision |
|---|---|
| R6 | GPU EGL on a native Linux host, **or** OSMesa/llvmpipe on this WSL host preregistered as a separate evaluation environment |
| R9 | Native-parity and action-unit acceptance criteria. Also: is the parity reference stock's reused-env procedure, or fresh-env? |
| R4 | Is `KeyedPolicy` (bitwise equal to stock at d=10) the official execution path? |
| N1 | State-generation stack (SmolVLA env MuJoCo 3.8.1 vs client MuJoCo 3.2.3) |
| CNT | Accept compiled-path counting (CNT-1 + CNT-2 + in-run controls) in place of v2's eager count |

## 10. Diagnostic evidence

`01_ROOT_CAUSE_JAX_MEMORY.md` (A01), `03_COUNTING_METHOD.md` (B01, X01, F01), `04_STOCK_VS_WRAPPER.md` (B01), `05_SIMULATOR_RENDERER_UNITS.md` (E01-E03, E05), `06_REPRODUCIBILITY.md` (A02, B02, F02), `07_INDEPENDENT_REVIEW.md`.

## 11. Reproducibility commands

See `06_REPRODUCIBILITY.md`. Every diagnostic is `drivers/mkjob.sh <ID> <driver> <reservation> <timeout>`, launched as a job on the RTX, with the driver hash checked in-job.

## 12. Not claimed

No policy quality, no parity, no R22 vs R25 effect, no SmolVLA instrument validity, no GPU-EGL equivalence, nothing about the invalid LeRobot campaign beyond `02_EARLIER_CAMPAIGN_BLAST_RADIUS.md`.
