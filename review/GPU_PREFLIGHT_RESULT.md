# RTX 5090 preflight result, 1 November 2026

## 1. Verdict

**PREFLIGHT_FAILED.** This is the verdict the frozen rule in `review/GPU_PREFLIGHT_PREREG.md` gives the counted run.

**Correction, 2 November 2026.** The first version of this report, at commit `734dec3`, recorded a pass with narrowing and relied on a later rerun. That verdict is withdrawn. The prereg sets the counted run as the first to pass G0 to G2, makes a G4 failure PREFLIGHT_FAILED, and leaves no verdict choice after data. A rerun cannot change that outcome. The raw logs are unchanged.

**What happened.** In the counted run, `alba-preflight-2`, the pi0.5 evaluator failed before starting, because the client script lacked a path to OpenPI's LIBERO package. The rerun, with the same code bundle and only `PYTHONPATH` set, finished its 3 episodes. It is separate engineering evidence and the failure stays in `run_counted/`. The rerun also used separate jobs instead of the entry-point script.

**This was not the mandatory preflight in Alba's specification.** Native parity, 100 + 100 profiling, videos, R52, R55, reset-state reuse across conditions and generated states were not completed. Alba's condition for the review checkpoint is not met, and confirmation readiness is not established.

No confirmation, observation-age or third-policy run took place. No protocol or schedule changed. Alba has approved nothing, and her open decisions are in section 5.

## 2. Model compatibility

| Check | SmolVLA | pi0.5 |
|---|---|---|
| Stack | torch 2.7.1+cu128, CUDA 12.8 | JAX and jaxlib 0.5.3, cuBLAS 12.6.4.1, device `cuda:0` |
| Compatibility with the 5090 | No error | No error; the feared cuBLAS failure did not occur |
| Load | Yes, 17.6 s | Yes, 9.2 s |
| d=2, 5, 10 | Finite, chunk 50 x 7 | Finite, chunk 10 x 7, 30 of 30 chunks distinct at each d |
| Evaluator: NATIVE, R22, R25 | 3 of 3 episodes finished | 3 of 3 finished in the rerun; 0 of 3 in the counted run |

Episodes finishing is what was tested. All six smoke episodes on this one task and state ended in success, which says nothing about success rates.
JAX printed slow-compile notices for one convolution, 3 to 7 s each, and continued.

## 3. Resources and timing

Measured over 30 calls per d after warm-up, on one fixed input per model. Other CPU jobs were running on the machine, so times are not from a quiet host.

| Generation, median host time (p95) | SmolVLA | pi0.5 |
|---|---|---|
| d=2 | 126 ms (160) | 59 ms (61) |
| d=5 | 249 ms (352) | 66 ms (70) |
| d=10 | 425 ms (551) | 77 ms (89) |

| Resource | SmolVLA | pi0.5 |
|---|---|---|
| GPU memory, framework peak | 1.53 GiB reserved at profile, 1.53 GiB (1.646 GB) in episodes | 7.14 GB XLA peak, with preallocation off |
| GPU memory, `nvidia-smi` peak (includes 0.7 GiB baseline) | 2.9 GiB | 9.3 GiB |
| Process RAM peak | 5.4 GiB | 16.0 GiB in the generation process; client 2.2 GiB |
| CPU, whole process | 180 s user + 33 s sys | 125 s user + 76 s sys; client 67 s + 7 s |
| Simulation with OSMesa | about 156 ms per step at 360 px | about 100 ms per step at 256 px |
| First-call compile | none | 29 s in the profile, 11 s through the server |

The earlier plan of 28 GiB and 24 hours was much too large. This whole preflight used about 5 minutes of GPU time.

**Not measured:** XLA arithmetic counts for pi0.5 (R13, not implemented, so no pi0.5 arithmetic figure exists); a CUDA-event time for pi0.5 (R5); and SmolVLA arithmetic and CUDA-event counters on this GPU, because these small scripts did not use them. Stock OpenPI preallocates 75% of VRAM, so the memory above needs preallocation switched off as it was here.

## 4. Rendering and action parity

**SmolVLA, one state only, tick 0: tested, descriptive.** Same noise, with Alba's recorded EGL observation against an OSMesa render of the same initial state. Both ran through this machine's pipeline on the 5090. Pixel similarity was not used to infer any of this.

| First action, L2 in environment units | d=2 | d=5 | d=10 |
|---|---|---|---|
| EGL repeat, numerical floor | 0 | 0 | 0 |
| **OSMesa vs EGL, same noise** | **0.019** | **0.012** | **0.011** |
| Camera swap, positive control | 0.124 | 0.101 | 0.092 |
| EGL with other noises, range over 8 | 0.062-0.280 | 0.158-0.439 | 0.114-0.483 |
| Frozen reading | BELOW_NOISE_RANGE | BELOW_NOISE_RANGE | BELOW_NOISE_RANGE |

- The rendering backend does change the action: the difference is non-zero at every d. It was smaller than the positive control and below all 8 noise draws.
- This is **not a parity pass.** It is one state at one tick, there is no threshold (R9), and it says nothing about closed-loop behaviour or success.
- The two states are not bit-identical (hash differs); the earlier check found differences of at most 1.2e-17.

**pi0.5 action parity: UNTESTED.** No EGL observation exists at its input size and no reference action exists in this repository.

**Across GPUs: UNTESTED.** Alba's recorded action hash includes an array header and mine does not, so the two hashes cannot be compared, and my script did not save the values.

**Not run:** the 400-episode native parity evaluators, for either model.

## 5. Decisions that remain Alba's

None of these was resolved. Detail is in `review/CROSS_POLICY_CONFIRMATION_AUDIT.md`, section 4.

| Item | Question | What this preflight adds |
|---|---|---|
| R1 Async runtime | Name the runtime, accept a custom harness, or drop the cohort | Nothing; still blocked |
| R6 Rendering backend | Accept and disclose OSMesa, or use native Linux with EGL | The one-state action result above |
| R7 Fresh or published states | Newly generated, or unused official IDs | Nothing; state generation waits for this |
| R9 Parity threshold | Set it before any parity data exists | Parity not run, so it can still be set blind |
| R3 Bootstrap coverage | Keep and report the undercoverage, or correct it first | Nothing |
| R4 Noise pairing | Does the server wrapper still count as the official path? | The wrapper ran and passed noise at d=2, 5, 10 |
| R2 Observation-age seeds | Accept 820001 and 820002, or supply others | Nothing |
| **R16 New: first-action divergence** | Which meaning is intended? | See below |

**R16.** The protocol names "first action divergence" without defining it. Under paired noise, FRESH and STALE have identical inputs at tick 0, so the literal first action can differ only numerically. `first_action_divergence.py` therefore reports both the tick-0 distance and the first tick at which actions differ. It is post-processing only, unit-tested on synthetic data, and leaves the protocol unchanged.

R5, R8, R10 and the deadline also stay open as listed in the audit.

## 6. Updated estimates for full experiments

Built from the timings above, sequential, one host. **Extrapolated from one task, not measured campaigns.** The low end assumes 76 steps per episode, as observed on this easy task. The high end assumes 330, the mean of pi0.5's suite limits (220, 280, 300, 520), which I also assume for SmolVLA. Resets, settling and environment rebuilds are excluded. R52 and R55 are computed, not run.

| Campaign | SmolVLA | pi0.5 |
|---|---|---|
| Native parity, 400 episodes | 5-22 h | 1-4 h |
| Confirmation, 2,000 attempts | 12-53 h | 5-23 h |
| Observation age, 800 attempts | 7-30 h | 3-13 h |
| Asynchronous | Blocked (R1) | Blocked (R1) |
| **Total** | **24-104 h** | **9-40 h** |

Altogether about 33-144 hours, or 1.4 to 6 days. The earlier SmolVLA figure of 33-39 h for confirmation sits inside this range. The pi0.5 figures replace "unmeasured generation".

## 7. Exact next permitted operation

**Corrected 2 November 2026.** Alba's condition to reply once protocols, schedules and preflight are committed is not met, because the mandatory preflight is not complete. Any status reply must say so and put section 5 to her as open decisions.

State generation needs R7, parity needs R9, and any rollout needs her release file, which the launch guard requires. The GPU is released and no further GPU work is permitted under this task.

## 8. Limits and deviations

- **Run order.** `alba-preflight-1` was refused by G1 (19,052 MiB in use) and ran nothing. `alba-preflight-2` is the counted run. The rerun used separate server and client jobs (`alba-pf-server`, `alba-pf-client`), with `gpu.lock` held by the server, and a VRAM reading of 0.7 GiB just before it instead of the script's G1.
- **Falsifier noted.** `nvidia-smi` in WSL did list the Ollama process, so G1 was not strictly needed for that case. It stays as a memory check.
- **Reservation.** R014 was exclusive, 18:26Z to about 19:11Z, authorised by Warren. It cannot stop direct Ollama use, so I waited for the other session's model to unload. That work was not stopped.
- **Local change outside the repo.** `estate reserve` needs a service row, so I added `alba-cross-policy-preflight` to `~/estate-ops/estate.json`, with backup `estate.json.bak-20261101-albapreflight`. The row remains.
- **Transfer.** One Warren-approved single-use token, one scp, tar sha256 `a7025413...af33`, and 77 of 77 file hashes matched. Files remain on the RTX: the tar in the Windows home folder, the code in `code-bb86691`, and results in `preflight-bb86691` and `preflight-bb86691-r2`.
- **Not exercised:** `generate_states.py`, the 400-episode parity paths, attempt journals with real rollouts, video writers and `provenance.py`. A later run may still find defects there.
