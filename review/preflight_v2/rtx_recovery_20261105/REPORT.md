# Preflight v2: RTX checks and pi0.5 step-count findings (technical appendix)

5 November 2026. Status: **PREFLIGHT_INCOMPLETE**. `run001` has not been executed. Every run described here is **non-counting**: none used `run_preflight_v2.sh`, none wrote to a run001 directory, and none ran parity, rollouts or confirmation attempts. No frozen file was changed. The 1 November PREFLIGHT_FAILED record, the excluded pi0.5 campaign, the stale-observation result and the submitted PDF were not touched or used.

Contents:
1. Identities
2. State checks
3. Host health
4. Results
5. pi0.5 step count
6. SmolVLA profile
7. Rendering
8. Not covered
9. Options for the pi0.5 step count
10. Reproducing
11. Evidence index

## 1. Identities

| Item | Value |
|---|---|
| Code | `experiments/tooling` and `experiments/protocols` at commit `078e166b91752fdfc4fa914d25661123843eaa18` (branch `warren/preflight-v2-review`). Copied to the RTX; 64 of 64 file hashes match (`evidence/p0/`) |
| Host | RTX 5090 (32,607 MiB), driver 610.88, CUDA UMD 13.3, WSL2 kernel 6.6.87.2 |
| pi0.5 | `gs://openpi-assets/checkpoints/pi05_libero/`, 16 files; all match the hashes from an independent download (`evidence/health/pi05_check.txt`). `norm_stats.json` `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
| SmolVLA | `model.safetensors` `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479`; JAX 0.5.3 (CUDA). SmolVLA env: torch 2.7.1+cu128 |
| Baseline env (pi0.5 runs) | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `OMP_NUM_THREADS=3`, `MUJOCO_GL=osmesa`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, offline HF; per run in `evidence/*/env.txt` |

## 2. State checks (MacBook)

- The 27 checksums recorded at the CPU stage still verify. A copy with one hash altered fails, so the check can fail.
- 102 harness tests pass at `078e166`.
- No release file exists under `review/releases/`.

## 3. Host health

GPU smoke passed: 50 matmuls of 8192 x 8192 at about 54 TFLOPS, finite, and a 20 GB allocation verified exactly. Torch and JAX see the GPU (`evidence/health/log`).

The WSL kernel has run since 2 November 07:34Z, which is before the 2 November fault, and the cause of that fault is not established. The WSL disk is 93% used (71 GB free).

## 4. Results

| Run | What | Outcome |
|---|---|---|
| `probe_r0` | pi0.5 loaded on CUDA: `Unnormalize` statistics against `norm_stats.json`; then an eager loop count | Normalization: 8 of 8 fields equal, quantile mode on; a perturbed value compares unequal. Loop count: d=2 gave 2 calls; the next d ran out of GPU memory |
| `probe_r1` | Fresh processes per d; compiled timing | Eager d=5, d=10 and default-steps all out of memory at the 23.88 GiB cap. Compiled path: d=2, 5, 10 give three different chunks for the same noise; warm 58, 62, 73 ms; peak 6.64 GiB |
| P1 | **Frozen** `server_openpi.py profile`, baseline env | Out of memory inside `count_denoise_loop`, in image encoding (`pi0.py:234 embed_prefix`). No `profile.json` |
| P2 | P1 + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` | Out of memory, further on: inside the denoising loop (`pi0.py:278` -> `step:261` -> `gemma.py:405`) |
| P5 | P1 + fraction 1.0 (accepted: limit 31.84 GiB) | Out of memory in the denoising loop; peak 31.29 GiB |
| P6 | P1 + `XLA_PYTHON_CLIENT_ALLOCATOR=platform` | Out of memory in the denoising loop; peak 31.40 GiB |
| P7a, P7b | Frozen functions under a separate instrumenting driver; per-stage JAX live memory | Section 5 |
| P3 | **Frozen** SmolVLA `preflight()` up to its profile; parity blocked | Passed; section 6 |
| P4a-c | EGL context and one rendered frame, three environments | Section 7 |

P2, P5 and P6 are experimental environment changes. They are not evidence about the frozen protocol.

## 5. pi0.5 step count

The frozen check counts `action_out_proj` calls under `jax.disable_jit()` at d=2, 5, 10 in one process. It returns the counts only at the end, so a failed run records none.

**P7: JAX live memory (`bytes_in_use`, GiB), default allocator, cap 23.88 GiB.** The driver wraps `embed_prefix`, `embed_suffix`, `action_out_proj` and the per-d call boundary at runtime, and the frozen file is unchanged.

| d | Stage | P7a: d=2 then d=5 | P7b: d=5 alone, fresh process |
|---|---|---|---|
| | model loaded | 6.47 | 6.39 |
| 2 | image encoding done | 8.93 (+2.46) | n/a |
| 2 | prefix pass done | 13.52 (+4.59) | n/a |
| 2 | steps 1, 2 done | 18.09, 22.64 (about +4.57 each) | n/a |
| 2 | returned; after `gc.collect()` | 22.59; **22.59** | n/a |
| 5 | outcome | **failed during image encoding** (23.87 live) | image +2.46, prefix +4.59, 2 steps; **failed in step 3** (23.81 live) |
| 10 | | not reached | not run |

**Established.**
- The frozen eager count does not complete on this host under any of four settings: the default 23.88 GiB cap, fraction 0.95, fraction 1.0 and the platform allocator.
- At the default cap, d=5 fails both after d=2 in the same process and alone in a fresh process.
- Live JAX memory rose +2.46 GiB with image encoding and about +4.6 GiB with each language-model pass, prefix or denoising step, and was not freed within the call. After the d=2 call returned and after garbage collection, live memory fell from 22.64 to 22.59 GiB, freeing 0.05 GiB. About 16.1 GiB above the post-load level stayed allocated.
- The prefix pass (images plus prompt, roughly 1,000 tokens; an estimate, not counted) and a 10-token denoising step retain about the same amount, so the retained memory is not KV-cache or activation memory.

**Inferred.**
- The per-pass amount is close to the size of the model weights, which suggests a weight-sized copy per pass under eager execution.
- The frozen run fails at d=5. Its log records no d, but its failure frame and memory plateaus match P7a.

**Not established.**
- What holds the retained references.
- Physical VRAM exhaustion, which P7 does not show: P7 ran under the 23.88 GiB cap.
- What d=10 alone would need. About 59 GiB is a straight-line projection from two steps, not a measurement.

## 6. SmolVLA profile (P3)

The frozen `runner_smolvla.preflight()` ran unmodified up to its profile. Parity was blocked by replacing `AttemptJournal`, first used where parity begins (`runner_smolvla.py:292`), with a function that raises. The output holds only `profile.json` and `provenance.json`.

| d | generations | denoise steps | host median (s) | CUDA median (ms) |
|---|---|---|---|---|
| 2 | 100 | 200 | 0.134 | 126 |
| 5 | 100 | 500 | 0.224 | 211 |
| 10 | 100 | 1,000 | 0.390 | 368 |

Counts equal generations x d at every d. Timing is on one fixed `libero_spatial` task-0 input, not rollout timing. **Parity has not been run.**

## 7. Rendering (P4)

An EGL context is created and made current in the SmolVLA (MuJoCo 3.8.1), OpenPI client (3.2.3) and OpenPI (2.3.7) environments. In each, EGL resolves to **Mesa llvmpipe**, a software renderer. The only EGL vendor is Mesa; WSL provides no NVIDIA EGL. A 64 x 64 test frame is pixel-identical between EGL and OSMesa.

On normal process exit, the EGL context destructor raised `EGLError`; that is unresolved. Nothing here shows compatibility with native NVIDIA EGL or equality on LIBERO scenes.

## 8. Not covered

Native parity, action-unit checks, matched-reset first actions, condition smoke tests, videos, rollout timing, generated-state hashes and reset-mismatch termination on the real simulator. They belong to run001, and parity also needs the R9 criterion before any parity data exists.

`run_preflight_v2.sh` runs SmolVLA conditions and parity (400 episodes) before the pi0.5 profile. So if run001 is launched as frozen, it would spend those episodes and then very likely fail, and under the protocol that failure would stand.

## 9. Options for the pi0.5 step count (none implemented)

Each changes the frozen measurement step, so each needs a new protocol version and run ID.

| Option | What it needs first | Open risk |
|---|---|---|
| Find and release the retained eager memory | A non-counting run listing live arrays at call end, then clearing JAX caches and re-measuring | d=10 may still not fit even after release |
| Count on the compiled path, independently validated | An in-loop host callback, read after `jax.effects_barrier()`, in a separate process and jit object from timing. Controls: d=2, 5, 10 give 2, 5, 10; the default gives 10; a repeat at the same d gives the full count; the hook disabled gives 0; output SHA equal to the plain compiled path; a mutation that makes the loop ignore `num_steps` must fail | It instruments a different executable from the one rollouts use; a counter keyed to object identity reads 0 under jit |
| Run the eager count elsewhere (CPU, or a GPU with more memory) | Memory sizing | If the same retention applies, d=10 needs more than WSL's 47 GiB of RAM. A 2 November CPU run reached d=5 at 27.4 GiB RSS before being stopped |

## 10. Reproducing

Each run's exact command is `evidence/<run>/job_command.sh`. Driver scripts are alongside (`p3/p3_profile_only.py`, `p7a/p7_diag.py`, `p4b/probe_egl.py`, `p4c/probe_render.py`, `probe_r1/probe.py`). They need the RTX workspace (`~/alba-cross-policy-20261101`: environments, checkpoints, `code-078e166`). None of these runs was repeated for this handoff. `MANIFEST.sha256` and the relative links were re-verified from a fresh clone of this branch, and the 102 harness tests pass there.

## 11. Evidence index

| Folder | Content |
|---|---|
| `health/` | GPU, framework and checkpoint-hash checks |
| `probe_r0/`, `probe_r1/` | First pi0.5 CUDA probes; jitted chunks (`.npy`) |
| `p0/` | Code-transfer hashes, local and RTX |
| `p1/`, `p2/`, `p5/`, `p6/` | Frozen pi0.5 profile, baseline and three settings; `python.log`, `resources.txt` (`nvidia-smi` every 3 s), `env.txt`, `rc.txt` |
| `p3/` | SmolVLA profile: `out/profile.json`, `summary.txt` |
| `p4a/`, `p4b/`, `p4c/` | EGL and OSMesa |
| `p5pre/` | Fraction 1.0 accepted by the runtime |
| `p7a/`, `p7b/` | Per-stage memory: `out/diag.json` |

Exit codes: job wrappers ran under zsh, so early `rc_*` fields are empty, and exit 0 is not evidence of success. Results rest on logs and output files.
