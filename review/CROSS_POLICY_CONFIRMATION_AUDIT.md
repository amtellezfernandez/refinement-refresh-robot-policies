# Cross-policy confirmation audit

**Status: preparation only. The review checkpoint is not ready yet.** No rollout, preflight or GPU
work has run for this campaign. Preflight is waiting for the RTX GPU to be released. Generating the
initial states is waiting for authorisation to copy files to the RTX.

Branch `warren/cross-policy-confirmation`, from `8c92e66`. Local commit, not pushed.

## 1. Planned versus completed

| Cohort | Planned per model | Total | Completed | State hashes |
|---|---:|---:|---:|---|
| `cross_policy_confirmation_v1` | 2,000 | 4,000 | 0 | Pending |
| `observation_age_confirmation_v1` | 800 | 1,600 | 0 | Pending |
| `asynchronous_confirmation_v1` | 1,200 | 2,400 | 0 | Pending, and protocol blocked (R1) |
| Native parity (preflight) | 400 | 800 | 0 | Official states 0-9 |

Each cohort has 400 identities per model. Schedules are deterministic and regenerate with `--check`.
They already fix the attempt IDs and order. The identity and condition order is hashed from the
identity, so it will not change when state hashes are bound, but each `schedule_sha256` will. Both
runners refuse a schedule whose rows have no state hash.

**Failures and interrupted attempts:** none, because no attempt exists.
**Raw journal checksums:** none, because no raw journal exists.
**Invalid or excluded records:** the earlier LeRobot PyTorch pi0.5 campaign (160 attempts), as Alba
specified. Its records are not in this repository (see R8).
**Predeclared tests complete:** none.

## 2. Preflight status

| Check | SmolVLA | pi0.5 |
|---|---|---|
| Checkpoint identity | **PASS**, details below | **PASS**, details below |
| Normalization asset SHA-256 | Not applicable | **PASS**, `b3a44bb2...bd84` matches |
| Source supports d=2 and d=5 | **PASS**, existing runner | **Source reading only**, see below |
| Evaluator runs on the RTX 5090 | NOT RUN | NOT RUN |
| Action units, shapes, finite outputs | NOT RUN | NOT RUN |
| Simulator state serialization and reuse | **CPU pass, one task** | NOT RUN in the client environment |
| Native parity | NOT RUN | NOT RUN |
| Profiling at d=2, d=5, native | NOT RUN | NOT RUN |
| GPU identity | Read: RTX 5090, `GPU-ba05e81b-15e9-b15d-44f4-efddf5fa1e5c`, driver 610.88 | Same GPU |

**Overall preflight: not passed for either model, because it has not run.**

Evidence, all in `experiments/runs/cross_policy_confirmation_v1/preflight_cpu_evidence/`:

- **SmolVLA checkpoint.** Downloaded at revision `6721902b...`. `model.safetensors` is
  `71d9563c...`, which matches both the Hugging Face LFS hash and Alba's recorded runtime manifest.
- **pi0.5 checkpoint.** All 16 GCS objects (11.58 GiB) fetched. Ten were checked by MD5 and size. Six
  composite objects have no MD5, so they were checked by CRC32C and size. No mismatches. GCS publishes
  no SHA-256, so the SHA-256 values recorded are ours. This shows the copy matches the bucket as of
  today; it does not show that the bucket is unchanged since any earlier use.
- **pi0.5 refinement, from source only.** `pi0.py` sets `dt = -1 / num_steps` and integrates from t=1 to
  t=0, so d=2 and d=5 keep the full integration interval, and `num_steps` is a documented argument.
  **This is not the required verification.** `unsupported_refinement_setting` can still fire at runtime.
- **Pinned LeRobot.** Three of the seven pinned source hashes in `run_paired.py` were checked here
  (`modeling_smolvla.py`, `envs/libero.py`, `lerobot_eval.py`), and all three match. The rest are
  checked at runtime by `verify_source_tree`.
- **Shared task files.** The BDDL and init-state trees in `hf-libero` and OpenPI's LIBERO submodule have
  identical digests (135 and 250 files).

## 3. RTX 5090 compatibility

| Item | Finding | Status |
|---|---|---|
| PyTorch | `2.7.1+cu128` installed. Alba's runs used `2.7.1+cu126`, a different binary. | Installed, never run on the GPU |
| JAX | `0.5.3` from `uv.lock`, **with cuBLAS `12.6.4.1`**. The 5090 may need a newer cuBLAS. | **Highest compatibility risk. Unknown until run.** |
| Rendering | No NVIDIA EGL in this WSL (`50_mesa.json` is the only vendor). OSMesa works. | Works on CPU, but see R6 |
| Docker | Not available in this WSL, so the recommended OpenPI route is out. Uses `uv` environments. | Known |
| Environments | `envs/smolvla`, `envs/openpi`, `envs/client` built under `~/alba-cross-policy-20261101`. 36 GiB in total with checkpoints. | Built, locks committed |

If the JAX stack fails on this GPU, changing cuBLAS would change the OpenPI lock. That would be a
deviation to report, not a silent fix.

## 4. Decisions Alba needs to make

**Blocking**

- **R1. There is no official asynchronous runtime for this design.** The specification requires the
  official runtime for both policies. In pinned LeRobot, `async_inference` lists only real robots in
  `SUPPORTED_ROBOTS` and has no simulator path, and `rollout` is built on a `Robot` wrapper. Its `pi05`
  is the PyTorch route that the specification excludes. In OpenPI, `ActionChunkBroker` and the client
  `Runtime` are synchronous, and I found no asynchronous runtime. RTC would also change the sampler
  (guided or inpainted), so it would not be the same policy as NATIVE. The protocol is therefore frozen
  as `BLOCKED_NO_OFFICIAL_RUNTIME_FOR_THIS_DESIGN`, and the guard refuses it. Options: name the runtime
  you meant, accept a custom simulated real-time harness (which is then not "official"), or drop or
  narrow this cohort.
- **R6. Rendering differs from the existing SmolVLA evidence.** Alba's runs used NVIDIA EGL on a 4060 Ti.
  On this machine only CPU OSMesa is available. For the same state and camera: 57-64% of channel values
  differ, mean absolute difference 1.0-1.1 out of 255, maximum 122, about 1% differ by more than 8. A
  negative control using the other camera's frame gave 81.8. This compares one state only. Whether it
  changes policy behaviour is unknown, and native parity is the first evidence. A new campaign would be
  internally consistent but not pixel-comparable with earlier evidence. The options are to accept and
  disclose it, or to run on a native Linux host with NVIDIA EGL.
- **R7. Meaning of "fresh initial-state slots".** Implemented as you wrote it ("newly generated"): states
  drawn from each task's BDDL sampler with a keyed seed, then serialized. These are not the published 50
  LIBERO init states. In one probe on one task, the nearest official state differed by at least 0.034 in
  some state value. The sampler's states have not been shown to be as well settled as the official ones.
  The alternative is unused official IDs (for example 30-34 and 35-39), which stay inside the published
  benchmark but are not "newly generated". Please confirm.
- **R9. No pass criterion for native parity.** The specification lists what to save but gives no threshold
  or reference value. The evaluators will report per-task successes without a PASS or FAIL. Please set the
  criterion before parity runs, so it is not chosen after seeing results.

**Needed before freezing**

- **R2. Observation-age seeds.** You specified "fresh identities" without seeds. Proposed: 820001 and
  820002, clearly marked as proposed in the protocol, with separately drawn states.
- **R3. The frozen bootstrap undercovers.** "Hierarchical" does not say which levels resample. Implemented:
  suites and tasks fixed, states resampled within each task, both seeds kept in each state. This matches
  your earlier analysis scope. In 400 synthetic trials per scenario on this exact design (independent
  outcomes, three known effects), two-sided "95%" intervals covered 90.0-92.2% and the one-sided bound
  92.7-93.8%. Five states per task shrink bootstrap variance by (n-1)/n, or 20%. So the -5 point test is
  somewhat too permissive. The method has not been changed. The options are to keep it and report this
  limit, or to apply a declared correction before any data exists.
- **R4. Noise pairing needs a small server wrapper.** The stock OpenPI server calls `infer(obs)` with no
  noise and draws from one process-wide key, so noise cannot be paired by identity and tick. `KeyedPolicy`
  in `server_openpi.py` passes `num_steps` and `noise` through documented parameters. It sets the private
  attribute `Policy._sample_kwargs`, and noise comes from NumPy rather than a JAX key. Please confirm this
  still counts as "the official server/client path".
- **R5. pi0.5 has no CUDA-event time.** JAX exposes no equivalent of `torch.cuda.Event`. The field is
  recorded as unavailable and host time is not substituted, so that required measure is not comparable
  across models.
- **R8. Identity overlap cannot be fully certified.** The invalid pi0.5 campaign is not in this repository
  (`git grep` for `pi05`, `openpi` and `pi0.5` finds nothing), and neither is `stale_instrumented_004`. Seeds
  are disjoint from every recorded seed (1000, 2000, 2001, 3000, 3001), which makes the four-part identities
  distinct. Generated states are new draws. If you have those records, please add their seeds and state IDs.
- **R10. Failed attempts.** The specification keeps them but does not say how they enter the tests. Implemented:
  no automatic retry, and any missing or failed attempt marks the affected test `incomplete`. A retry
  requires a stated reason and keeps the earlier record. A single unrecoverable attempt therefore prevents
  a test from completing. Please decide whether you want a different rule.

**For your information**

- **R11. The two models do not see the same inputs.** SmolVLA uses 360 px and the LeRobot pipeline; pi0.5 uses
  256 px resized to 224, rotated 180 degrees as in the example. Both settle for 10 no-op steps, and episode
  limits come from each evaluator. These are within-model contrasts only, as you specified, and the
  sign comparison is descriptive.
- **R12. pi0.5 NATIVE is read from source:** h=5, d=10, chunk 10 (`examples/libero/main.py`, `pi0.py`, `config.py`).
  Unlike SmolVLA (h=1), its NATIVE has the same prefix as R52 and R55. Please confirm.
- **R13. Arithmetic is not comparable across frameworks.** SmolVLA uses your length-20 calibration. pi0.5 would need
  XLA cost analysis, which is not implemented yet, and its counting boundary differs. Those columns must
  be reported within model.
- **R14. Precision of the quality test.** Scaling your earlier interval (about +/-6.3 points on 80 states, a different
  contrast) to 200 states suggests the one-sided bound sits about 3 points below the estimate; with independent
  outcomes about 4.6. So R52 minus R22 passes only if the estimate is above roughly -2 to -0.4. These are
  rough indications, not a power analysis, and the outcome correlation for this contrast is unknown.
- **R15. Bit-exact hashes do not transfer across machines.** After settling, one recorded state differed from yours
  by at most 1.2e-17 on 8 of 92 values (AMD Ryzen 9950X, your CPU unknown). Repeat resets on this machine were
  exact. Pairing checks are valid within one machine only, so the campaign should stay on one host.
- **Deadline.** Your email gives no date. The estimates below are many hours per cohort, so please say when the
  results are needed. The repository's notes give ICLR dates that have passed.

## 5. Estimates for later campaigns

Not measured on this GPU. All generation times are from Alba's 4060 Ti discovery run (d=2: 214 ms, d=5: 369 ms,
d=10: 629 ms). Step counts are from four Spatial tasks, so longer suites will exceed them. Simulation with
OSMesa was measured at 125 ms per step for one task at 360 px on 3 cores, against 20.5 ms with EGL.

| Campaign | SmolVLA estimate | pi0.5 estimate |
|---|---|---|
| Preflight | About 16 h (400 native episodes at about 140 s) | Simulation about 3 h, plus unmeasured generation |
| Confirmation | About 33 h, up to about 39 h with resets | Simulation about 13 h, plus unmeasured generation |
| Observation age | About 17 h | Unknown |
| Asynchronous | Blocked (R1) | Blocked (R1) |

GPU generation should be faster on a 5090, and more CPU threads may speed up OSMesa, but neither has been
measured. Models run one at a time.

**GPU memory.** SmolVLA: 1.54 GiB peak reserved on the 4060 Ti (measured, different card and torch build). pi0.5:
unmeasured; parameters are 11.58 GiB on disk, so plan for more than 12 GiB. The 5090 has 31.8 GiB, of which 30.7 GiB was free.

## 6. GPU reservation needed for preflight

One exclusive reservation, released afterwards:

```bash
estate reserve rtx alba-cross-policy-preflight --workload alba-cross-policy-preflight \
  --purpose "Alba cross-policy preflight: one model at a time, no confirmation rollouts" --exclusive
```

- **VRAM:** declare 28 GiB. SmolVLA needs about 2 GiB. pi0.5 is unknown, and this leaves margin.
- **RAM:** 24 GiB.
- **CPU:** rendering runs on the CPU, so the current limit of under 4 RTX threads should be lifted to 8 for this
  work if the coordinator allows it. Otherwise the estimates above stand.
- **Duration:** about 24 h, sequential. Run it under `~/local-science-programme-20261008/gpu.lock`. The reservation ID
  goes in `ALBA_GPU_RESERVATION`, because the guard refuses to start without it.

## 7. Reproduction commands

Offline, with NumPy, from the repository root:

```bash
python3 -m unittest discover -s experiments/tooling/cross_policy -p 'test_*.py'
python3 experiments/tooling/cross_policy/build_protocols.py --check
```

Remaining steps, in order. **None of these has been executed**, and `runner_smolvla.py`, `server_openpi.py` and
`client_openpi.py` have only been syntax-checked:

```bash
# CPU: generate each cohort's states on the RTX, then copy states/ back and rebuild schedules.
generate_states.py --cohort cross_policy_confirmation_v1 --output experiments/runs/cross_policy_confirmation_v1/states
generate_states.py --cohort observation_age_confirmation_v1 --output experiments/runs/observation_age_confirmation_v1/states
build_protocols.py

# GPU, after release. SmolVLA first, in the smolvla environment:
ALBA_GPU_RESERVATION=<rid> runner_smolvla.py preflight --checkpoint $W/ckpt/smolvla_libero --output <new dir>

# Then pi0.5 in the openpi environment, then its server with the client from envs/client:
ALBA_GPU_RESERVATION=<rid> server_openpi.py profile --checkpoint $W/ckpt/checkpoints/pi05_libero --output <new dir>
ALBA_GPU_RESERVATION=<rid> server_openpi.py serve --checkpoint $W/ckpt/checkpoints/pi05_libero
ALBA_GPU_RESERVATION=<rid> client_openpi.py parity --output <new dir>

# Tables, only after a released campaign:
analyze.py --run-root experiments/runs/cross_policy_confirmation_v1
```

Rendering environment on the RTX: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
LD_LIBRARY_PATH=$W/syslibs/usr/lib/x86_64-linux-gnu LIBERO_CONFIG_PATH=$W/libero-config`.

## 8. What has been tested

| Component | Status |
|---|---|
| Journal: write-ahead, terminal immutability, no silent retry, crash recovery | Unit-tested |
| Population counts, pairing structure, condition resolution, determinism | Unit-tested on synthetic state hashes |
| Analysis: planted effect, null effect, incomplete data, margin boundary | Unit-tested on synthetic outcomes |
| Launch guard: refusal without reservation, release, matching hash, or with busy GPU or blocked protocol | Unit-tested with injected inputs |
| `generate_states.py` as a whole file | **Not run.** Its method was probed on one task in `render3.py`. |
| Video, per-attempt array and provenance writers; `provenance.py`; XLA arithmetic | **Not run or not implemented** |
| Observation-age `first action divergence` | **Not implemented.** Inputs are recorded; it needs a paired post-processing step. |
| Asynchronous runner | **Not written,** blocked by R1 |

The offline tests use synthetic data. They show the code follows the stated rules, not that any model behaves
in a particular way. No prior evidence, frozen prediction, submitted PDF or existing experiment directory was changed.

## 9. Environment corrections found on the way

The estate registry records SmolVLA weights as cached at 1.2 GB on the RTX. The cache held 56 KB, so the weights were
absent. This campaign's copy is separate and verified.
