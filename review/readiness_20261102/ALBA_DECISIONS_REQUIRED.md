**Superseded by `review/preflight_v2/DECISION_REGISTER.md`. Kept as a record.**

# Decisions required from Alba, 2 November 2026

Defaults below are what the code does now. They are proposals, and nothing is decided until Alba says so. R numbers are defined in `review/CROSS_POLICY_CONFIRMATION_AUDIT.md`, section 4, and R16 in `review/GPU_PREFLIGHT_RESULT.md`.

## Blocks the full preflight

| ID | Question | Current default | Why it blocks |
|---|---|---|---|
| R9 | What is the native-parity pass criterion? | None | Must be set before parity data exists |
| R6 | OSMesa with disclosure, or native Linux with EGL? | OSMesa | Decides the host; hashes do not transfer (R15), and parity would be repeated |
| N1 | **New.** Which simulator stack defines the states for pi0.5? | States generated in the SmolVLA environment | See below |
| N2 | **New.** Is the LIBERO asset and task source acceptable for the "official" pi0.5 client? | Client reads the SmolVLA environment's BDDL and init files | See below |
| R4 | Does the noise and `num_steps` wrapper count as the official path? | Wrapper in use | Decides whether the pi0.5 gate can pass |
| R12 | Is pi0.5 NATIVE h=5, d=10, chunk 10? | Yes, from source | Parity uses it |

### N1: different MuJoCo versions

Lock files: SmolVLA `mujoco==3.8.1`, `robosuite==1.4.0`. pi0.5 client `mujoco==3.2.3`, `robosuite==1.4.1`. `generate_states.py` runs in the SmolVLA environment, so pi0.5 would restore states generated under a different simulator version.

Within-model R22 vs R25 stays paired, because both arms use one stack. But the same state may not be equally valid under both, and settled states may differ between models. Whether either version matches published LIBERO evaluation has not been checked.

Options: generate one set per model; or keep one set and report the difference; or align versions, which would be a deviation from the OpenPI lock.

### N2: LIBERO source for the client

The client's `LIBERO_CONFIG_PATH` file points `bddl_files` and `init_states` at the SmolVLA environment, and `assets` at `~/.cache/libero/assets`. BDDL and init trees were hash-matched with OpenPI's submodule (135 and 250 files). **Assets have not been compared.**

## Blocks state generation and final schedules

| ID | Question | Current default |
|---|---|---|
| R7 | Newly sampled states, or unused official IDs? | Newly sampled |
| R2 | Observation-age seeds | 820001, 820002, proposed |

## Blocks the R22/R25 release

| ID | Question | Current default |
|---|---|---|
| N3 | **New.** R22 and R25 first, or all five conditions? | Schedules contain all five; no subset path exists |
| N4 | **New.** What independent outcome verification do you want? | None beyond journal checks; see `R22_R25_EXECUTION_PLAN.md` |
| R3 | Bootstrap undercoverage: keep and report, or declare a correction first? | Keep; matters most for the -5 point test |
| R10 | How do failed attempts enter tests? | Any failed or missing attempt makes the test `incomplete`; a retry needs a reason |
| R13 | How should arithmetic be reported across frameworks? | SmolVLA calibration only; pi0.5 not implemented |
| R5 | pi0.5 has no CUDA-event time. Acceptable as recorded? | Recorded as unavailable |
| N5 | **New.** Is stopping on the first failed attempt acceptable? | Stop and review |
| N6 | **New.** Release format | `review/releases/<protocol>.release.json` with `approved_by`, `protocol_sha256`, `schedule_sha256s`; `full_preflight.release.json` for preflight |
| R8 | Records of the invalid campaign and `stale_instrumented_004`, for identity overlap | Seeds disjoint from recorded seeds only |
| — | Deadline for this work | Unknown |

## Blocks later cohorts

| ID | Question |
|---|---|
| R1 | Which asynchronous runtime, or a custom harness, or narrow or drop the cohort? |
| R16 | Meaning of first action divergence |

## Needed for the SmolVLA evidence

From the handover, unanswered as of this document:

1. Identity and location of the invalid pi0.5 campaign.
2. Why it was invalid.
3. Whether that defect could affect `run_paired.py`, pinned LeRobot or the SmolVLA runs.

## Smallest useful reply

R9, R6, N1, N2, R4 and R12 are enough to write the release for the full preflight, once the GPU is released to it.
