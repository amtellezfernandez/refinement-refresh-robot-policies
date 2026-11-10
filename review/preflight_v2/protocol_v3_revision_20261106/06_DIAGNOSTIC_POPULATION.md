# Minimal diagnostic population (proposal; nothing generated)

Five policy inputs and one simulator state family. **All are excluded from confirmation.** They are engineering fixtures, not a pilot.

- No policy-driven episode is run on them.
- No success value is computed.
- None of them is drawn from either model's confirmation population.

## D1. Synthetic counter cases (pi0.5 CNT; SmolVLA SCNT analogue)

| ID | Construction | Used by | Detects |
|---|---|---|---|
| **SC1** | `np.random.seed(20261105)`, then `make_libero_example()`; noise key `["fixed-noise"]`. Input-field hashes recorded (X01: state `064003f5...`, image `c6fca23a...`, wrist `465a78e9...`) | CNT-0..REP, CNT-IND, CNT-T, SCNT | wrong executed step count (M1, M2, M6), wrong step size (M3), discarded or stale evaluations (K, W1), non-repeatable compile |
| **SC2** | `np.random.seed(20261106)`, then `make_libero_example()`; noise key `["fixed-noise-2"]`. This is exactly G02's case B (`pdr_g2_gate.py:112`) | CNT-1, CNT-2 | an implementation correct only on SC1 (partial guard against probe-input specificity, W4) |

**What SC1 and SC2 cannot detect:** input-dependent control flow that appears only on real observations. Per-request attestation (`09`) is the remedy for that, and adopting it is Alba's choice.

## D2. Frozen diagnostic observations (R9 parity, SENS, CHUNK)

Each observation is a full policy input (agentview image, wrist image, 8-dim state, prompt), captured once in a fresh env in the **pi0.5 client stack**, serialized with SHA-256, and then frozen. SmolVLA does not use them: R9 is a pi0.5 check.

| ID | State | Image | Prompt | Varies vs DO1 | Detects |
|---|---|---|---|---|---|
| **DO1** | S0: `libero_spatial` task 0, `env.seed(7)`, BDDL placement via `env.reset()`, then 10 no-op wait steps (S0 sim hash `06003c87...`; settled hash to be recorded) | rendered at that state | task 0's language | baseline | wrapper vs stock action-path differences (R9); d not reaching the integrator (SENS, d = 2, 5, 10 differ); chunk slicing (CHUNK) |
| **DO2** | S0 after 20 synthetic fixed actions (seeded sequence, seed 20261108, fresh env), so the arm pose and gripper differ | different (arm moved) | same as DO1 | **image + state** | preprocessing errors that are invisible at a near-default pose: axis-angle conversion away from the reset orientation, state normalization over a wider range, gripper qpos path, wrist-camera content |
| **DO3** | identical to DO1 | identical to DO1 | the language of `libero_spatial` task 1 | **prompt only** | prompt plumbing (`InjectDefaultPrompt`, `TokenizePrompt`). Stock = candidate is required, and DO3's chunk must **differ** from DO1's. If it does not, the prompt is not reaching the model and the DO3 check cannot fail |

**Note on DO3.** A task-1 prompt over a task-0 scene is a counterfactual input, by design. It is used only for action-path equality and sensitivity, never for behaviour.

## D3. Mechanics fixtures (synthetic fixed actions; no policy)

| Fixture | Construction | Used by | Detects |
|---|---|---|---|
| S0 + 20 fixed actions (seed 20261105, as E02) | per model stack, 2 fresh envs + 1 reused env | ENV, ENV-NC | non-deterministic fresh env; a blind determinism check |
| S0 + wrong expected reset hash | per model stack | RESET | a reset check that cannot fire |
| S0 + unit probes (zero, ±x 0.5, +x 1.0, +y 0.5, +z 0.5, +rz 0.5, gripper ±1; 5 steps each; fresh env) | pi0.5 client stack | UNITS | sign or axis errors against P3. Magnitudes recorded, not gated |
| Stub chunk (a fixed synthetic (10, 7) array, no model) | client plan logic | CHUNK pre-check | deque or slicing errors, isolated from the model |

## D4. Exclusion from confirmation (binding)

1. No diagnostic item uses policy seeds 810001 or 810002, or any state slot in either confirmation manifest.
2. **Hash check (the binding check):** every diagnostic state, settled-state and observation hash is compared against **both** model-native confirmation manifests once they exist. Any collision aborts release.
3. **Task-level overlap is unavoidable and is declared:** `libero_spatial` tasks 0 and 1, and their prompts, also appear in confirmation. Isolation is at the level of state, seed and identity, not task.
4. **Unverified assumption carried from revision 1:** that `env.seed(7)` fixes S0's placement. Check 2 does not rely on it.
5. Diagnostic outputs live under `review/preflight_v2/` and are never inputs to Protocol B analysis.

## D5. What this population is not

- It is not a sample of task conditions.
- It supports no claim about behaviour, success or generalisation.
- Parity on DO1–DO3 is parity on these three inputs; generalisation to other inputs is argued from code identity, not measured.
