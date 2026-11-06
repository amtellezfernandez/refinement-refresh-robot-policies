# R9 action parity, action-unit specification, and R4 approval evidence

**Status:** specification only. **No parity has been run under this specification.**

Prior diagnostic B01/B02 found stock = wrapper bitwise at d=10 on two **synthetic** inputs, in **one process**. That is not this check, and it is not cited as its result.

## P1. The independent stock-native reference (R9)

| Item | Reference (stock) | Candidate (`KeyedPolicy` via server) |
|---|---|---|
| Code | OpenPI `215abfb`: `policy_config.create_trained_policy(get_config("pi05_libero"), ckpt)`, then `policy.infer(obs, noise=N)` | frozen `server_openpi.py` `c3eca9a4...`: `KeyedPolicy.infer` with `__d__=10`, `__noise_key__=K` |
| Wrapper, server, websocket | **none** | as served |
| Step count | stock default (`num_steps=10`, a Python constant baked into the trace; `pi05_libero` sets no `sample_kwargs`) | `num_steps=10` passed as a traced int |
| Noise | the frozen array N, loaded from `.npy` and hash-checked | derived inside the wrapper from key K. **Precondition:** the wrapper's `noise_for(K)` must equal N bitwise, checked and recorded first |
| Process | separate process, own interpreter (`envs/openpi`) | the serving process |
| Flags | the A4 deterministic XLA flags | the same |
| Inputs | DO1, DO2, DO3 (`06_DIAGNOSTIC_POPULATION.md`), loaded from frozen files by hash | the same files |
| Written by | a script that does not import `server_openpi`, `client_openpi` or `common` | n/a |

**Why separate processes.** Independence: the reference shares no wrapper or serving code with the candidate. The cost is that bitwise equality must hold **across processes**.

- Under default flags, cross-process drift was about 5e-3 (D01).
- Under the deterministic flags it was 0 in 2/2 processes, for the *same* program (D01).
- Stock and the candidate compile **different HLO**: constant versus traced `num_steps` (`04_STOCK_VS_WRAPPER.md`). Cross-process bitwise equality between two *different* programs has **not been measured**.

A failure here is a failure. It is reported as R9_PARITY_FAILED, with no tolerance applied after the fact (P4).

## P2. Pass conditions (same pinned host and toolchain)

For each of DO1, DO2 and DO3 at d=10:

1. **Shape:** both outputs are (10, 7). That is chunk length × action dims, after `LiberoOutputs` takes the first 7 of 32.
2. **Dimension order:** identical, and equal to the P3 table. The check compares dimension indices, and also confirms that the 7 dims map one-to-one to the env action vector.
3. **Finite:** all 70 values finite in both outputs.
4. **Bitwise identical:** the SHA-256 of the C-contiguous array with dtype and shape is equal (`client_openpi.digest`), and max |a − b| = 0 is reported alongside.
5. **Dtype:** equal. The expected value is float64 at the `Policy.infer` output (B01). **Verified on two synthetic inputs only.**

**Native chunk use (h=5), the only use of h=5:** this is Protocol A's CHUNK check.
- The DO1 d=10 chunk is saved by this parity step as a frozen array.
- That array is replayed as synthetic fixed actions, with no live policy, in a fresh env restored to DO1's state.
- It is replayed through the plan logic at the stock native setting (d=10, h=5). This is a preflight setting, not a Protocol B arm.
- The plan logic must execute exactly chunk[0], …, chunk[4], in order, then issue its next request at tick 5. Each executed action hash must equal the matching chunk-element hash.
- The check runs 6 ticks, in a harness that computes no success value.

## P3. Action-dimension specification (pi0.5 → LIBERO)

| Dim | Meaning | Model space | After `Unnormalize` (env input) | Controller mapping (commanded) | Frame |
|---|---|---|---|---|---|
| 0 | Δx end-effector | normalized | [-1, 1] nominal (quantile q01/q99 in [-1, 1]) | ×0.05 m → commanded goal-position offset | see note F |
| 1 | Δy | same | same | ×0.05 m | F |
| 2 | Δz | same | same | ×0.05 m | F |
| 3 | Δrx (axis-angle) | same | same | ×0.5 rad → commanded goal-orientation offset | F |
| 4 | Δry | same | same | ×0.5 rad | F |
| 5 | Δrz | same | same | ×0.5 rad | F |
| 6 | gripper | same | q01 −1.0, q99 0.9996 | −1 open, +1 close | n/a |

- **Range note.** The rotation dimensions (3–5) are **not** spread across [-1, 1] in the training statistics. Action q01/q99 are −0.116/0.140, −0.169/0.181 and −0.195/0.312 (`norm_stats.json` via `runs/E05/log`). For dims 0–2 they are −0.747/0.937, −0.796/0.860 and −0.938/0.937. Full-scale ±0.5 rad commands are therefore outside the training range.
- **Dtypes:**
  - model output float32 on device; stated as an expectation, **not verified per stage**;
  - `Policy.infer` output float64, (10, 7) (B01, two synthetic inputs);
  - sent to the env as Python floats (`action.tolist()`, `client_openpi.py:124`);
  - robosuite's internal dtype UNVERIFIED.
  - The R9 check records the dtype at each of these stages.
- **Normalization (input):** quantile `Normalize` on the 8-dim state: eef position 3, axis-angle 3, gripper qpos 2. The state is then padded to the model width (`PadStatesAndActions`). Images are resized to 224 with padding, after a 180-degree rotation in the client.
- **Unnormalization (output):** `(x + 1)/2 · (q99 − q01 + 1e-6) + q01`, quantile mode, **no clipping** in OpenPI (`transforms.py`, `04`). Loaded statistics equal `norm_stats.json` `b3a44bb2...` (probe_r0, same author).
- **Clipping:** none in OpenPI. Clipping of controller input to [-1, 1] by robosuite's OSC `scale_action` is **asserted from robosuite design, not verified against the pinned robosuite 1.4.1 source in this revision**. Owed before release: a source read.
- **Controller convention:** robosuite `OSC_POSE`, delta mode, input range [-1, 1], output range ±0.05 m and ±0.5 rad, kp 150, 20 Hz. The configuration was read from the live env in E01.
- **Units:** env input is dimensionless in [-1, 1]. Commanded offsets are in metres and radians. The gripper is a dimensionless command.
- **Note F (frames):** the reference frame of the delta (world/base versus end-effector) is **UNVERIFIED** in this revision. Owed: a source read of pinned robosuite `OperationalSpaceController` and its `control_delta` handling. E02 showed that +x commands move the eef along +x in the observation frame (+2.16 cm in 5 steps at 0.5), which is consistent with a base or world frame. That is not proof.

**Commanded is not measured.** Translation ±0.05 m and rotation ±0.5 rad are **commanded goal offsets per step** at full-scale input. They are not physical displacements. The one measured response, E02 (fresh env, one state, 5 steps), was 4.41 cm at action 1.0, about 0.0088 m per step. It is a diagnostic observation of one configuration, and it is not a unit specification.

## P4. If cross-host comparison becomes necessary (specified, not run)

Bitwise parity is required **on one host and toolchain**. If a comparison across hosts or toolchains is ever needed (for example, this WSL host against Alba's native Linux host), no tolerance may be chosen after seeing parity data. The procedure is:

1. **Freeze first.** Fix the inputs (DO1–DO3, SC1, SC2), the d values, and the metric: per-dimension max |Δ| on the (10, 7) chunk, in env units.
2. **Repeatability, per host, before any parity data.** On each host separately, measure within-host spread from R repeated fresh-process runs of the **same** program on the same inputs, with no cross-host comparison. R = 10 is PROPOSED. Record the per-dimension maximum.
3. **Preregister the tolerance** from step 2 alone. PROPOSED rule: tolerance = 2 × the largest within-host per-dimension spread, capped below half of the smallest d-neighbour gap observed for that input. The neighbour-gap floor recorded so far is 0.0175 (G01, d=10, synthetic). The tolerance and the rule are frozen and sent to Alba **before** any cross-host parity output exists.
4. **Then** collect the cross-host parity data once and apply the frozen tolerance.
5. **If the within-host spread is zero on both hosts,** the tolerance is zero: bitwise is required.

None of these measurements is run here.

## P5. R4: what `KeyedPolicy` must show before Alba can approve it

`KeyedPolicy` is a **candidate**. **It is not approved.** Approval is Alba's, and it needs all of the following, recorded under Protocol A:

| # | Evidence | Status now |
|---|---|---|
| 1 | Independent stock-native parity (P1, P2) on DO1–DO3 at d=10, bitwise | **not run** |
| 2 | Frozen diagnostic observations used, hashes bound, isolation check passed (ISO) | **not generated** |
| 3 | Verified input preprocessing. The client's `policy_input` output equals stock `examples/libero/main.py` preprocessing on the same raw env observation (image rotation, resize-with-pad, state concatenation, axis-angle), bitwise. B01 noted this rested **on source reading only** | **not run** |
| 4 | Verified normalization. NORM check: file hash plus loaded-statistics equality, quantile mode | done once in probe_r0 (5 Nov, same author, not independent); to be repeated in the counted preflight |
| 5 | Verified action mapping. P3 table checked against source, including the owed clipping and frame reads; UNITS sign and axis check | **partly** (E02 signs and axes; source reads owed) |
| 6 | Counting (CNT) passes on the candidate path | prototype only (`09`) |

The echo `received_num_steps` is not evidence for any of these.
