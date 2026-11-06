# Phase E: simulator, renderer and action units

**Diagnostics E01, E02, E03 and E05:** CPU only, OpenPI client environment (`envs/client`, MuJoCo 3.2.3, LIBERO from OpenPI's submodule), the frozen `client_openpi` helpers, **no policy, no episode, no success signal read**.

**Diagnostic state S0:** `libero_spatial` task 0, `env.seed(7)` as in the client, then the BDDL placement sampler via `env.reset()`; SHA `06003c87...`, 92 values. It is not an official init state and not a generated confirmation slot. Actions are synthetic: a fixed pseudo-random sequence (seed 20261105) and unit probes.

## Issue 1: EGL against OSMesa

| | |
|---|---|
| Reproduction | E01: same S0, separate processes with `MUJOCO_GL=osmesa` and `MUJOCO_GL=egl` |
| Result | **Byte-identical**: agentview, wrist, preprocessed 224 px policy image, 8-dim state, and the sim state after 20 fixed actions (`compare.json`: max_abs 0 for every item) |
| Root cause | **From P4, not from E01:** P4 logged `GL_RENDERER` = Mesa llvmpipe for EGL in all three envs, and only `50_mesa.json` as an EGL vendor. E01 did not log the renderer string *(review finding)*, so E01's byte identity is consistent with, but does not itself show, both being Mesa software |
| Negative control | **Weak** *(review finding)*. Agentview against wrist differs (max 224), which shows only that the comparison can report a difference between different images. A control that perturbs the renderer itself was not run |
| What it does **not** show | Equivalence to **GPU EGL** on NVIDIA, which is Alba's reference stack (4060 Ti). Earlier evidence suggests the two differ: OSMesa against her EGL gave a non-zero SmolVLA first-action L2 of 0.011-0.019 on one state (1-2 Nov) |
| Proposed correction | None possible on this host. **R6 is a choice for Alba:** (a) a native Linux host with NVIDIA EGL, or (b) preregister **OSMesa/llvmpipe on WSL as a separate evaluation environment**, with no claim of renderer equivalence to the reference. Not substituted silently |
| Residual | GPU-EGL equivalence unmeasured |

## Issue 2: a reused environment is history-dependent

| | |
|---|---|
| Reproduction | E01: in one reused env, the same restored S0 plus the same 20 actions ended at states differing by up to **0.79** (max abs over the sim state) between repeats. A fresh env matched the first repeat to 7e-15. E02: 3 fresh envs, 3 **bitwise-identical** rollouts. A reused env differs from fresh by **0.54-0.65** |
| What carries over (E02) | `set_init_state` restores the flattened sim state exactly (`H_simstate_restored_exact: true`). The env keeps `timestep` (30, not 0) and the OSC controller's `goal_pos`/`goal_ori` from the previous trajectory |
| One variable at a time (E03, `runs/E03/result_e3.json`) | Each condition: a fresh env, one identical dirtying rollout, then two test rollouts, **so reuse positions are balanced across conditions** (this answers the order confound the review found in E02's `X` comparison, which is kept but not relied on). Max abs vs fresh at reuse positions 2, 3: none 0.575 / 0.795; warm-start zeroed 0.575 / 0.795 (no effect); `reset_goal` 0.307 / 0.474 (partial); goal + warm 0.307 / 0.474; goal + warm + `timestep=0` 0.307 / 0.474. **Residual divergence remains under every combination.** `reset_goal` on a fresh env changes nothing (E02, 0.0) |
| Root cause | **Bounded, not identified.** State outside the flattened MuJoCo state: the controller goal (confirmed, part of it) plus something not reset by goal, warm-start or timestep. Candidates, untested: other controller or gripper-controller internals, actuator `ctrl`/`act`, observable buffers |
| Correction | **Fresh env per attempt**, which `client_openpi.py` already does for `attempts` and `conditions` (from probe-4). Validated deterministic here (3/3 bitwise) and before (probe-5, 5/5 builds) |
| Negative control | The reused env is the control: the same comparison goes red there |
| Consequence for parity | Stock `examples/libero/main.py`, and the frozen `parity` mode that mirrors it, **reuse one env per task across episodes**. Stock outcomes therefore depend on episode order and history. A native-parity comparison is apples-to-apples only if the episode order and env reuse are replicated exactly. Confirmation attempts (fresh env) run in a **different, cleaner environment** than the stock reference. **R9 should say which is the reference.** |
| Residual | One task, one state. The unexplained carried state is not named |

## Issue 3: state restoration and determinism

- Restoration of the flattened state is exact (E02).
- Renders of one fixed state repeat exactly: 3/3, max 0 (E02 `P`).
- **Not explained:** in E01, five settles of the same restored state in one reused env gave **4 distinct agentview hashes**, while the settled states differed by at most 2e-16. Either sub-ulp state differences change pixels, or the reused env's render path carries state. Fresh env per attempt avoids both. The cause is not established.
- Fresh-env physics is deterministic under fixed actions: 3/3 bitwise (E02 `F`); under EGL and OSMesa alike (E01).
- One task and one state only. `alba-v2-probe-5` covered 6 combinations across all four suites for settling.

## Issue 4: action units and conversion

| Link | Evidence | Status |
|---|---|---|
| Model output to controller input | Quantile `Unnormalize`, affine, unclipped; action q01/q99 in [-1, 1]; gripper -1.0 / 0.9996 (E05, `norm_stats.json` `b3a44bb2...`). The wrapper inserts no transformation (bitwise equal to stock, `04`) | consistent at source level |
| Controller | `OperationalSpaceController`, input range [-1, 1], **command** range +/-0.05 m (pos) and +/-0.5 rad (rot) of goal offset per step, kp 150, 20 Hz (E01) | config read from the env. **Measured displacement is smaller:** about 0.0088 m per step at action 1.0 (E02: 4.41 cm in 5 steps) *(review correction)* |
| Empirical response, fresh env, 5 steps (E02 `U`) | zero: 0.0 m; +x 0.5: +2.16 cm; -x 0.5: -2.15 cm; +x 1.0: +4.41 cm (linear); +y 0.5: +2.69 cm; +z 0.5: +2.72 cm; gripper +1: fingers close (+/-1.34 cm), -1 open (`DUMMY_ACTION`) | signs, axes and linearity as the delta-EEF convention; small cross-coupling (x motion gives -0.5 cm z) |
| Rotation | +rz 0.5 changes the absolute axis-angle mostly in its 2nd component | **untested explanation**: plausibly an axis-angle representation effect near pi. No rotation-matrix measurement was made |
| Units against the training data | Not directly testable without dataset actions. Indirect: stock example passes the same actions straight to `env.step` | **assumed from source** |

The earlier E01 probes ran in a reused env and showed 5 cm drift under a zero action. That was contamination from Issue 2. They are superseded by E02 and preserved as a failed approach.

## Wrapper compatibility

The client checks shape (n, 7), finiteness and length >= h. B01 gave (10, 7), float64, finite at d = 2, 5, 10.
