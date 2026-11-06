# Phase D: stock pi0.5 against the server wrapper, at native settings

**Diagnostic B01** (GPU; synthetic `make_libero_example()` input, unseeded but shared by every comparison in the process). Static source comparison on the MacBook against OpenPI `215abfb` (`examples/libero/main.py`, `policy.py`, `pi0.py`, `libero_policy.py`, `transforms.py`).

## Result

At native settings the wrapper and the stock policy produce **bitwise-identical actions**. On this input, `KeyedPolicy.infer(d=10, noise=n)` equals `Policy.infer(obs, noise=n)` with the config's default sample kwargs: `stock_vs_wrapper_max_abs = 0.0`, both SHA `07c2c7f7...`. The wrapper adds keys to the response and changes nothing in the action path.

| Item | Stock | Wrapper (`server_openpi.KeyedPolicy`) | Classification |
|---|---|---|---|
| Model and checkpoint | `create_trained_policy(get_config("pi05_libero"), ckpt)` | the same call (`load()`), plus a hash check of `norm_stats.json` = `b3a44bb2...` | identical; hash gate is an intended addition |
| Input transforms | `InjectDefaultPrompt, LiberoInputs, Normalize, InjectDefaultPrompt, ResizeImages, TokenizePrompt, PadStatesAndActions` | same object, untouched | identical |
| Output transforms | `Unnormalize` (quantiles), `LiberoOutputs` (first 7 dims) | same object; loaded stats = file (probe_r0) | identical |
| Normalization | quantile: `(x+1)/2*(q99-q01+1e-6)+q01`, no clipping | same | identical. Action q01/q99 lie in [-1, 1] (gripper -1.0 / 0.9996): LIBERO controller units |
| Observation structure | client sends `observation/image`, `observation/wrist_image` (224 px, rotated 180 degrees), `observation/state` (eef pos 3, axis-angle 3, gripper qpos 2), `prompt` | `client_openpi.policy_input` builds the same keys the same way; the server pops only `__d__` and `__noise_key__` | identical. The one source difference: stock `_quat2axisangle` clips `quat[3]` in place; the client's copy does not mutate. Returned value identical. Intended |
| Number of denoising steps | `sample_actions(num_steps=10)` default; `pi05_libero` sets no `sample_kwargs` (`{}`) | sets `num_steps=d` through the documented `sample_kwargs` | d=10 identical; other d intended |
| Randomness | `jax.random.split` of a process key when no noise is passed | passes `noise=noise_for(key)` (documented `Policy.infer` argument) | **intended transformation**: the stock draw is order-dependent (`stock_rng_noise_differs: true`); keyed noise makes it per-identity |
| Compilation | `module_jit(model.sample_actions)` called **without** `num_steps`: the default 10 is a Python constant baked into the trace | the same `module_jit` object, called with `num_steps=d` as a traced int (one compile serves every d: first call 4.7 s, later 0.06 s) | **different HLO programs** from the same source *(corrected after review)*; their outputs at d=10 are bitwise equal on the inputs tried |
| Output | `actions` float64 (10, 7) + `policy_timing` | same + `host_generation_s`, `memory`, `received_num_steps`, `server_configured_num_steps` | intended additions |
| Action chunk use | `replan_steps=5` (NATIVE h=5); the first 5 of 10 actions executed | condition-dependent h; NATIVE h=5 | intended (the experimental variable) |
| Env constants | resolution 256, resize 224, 10 no-op waits, `env.seed(7)` once per env, max steps 220/280/300/520; **also `np.random.seed(7)` at start** | same constants; the client does **not** call `np.random.seed(7)` | constants identical. The global numpy seed is **unresolved**: whether anything in the client's episode path draws from it was not tested |
| **Env reuse** | **one env per task, reused across that task's episodes** | `attempts`/`conditions`: **fresh env per attempt**; `parity`: one env per task, as stock | **intended, but not neutral**: see `05_SIMULATOR_RENDERER_UNITS.md`. A reused env is history-dependent |

## Implementation errors found

**None found in the action path**, on two synthetic inputs at d=10 (B01, B02) plus source reading. Observation-structure equivalence rests on **source reading only**: the client's `policy_input` was not run against stock `main.py` on a common observation.

## Unresolved

1. **R4 (Alba):** whether the wrapper counts as the official execution path. Technically it is stock plus documented arguments; bitwise identity at d=10 is shown on one input. That is the evidence. The ruling is hers.
2. **The echo `received_num_steps` is not proof of execution.** In B01 it reported the requested d for both negative-control mutants: 10 steps executed when 2 or 5 were requested, and d+1 executed. Keep it as a necessary check only.
3. **Native parity in simulation is not tested here** (not authorised). Only the action path was compared.
4. One synthetic input. Bitwise identity at d=10 is a property of the code path. Other inputs were not tried. Generalisation is argued from the code, not measured.
