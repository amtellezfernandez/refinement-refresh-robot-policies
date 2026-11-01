# Bounded RTX 5090 preflight: frozen before any GPU result

Question: can both models load, generate at d=2, 5, 10, and run their evaluators on this GPU, and what do they cost?
This is engineering readiness. It is not scientific approval, and it does not touch any confirmation identity.

Pinned artifacts: the commit containing this file; checkpoints and locks in
`experiments/runs/cross_policy_confirmation_v1/preflight_cpu_evidence/`; reference inputs
`experiments/runs/development_gate_001/inputs.pt`, checked against `manifest.json`.

## What runs (entry point `experiments/tooling/cross_policy/run_gpu_preflight.sh`)

| Step | SmolVLA | pi0.5 |
|---|---|---|
| Load and device | yes | yes, plus JAX, jaxlib and cuBLAS versions and a GPU matmul |
| Generation at d=2, 5, 10 | 10 warm-up + 30 measured, one fixed input | 3 warm-up + 30 measured, the OpenPI example input |
| Evaluator | 3 episodes: NATIVE, R22, R25 | 3 episodes: NATIVE, R22, R25, through server and client |
| Action comparison | EGL reference against OSMesa render | UNTESTED, see below |

Authored values, not measured costs: the call counts, episode count, 2 h wall ceiling, 3 CPU threads per process,
smoke seeds 790011 and 790012, state `official0` of `libero_spatial` task 0. Those seeds and that state are
outside every confirmation and parity identity. Episode outcomes are smoke results and are not success-rate evidence.

## Action comparison

- **SmolVLA, one state.** `inputs.pt` holds Alba's EGL observation and the noise used for it. With that same noise,
  at d=2, 5, 10: EGL repeat (numerical floor), OSMesa render of the same state, EGL with 8 other noises (natural
  scale), and `camera_swap` (positive control: the measure must see a large image change).
- Reported: first-action L2 and per-dimension maximum in model and environment units, plus chunk mean L2.
- Reading, descriptive only: `BELOW_NOISE_RANGE`, `WITHIN_NOISE_RANGE` or `ABOVE_NOISE_RANGE`, comparing OSMesa to the
  8 noise draws. **Not a parity pass.** Alba has set no threshold (R9), and one state cannot establish parity.
- Pixel similarity is not used to infer action similarity.
- **pi0.5: action parity UNTESTED.** No EGL observation exists at its 256 px input and no pi0.5 reference action
  exists in this repository. Resizing the 360 px reference would not be the same observation.

## Gates (each enforced by a branch in the code that spends the budget)

| Gate | Condition | Enforced at |
|---|---|---|
| G0 authorisation | `ALBA_GPU_RESERVATION` set | `launch_guard.py`, `run_gpu_preflight.sh` |
| G1 GPU quiet | under 3 GiB used before start. `launch_guard` lists compute processes, which WSL may not report, so this is separate | `run_gpu_preflight.sh` |
| G2 lock | `gpu.lock` taken without waiting | `run_gpu_preflight.sh` |
| G3 model | loads; every chunk finite, expected shape | preflight scripts; failure stops the run |
| G4 evaluator | all 3 episodes finish without exception, success or not | preflight scripts; failure stops the run |

```selection-and-gates
counted_run: the first execution of run_gpu_preflight.sh that passes G0 to G2
rerun_policy: none automatic; a failed run is kept, and a rerun needs a recorded reason and a new output directory
post_data_free: NONE for verdicts; the prose of the report
selection_risk: low: no scientific estimate is drawn; outputs are engineering checks on non-confirmation identities
stop_on_confirmation: stop and report; no further GPU work
verdicts: PREFLIGHT_PASS | PREFLIGHT_PASS_WITH_NARROWING | PREFLIGHT_FAILED | PREFLIGHT_BLOCKED
review_rounds_max: 0
```

## Verdict rule, fixed now

- **PREFLIGHT_BLOCKED:** a gate G0 to G2 fails, or the transfer cannot happen, so no model was tested.
- **PREFLIGHT_FAILED:** G3 or G4 fails for either model, including JAX or cuBLAS incompatibility.
- **PREFLIGHT_PASS_WITH_NARROWING:** G3 and G4 pass for both, with any claim narrowed. This applies while the asynchronous
  cohort is blocked (R1) or pi0.5 action parity is UNTESTED, so it is the best outcome reachable now.
- **PREFLIGHT_PASS:** G3 and G4 pass, nothing blocked, action parity tested for both models.

Falsifiers of the current account: d=2 or d=5 unsupported or non-finite on pi0.5; JAX falling back to CPU; WSL's
`nvidia-smi` listing compute processes (which would make G1 redundant, not wrong).

Not established under any outcome: native parity, success rates, the 400-episode evaluators, EGL against OSMesa beyond one
state, any effect of refinement or refresh, or XLA arithmetic counts, which remain unmeasured (R13).
