# Preflight status and corrections, 2 November 2026

## Recorded verdict

**PREFLIGHT_FAILED** for the bounded engineering preflight of 1 November. The counted run, `alba-preflight-2`, failed gate G4 for pi0.5 because of a harness path error (`ModuleNotFoundError: No module named 'libero'`). The frozen rule in `review/GPU_PREFLIGHT_PREREG.md` gives this verdict. A later rerun does not change it.

**The mandatory preflight in Alba's specification is separate, and is PREFLIGHT INCOMPLETE.** Confirmation readiness is not established. Alba's condition for the review checkpoint is not met.

## Runs

| Job | Outcome | Evidence |
|---|---|---|
| `alba-preflight-1` | Refused at G1 (19,052 MiB in use); ran nothing; exit 4 | `preflight_gpu_evidence/control/attempt1_G1_refusal.log` |
| `alba-preflight-2`, counted | SmolVLA passed G3 and G4. pi0.5 passed G3. **pi0.5 G4 not reached; exit 1** | `preflight_gpu_evidence/run_counted/` |
| `alba-pf-server`, `alba-pf-client`, rerun | pi0.5 evaluator finished 3 episodes, using separate jobs and with `PYTHONPATH` set. Separate engineering evidence only | `preflight_gpu_evidence/run_rerun_pi05_evaluator/`, `control/rerun_*` |

`preflight_gpu_evidence/` is under `experiments/runs/cross_policy_confirmation_v1/`.

## Check by check

| Check | SmolVLA | pi0.5 |
|---|---|---|
| Checkpoint identity | PASS | PASS: copy matches bucket as of 1 November |
| Normalization asset hash | Not applicable | PASS |
| Load on RTX 5090 | PASS | PASS |
| Finite chunks at d=2, 5, 10; 30 measured calls | PASS | PASS |
| Evaluator, NATIVE, R22, R25, one episode each | PASS | **FAIL in the counted run**; rerun finished |
| R52, R55 | NOT RUN | NOT RUN |
| Native parity: 4 suites, 10 tasks, 10 episodes | NOT RUN | NOT RUN |
| Per-task successes, videos | NOT RUN | NOT RUN |
| Action units | NOT RUN | NOT RUN |
| First-action hashes | PARTIAL: one state, tick 0 | NOT RUN |
| Profiling, 100 + 100 calls | NOT RUN | NOT RUN |
| CUDA-event time, full cost trace | NOT RUN | NOT RUN; JAX has no CUDA-event equivalent (R5) |
| Reset reuse across conditions | PARTIAL: repeated reset exact on one task, CPU | NOT RUN |
| Generated initial states | NOT RUN | NOT RUN |

**pi0.5 refinement gate.** `unsupported_refinement_setting` has not fired, and the gate has not passed. Source shows `dt = -1.0 / num_steps` from t=1. Outputs were finite. Action units are unverified. The wrapper that sets `num_steps` and noise per request needs Alba's ruling (R4).

## Corrections made

| Item | Change |
|---|---|
| `review/GPU_PREFLIGHT_RESULT.md` | Sections 1 and 7 corrected; the withdrawn verdict is named once as withdrawn |
| `review/CROSS_POLICY_CONFIRMATION_AUDIT.md` | Status line and preflight update corrected |
| `ALBA_RTX5090_PREFLIGHT_RESULT_CORRECTED.pdf` | New PDF from the corrected report (ZIP only) |
| Raw logs and evidence files | **Unchanged** |
| `review/GPU_PREFLIGHT_PREREG.md` | **Unchanged.** It lists all possible verdicts, and is the frozen rule |

**Superseded and not to be used:** `ALBA_RTX5090_PREFLIGHT_RESULT.pdf` (sha256 `32c6f25b6a931cf209db0368f5b6d109165ee16a3928796e6bb615f5bd1bf446`) and `ALBA_RTX5090_PREFLIGHT_EVIDENCE.zip` (sha256 `719f2b4cc78d48d7ce85e808f8cfd80de955ad5eeba19d33c7f4214107897022`). We have no record of either being sent. Their wording recorded a pass with narrowing and said a reply met Alba's condition. The first version of the report is preserved in git at commit `734dec3`.

## Known code defects to fix before any rollout

These come from reading the code. None has been fixed.

1. A reset-state mismatch is recorded in `settled_state_matches_block`, but nothing stops the attempt, and `analyze.py` does not read the flag.
2. `launch_guard.py` binds the release to the protocol hash only, and does not bind the schedule.
3. `analyze.py` does not output paired state-level differences, arithmetic per chunk, step and episode, or separated timing.
4. `generate_states.py`, the 400-episode parity paths, real journals, video writers and `provenance.py` have never been run.
