**Superseded by `review/preflight_v2/`, after Alba's decisions. Kept as a record.**

# Next preflight readiness, 2 November 2026

**Verdict: NOT READY for the full preflight.** Engineering that could be done alone is done and unit-tested. The remaining blockers are Alba's decisions, a GPU release and a code transfer. No model rollout, GPU job or RTX write took place in this work.

Base commit: pushed `6ad02ec`. Fixes: `31748c8`, `24288bd`, `caebdd9`. See `ENGINEERING_FIXES.md`.

## Dependency-ordered checklist

Class: **A** engineering; **B** Alba; **C** RTX execution; **D** waits for successful preflight.

| # | Requirement | Class | Status | Depends on |
|---|---|---|---|---|
| 1 | Fix client LIBERO path, with regression test | A | DONE | |
| 2 | Enforce reset-state reuse; bind schedule to release; stop on failure | A | DONE | |
| 3 | Full preflight entry point; server provenance; five-condition check | A | DONE, not run | |
| 4 | Decide R9, R6, N1, N2, R4, R12 | B | OPEN | |
| 5 | Invalid-campaign reason and shared-defect question | B | OPEN | |
| 6 | Revise scripts for R6 and N1, if the rulings require | A | NOT STARTED | 4 |
| 7 | Freeze a full-preflight prereg with R9's criterion, including verdict rule and counted-run rule | A + B | NOT STARTED | 4, 6 |
| 8 | Write Alba's `full_preflight.release.json` | B | NOT STARTED | 7 |
| 9 | Warren's approval to push; transfer the bundle to the RTX | Warren | NOT STARTED | 7 |
| 10 | GPU release and exclusive reservation | Warren | NOT STARTED | 9 |
| 11 | Compare OpenPI and SmolVLA LIBERO asset trees (CPU, read-only) | C | NOT STARTED | 9 |
| 12 | Run full preflight, about 6-26 h | C | NOT STARTED | 8, 10 |
| 13 | Review preflight evidence: parity, profiling, videos, conditions, pi0.5 gate | B | NOT STARTED | 12 |
| 14 | Decide R7, R2 | B | OPEN | 4 |
| 15 | Run `generate_states.py`; bind hashes; rebuild schedules; commit | C + A | NOT STARTED | 14, N1 |
| 16 | Decide N3, N4, R3, R10, R13, R5, N5, N6, R8 | B | OPEN | 13 |
| 17 | Fix analysis for R13; implement N4 if requested | A | NOT STARTED | 16 |
| 18 | Protocol and schedule checkpoint committed and pushed | A + Warren | NOT STARTED | 15, 17 |
| 19 | Alba's protocol release | B | NOT STARTED | 18 |
| 20 | Execute R22/R25 or full confirmation | D | NOT STARTED | 19 |
| 21 | Independently validate | D | NOT STARTED | 20 |
| 22 | Observation age (R16) | D | NOT STARTED | 20 |
| 23 | Asynchronous (R1) | D | BLOCKED | 22 |

Items 4 and 5 can run in parallel. Whether step 7's prereg needs Alba's approval beyond step 8 is for her to say.

## Interface check

| Need | SmolVLA | pi0.5 | State |
|---|---|---|---|
| Native parity | `runner_smolvla.py preflight` | `client_openpi.py parity` | Written; never run; **no pass criterion (R9)** |
| 100 + 100 profiling | `runner_smolvla.py preflight` | `server_openpi.py profile` | Written; never run |
| Videos | `imageio` in parity | `imageio` in parity | Written; never run |
| R52, R55 | `runner_smolvla.py conditions` | `client_openpi.py conditions` | New; never run |
| Reset reuse across conditions | `conditions`; enforced in attempts | Same | Logic tested; never run on a simulator |
| Initial states | `generate_states.py` | Same output | Never run; R7, N1 |
| State hashing | Per-row check | Per-row check | Tested with fake rollout, pi0.5 path |
| Official interface | Pinned LeRobot; source hash check | `create_trained_policy` + `WebsocketPolicyServer`, plus wrapper | **R4 open** |
| CUDA-event time | In profile and now attempts | Unavailable | R5 |
| Arithmetic | Calibration exists | None | R13 |

## Blocked by stop conditions

| Item | Stop condition |
|---|---|
| Running `generate_states.py` | R7 and N1 unresolved |
| Choosing a parity threshold | R9 is Alba's |
| Arithmetic for pi0.5 | R13 unresolved |
| R22/R25-only execution path | N3 unresolved; a subset changes the schedule |
| Independent outcome verification | N4; not in the frozen protocol |
| Validity of SmolVLA figures | Invalidation reason unavailable |

## Shared instrumentation audit (prepared)

**No cause is asserted here, and historical data is not certified.** Alba's reason selects which row to run.

`run_paired.py` as committed loads only `SmolVLAPolicy` and requires h=1, d=10, chunk 50. So its model loading was not shared with a pi0.5 campaign in this form. What ran that campaign is unknown.

| Surface | Shared with a LeRobot pi0.5 path? | What could go wrong | Check once the reason is known | Available now |
|---|---|---|---|---|
| `LiberoEnv` camera, relative control, settling, init state | Plausibly | Wrong orientation, control mode or settle count | Read pinned `envs/libero.py`; compare frames and first actions with OpenPI's example | Upstream parity below |
| Environment pre/post processors | Plausibly | State or action conversion | Compare one state vector and action with the example | Not checked |
| Normalization processors | Framework shared; modes differ | SmolVLA checkpoint uses MEAN_STD; pinned pi0.5 config defaults to QUANTILES | Check `policy_preprocessor` stats against checkpoint | Manifests agree on MEAN_STD |
| Denoising setting | Attribute differs | SmolVLA reads `config.num_steps`; pinned pi0.5 reads `num_inference_steps`, so a harness setting one could silently miss the other | Compare counted denoise steps with d | **Passes for SmolVLA journals** |
| `n_action_steps` and queue | Plausibly | Wrong prefix | Compare generations with steps and h | **Passes for SmolVLA journals** |
| Noise shape and injection | Plausibly | `chunk_size`, `max_action_dim` mismatch | Re-derive the shape from config | Manifests: 50, 32 |
| Image size | Plausibly | 360 evaluated; pi0.5 expects its own resolution | Compare preprocessed tensors | Not checked |
| Journals and analysis | Plausibly | Recording or success defect | Fresh replay of recorded actions | Not checked. The archive has no `.npy` arrays, only `steps.jsonl` |

**Result of `audit_shared_instrumentation.py` on `confirmation_completed.tar.gz`:** 1,122 records, 0 violations of: `generations = ceil(steps / h)`; `denoise_steps = generations x d`; `action_expert_calls = denoise_steps`; `prefill_calls = generations`; `vision_encoder_calls = 2 x generations`; steps within horizon; failures only at horizon. Horizons: Object 280, Goal 300.

**This is self-consistency only.** The counters come from the instrument under question. The result rules out neither a shared defect nor the reason Alba has in mind.

`development_parity_001` records one native episode matching the pinned upstream evaluator, which is also LeRobot code.

## Environment

| Item | State |
|---|---|
| Local tests | 51 OK at `caebdd9` |
| RTX 5090 | Idle at last poll: 0.7 of 31.8 GiB, no job running or queued |
| RTX environments | Built 1 November; not re-inspected directly, because direct ssh is gate-blocked |
| RTX code | `code-bb86691`, which **lacks all these fixes** |
| GPU release | D739 covered the bounded preflight only. A new release is needed |
| Memory | Bounded run peak 9.3 GiB VRAM, 16.0 GiB RAM for pi0.5 |
