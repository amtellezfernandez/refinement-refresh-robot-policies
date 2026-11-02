# Reconciliation of Alba's PR #1, 2 November 2026

Local branch `reconcile/alba-pr1-20261102`. **Nothing is pushed.** GitHub's `warren/cross-policy-confirmation` and PR #1 are unchanged. Both parents are preserved in a real merge commit; no history was rewritten.

## Who took what

| Area | Taken from | Result |
|---|---|---|
| R22/R25 route | **Alba** | Primary schedule derived from frozen `cross_policy_confirmation_v1` by `make_primary_schedule.py`: 800 per model, 1,600 in total. **Our separate `matched_r22_r25_confirmation_v1` cohort is retired.** It had hashed its own order, and is removed by a new commit |
| Release format and schedule verification | **Alba** | `PRIMARY_R22_R25_ONLY`, approved conditions and attempts, per-model row and file hashes |
| Reset check before inference | **Alba's scope, our enforcement** | Settled state and first observation (hers), with the reference rebuilt from the journal after a restart (ours) |
| SmolVLA AMP, determinism and TF32 settings | **Alba** | Applied to attempts, preflight and the conditions check |
| Analysis | **Both** | Her flag requirement and `not_planned`; our independent recomputation of block hashes, paired differences and timing |
| d attestation | **Both** | Her echo; our value observed at the sampler call; client rejects either mismatch |
| CI workflow and environment check | **Alba** | Kept. One line of the check changed (below) |
| Fresh pi0.5 environment per attempt | **Ours** | Needed: reuse breaks exact reset on `libero_object` |
| Loaded normalization check, server provenance, package list without pip | **Ours** | Kept |
| Stop on failed attempt | **Ours** | Kept, pending R10. Her code journals and continues |
| Versioned entry point, verdict tool, video check, five-condition check | **Ours** | Kept, with N2 config and her check added at the start |
| Deferred cohorts | **Alba's decision 11** | Observation age and asynchronous refused, release or not |

## Scientific decisions found in code

| Item | Treatment |
|---|---|
| Parity threshold | None encoded. Verdict tool stops at "awaiting review" |
| OSMesa | Only implemented renderer. Gate refuses any other value of R6 until built. **This limits the options; it does not decide them** |
| States generated in the SmolVLA stack | Entry point's current behaviour. **N1 unresolved; may need to change** |
| Wrapper | Still in use. **R4 unresolved** |
| Bootstrap | Unchanged percentile method. Her required calibrated method is not implemented |
| Stop on failure vs journal and continue | Engineering default with a scientific effect on completeness. **R10 unresolved** |
| Seeds, conditions, identities, schedules | Unchanged |

## Conflicts left open

1. **R9, R6, N1, R4:** Alba's.
2. **Release path for preflight:** `cross_policy_preflight_v2.release.json` is our proposed format. Her document specifies only the confirmation release.
3. **Her check does not catch the old config.** With `LIBERO_CONFIG_PATH` at the SmolVLA environment, it still reports ready, though `bddl_directory` is under `python3.12` site-packages. Her sys.path rule does not inspect config paths. The entry point avoids this, but the check is unchanged.

## Frozen boundary

| Check | Result |
|---|---|
| `run_gpu_preflight.sh` | Blob `6ae6c11acdc4aad49cdfda0ea2b9fee3c8302dcc`, identical to `bb86691` (the counted run) and to PR #1. A test pins it |
| 1 November evidence, prereg | No difference from the commits that recorded them |
| Frozen protocols and schedules, `release/`, `paper/`, `run_paired.py` | No difference from `6ad02ec` |
| Counted verdict | Still PREFLIGHT_FAILED |

Commit `31748c8`, which patched the script, stays in history and on GitHub until this branch is pushed. Commit `ffc2261` restores it, and also contains a harness test with a stub bug, fixed in the merge.

## CPU diagnostics

All on the RTX, CPU only, `--vram 0`, no weights and no policy; GPU at its idle 720 MiB afterwards. Scripts are in the job records.

| Job | Finding |
|---|---|
| `alba-v2-probe-7a` | Her check, verbatim: `TypeError: expected str, bytes or os.PathLike object, not NoneType`. Cause: `Path(libero.__file__)`, where `libero` is a namespace package and `__file__` is `None`. With `libero.libero.__file__`: `READY_FOR_CLIENT_IMPORT_CHECK_ONLY`, module under OpenPI's tree. Wrong root is still refused; no `PYTHONPATH` is still refused |
| `alba-v2-probe-7b` | Pinned OpenPI `215abfb`, `Pi0Config(pi05=True)` with dummy variants, JIT disabled: `action_out_proj` called 2, 5 and 10 times at d=2, 5, 10; chunks differ. Direct model call, random weights |
| `alba-v2-probe-7c` | First-observation hash identical over 3 fresh environments on 4 pi0.5 client tasks, and 3 hard resets on 2 SmolVLA tasks |
| `alba-v2-probe-6` | Client on OpenPI's tree: 40 of 40 tasks; settled state and images identical to the old config |

## Follow-up on local branch `local/preflight-v2-engineering` (after the reconciliation branch)

| Item | Change | Evidence | Not established |
|---|---|---|---|
| Real-weight denoising count | `server_openpi.count_denoise_loop` runs `Policy.infer` uncompiled with the loaded model, counting `action_out_proj` calls at d=2, 5, 10. `profile` records `denoise_loop_calls` and, without a gate, how far uncompiled chunks differ from compiled ones. `preflight_verdict.py` fails on a wrong or missing count | Tests on a fake model, including one that ignores `num_steps`. The counting method itself ran on pinned `Pi0` with a dummy model on the RTX (`alba-v2-probe-7b`) | **Never run on the real `Policy` class or real weights.** That needs the checkpoint, which is on the RTX. Uncompiled run time is unknown |
| Environment false positive | `check_openpi_client_environment.py` now requires `benchmark_root`, `bddl_files`, `init_states` and `assets` under the pinned root, and refuses another interpreter's paths even without a pinned root | Tests with the real check on a fake LIBERO; reproduced the false positive first | **Not re-run in the real client environment.** Assumes `get_libero_path("benchmark_root")` works; config has that key |

Both change Alba's check or our harness only. No frozen protocol or schedule changes, and neither decides R9, R6, N1 or R4.

## Limits

- No fix has run with real weights, the server, the GPU or the new code on the RTX, whose bundle is still `code-bb86691`.
- The SmolVLA changes, including AMP and `Counters` in rollouts, cannot run here without torch and LeRobot. A static undefined-name check is clean.
- The loop count shows pinned code honours `num_steps`. It does not observe the real checkpoint's run through `Policy.infer`.
- The AMP, determinism and first-observation behaviour is first exercised by the preflight.
