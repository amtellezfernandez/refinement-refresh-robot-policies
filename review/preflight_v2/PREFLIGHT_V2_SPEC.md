# Versioned preflight specification

Protocol `experiments/protocols/cross_policy_preflight_v2.json`. Run ID `cross_policy_preflight_v2.run001`. Entry point `experiments/tooling/cross_policy/run_preflight_v2.sh`.

**Status: PLANNED. Not executed. Not authorised until Alba's release exists.** The 1 November preflight stays PREFLIGHT_FAILED, and its records are not touched.

## Stages

Frozen before any stage below ran. **CPU stage** `cross_policy_preflight_v2.run001.cpu`: no GPU, no reservation, no rollout with a policy. **GPU stage** `cross_policy_preflight_v2.run001`: not started; needs a GPU release. Until both finish, the verdict is PREFLIGHT_INCOMPLETE. CPU checks that pass do not make the preflight pass.

## Question

Are both pinned models, through their evaluation paths on this host, fit to run the matched R22 vs R25 comparison?

## Pinned artifacts

The commit containing this file; SmolVLA `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, `model.safetensors` `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`; pi0.5 `gs://openpi-assets/checkpoints/pi05_libero/`, normalization `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`; OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`; LeRobot `b6ec0060779550c0a157ae34feb89e0cf86012a8`; environments as recorded in `preflight_cpu_evidence/*.lock.txt`.

## Checks

| Required check | How | Enforced or evaluated in |
|---|---|---|
| Checkpoint provenance | Checkpoint tree hash, revision marker | `provenance.py`, `run_paired.py`; `check_checkpoint` |
| Dependency provenance | Installed packages from `importlib.metadata`, since the environments have no pip | `provenance.py`; `check_dependency` |
| Official pi0.5 normalization | File hash, then loaded `Unnormalize` statistics equal the file, quantile mode on | `server_openpi.py`; `check_normalization` |
| Native evaluator parity | 4 suites x 10 tasks x 10 episodes, per model | `runner_smolvla.py preflight`, `client_openpi.py parity`; `check_parity`. **Criterion: R9** |
| Action units and shapes | 7 dimensions; per-dimension ranges per episode | `check_actions`. **Unit acceptance: R9** |
| Implementation received d=2, 5, 10 | SmolVLA denoise count equals generations x d. pi0.5 `num_steps` observed at the sampler call, and same noise gives different chunks across d. Alba's echo also checked; an echo alone is not proof. Pinned source's loop count measured on a dummy model: 2, 5, 10 (`alba-v2-probe-7b`). **Not yet measured with real weights** | Both rollouts; `check_received` |
| Matched-reset first-action hashes | R22 = R52 and R25 = R55 | `summarize_condition_check` |
| Reset reuse | One settled hash and one first-observation hash across all conditions, with a fresh pi0.5 environment per attempt | `summarize_condition_check`; `attempt_loop.py` |
| Client environment | Alba's check with the actual client interpreter, before GPU work | `check_openpi_client_environment.py`, called by the entry point |
| R22, R25, R52, R55 smoke | Plus NATIVE, once each, on a non-confirmation identity | `conditions` mode |
| Video inspection | Machine check of all, then a person inspects one contact sheet per task | `inspect_videos.py`; **human step** |
| Per-call and rollout timing | 100 + 100 per d; per-generation and per-rollout | `check_timing` |
| Generated-state hashes | 200 freshly generated states with SHA-256 (Alba, PR #1) | `generate_states.py`; `check_states`. **Ruling: N1** |
| Reset-mismatch termination | A wrong reference on the real simulator stops before any policy query | `conditions` mode negative control |
| Protocol and schedule hash binding | Release bound to protocol hash; confirmation needs Alba's PRIMARY_R22_R25_ONLY release with per-model row and file hashes; deferred cohorts refused | `launch_guard.py`; `check_binding` |

## Authored values, not measured costs

Call counts, episode counts, seeds 790011, 790012 (smoke), 790001, 790002 (parity), state `official0`, 3 CPU threads per process, 3,072 MiB G1 threshold, one contact sheet per task.

## Negative controls

| # | Detects | Method |
|---|---|---|
| 1 | Reset check cannot fire | Wrong reference on the real simulator must raise before any policy query |
| 2 | d does not reach the integrator | Same input and noise at d=2, 5, 10 must give different chunks |
| 3 | Pairing or noise broken | R22 = R52 and R25 = R55 first actions must match exactly |
| 4 | Blank or frozen video | Machine check, tested on planted frames |
| 5 | Verdict tool passes missing evidence | Tested: each required file removed or altered gives PREFLIGHT_FAILED |

Control 3 could fail through GPU non-determinism rather than a pairing fault. Such a result is reported as a failure, and its cause is investigated afterwards under a new run ID.

## Standing and strongest claims

Confirming results are expected, and show that the harness behaves as written on this host.

- **If it passes:** "Both models ran through their evaluation paths on this host, with these checks passing under Alba's criterion." Nothing about R22 vs R25, generality beyond the host, or the historical SmolVLA results.
- **If it fails:** "The named check failed in the counted run."
- **Not claimed either way:** SmolVLA instrument validity, or any effect of denoising. Parity success rates are not confirmation evidence.

```selection-and-gates
counted_run: the first execution of run_preflight_v2.sh that passes G0 to G2, run ID cross_policy_preflight_v2.run001
rerun_policy: none for this run ID; a failure stays failed, and a further attempt needs Alba's agreement, a new run ID and a recorded reason
post_data_free: NONE for mechanical checks; the parity criterion is fixed in the release before the run; the human video inspection is a judgement, recorded with inspector and date
gate: G0 release | Alba's release matches the protocol hash and records every decision | enforced_at: experiments/tooling/cross_policy/run_preflight_v2.sh:17 | before: all GPU work | qualified_on: complete release -> pass; missing decision -> fail
gate: G2 lock | gpu.lock is free | enforced_at: experiments/tooling/cross_policy/run_preflight_v2.sh:28 | before: all GPU work | qualified_on: free lock -> pass; held lock -> fail
gate: G1 quiet | under 3072 MiB in use | enforced_at: experiments/tooling/cross_policy/run_preflight_v2.sh:30 | before: all GPU work | qualified_on: 719 MiB -> pass; 19052 MiB -> fail
selection_risk: low: no scientific estimate is drawn; identities are outside the confirmation population
stop_on_confirmation: stop, send Alba the commit, verdict, manifest and checksums, and take no confirmation step until her release
verdicts: PREFLIGHT_PASSED | PREFLIGHT_FAILED | PREFLIGHT_INCOMPLETE | UNINTERPRETABLE
review_rounds_max: 1
```

UNINTERPRETABLE applies if the host fails mid-run in a way that leaves no attributable result. It does not convert to a pass, and the attempt counts as used.

## Files outside this repository

Run outputs stay on the RTX until collected. Videos are not committed unless Alba asks.

See `DECISION_REGISTER.md` for what remains open.
