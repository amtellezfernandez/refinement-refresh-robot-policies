# cross_policy_v3: Protocol B runner, gates, evaluator harness, schedule and analysis (DRAFT, NEVER RUN)

Implementation of protocol v3 revision 2 with Alba's decisions D1–D8 (`review/preflight_v2/protocol_v3_revision_20261106/`, especially `02`, `09`, `12`, `13`).

**Status.** Code only. Nothing here has been run on a simulator, a model, a GPU, a fixture, or any real or reserved identity. Every item in the qualification records is UNVERIFIED. This code existing is not evidence that any check passes.

**Frozen files are not edited.** Everything here is new. The v2 modules in `../cross_policy/` (`common`, `journal`, `attempt_loop`, `launch_guard`, `client_openpi`, `server_openpi`, `runner_smolvla`) and `../run_paired.py` are imported unchanged.

## Files

| File | Python | What it does | Needs a stack? |
|---|---|---|---|
| `v3common.py` | 3.8+ | constants, the population object, paths | no |
| `make_schedule_v2.py` | 3.8+ | reads one model-native state manifest; emits that model's 800 rows (400 R22 + 400 R25) | no |
| `classify.py` | 3.8+ | attempt classes (02 B5) and per-row final class after retries (02 B7) | no |
| `runner_v2.py` | 3.8+ | Protocol B runner core and CLI: G0-B, PRE, journal, retries, stops, summary | the adapter does |
| `adapter_pi05.py` | 3.8 | pi0.5 client attempts: fresh env, attestation verified before any action is applied | OpenPI LIBERO client stack |
| `adapter_smolvla.py` | 3.12 | SmolVLA attempts: fresh env, D7 classification, count per generation | SmolVLA stack + GPU |
| `attestation.py` | 3.8+ | per-request attestation build and verification (09 C4) | no |
| `server_openpi_v3.py` | 3.11 | attesting pi0.5 server: instrumentation, startup G02 gate on the served object, CNT-IND, PRES, time grid, CNT-T overhead | OpenPI + JAX + GPU |
| `evaluator_validation.py` | 3.8+/3.12 | Protocol E harness: own BDDL goal parser, independent predicate reading, comparison, verdict | each stack |
| `analysis_v2.py` | 3.8+ | B6: exact contrast, four counts, bounds over 400, missingness by arm and suite, verdict, secondary interval | no |
| `r3_coverage.py` | 3.8+ | R3 coverage check for the secondary interval (synthetic); refuses without authorisation | no |
| `qualification.py` | 3.8+ | qualification-record validator and the PRE gate; has no function that marks anything PASSED | no |
| `release.py` | 3.8+ | G0-B release check (reuses frozen `verify_released_schedule`) and authorisation check | no |
| `test_v3.py` | 3.8+ | offline unit tests on synthetic data | no |
| `../../protocols/v3_qualification/{pi05,smolvla}.json` | – | qualification records, all UNVERIFIED | – |

## Requirements traceability

| Requirement | Code | Unit-tested offline? |
|---|---|---|
| A. Read each model-native manifest; fail closed when it is malformed | `make_schedule_v2.validate_manifest` | yes (8 malformed cases) |
| A. Exactly 800 rows: 400 R22, 400 R25, paired by identity, provenance preserved | `make_schedule_v2.build_schedule`, `validate_schedule` | yes (test population) |
| A. Uniqueness, coverage, balance, disjointness from other populations | `validate_schedule`, `check_disjoint` | yes |
| A. No real states generated | no state generator here; model-keyed `generate_states.py` still owed | n/a |
| B. Fresh environment per attempt | `adapter_pi05.run_attempt`, `adapter_smolvla.run_attempt` | no (needs simulator) |
| B. Explicit model-native stack configuration | `adapter_*.stack_config`, journaled on every attempt | no |
| B. Retry and stopping semantics | `runner_v2.execute`, `classify.final_class` | yes (fake adapter) |
| B. Scientific vs infrastructure failures | `classify.classify_outcome`, `classify_exception`, first-query marker | yes |
| B. Reserved identity boundary | `release.check_release` (Alba's release only); Protocol E fixtures refuse seeds and slots | partly (fixture rule) |
| B. No attempt silently skipped | `runner_v2.execute` loop; `summarize` refuses COMPLETED with a gap | yes |
| B. Action provenance and termination | `action_provenance` and `outcome.termination_reason` per attempt | no |
| C. Startup G02 integration | `server_openpi_v3.startup_gate` on the served object; port bound only after it passes | no (needs GPU) |
| C. Per-request attestation, bound to request and tick | `server_openpi_v3.AttestingPolicy`, `attestation.build` | build/verify yes; server no |
| C. Fail closed before action application | `adapter_pi05._rollout` verifies before planning any action | verifier yes; adapter no |
| C. Independent verification | CNT-IND in `startup_gate`; time-grid channel per request | grid arithmetic yes; GPU no |
| C. No fake success attestations | counts come only from callbacks; verifier tests reject consistent forgeries | yes (verifier) |
| C. Action-output preservation, overhead | PRES in `startup_gate`; `server_openpi_v3.overhead` | no |
| D. Model-free fixtures, three cases, independent BDDL inspection, comparison, π0.5 done semantics | `evaluator_validation.*` | parser, comparison, verdict, fixture rules yes; drivers no |
| E. Exact contrast, four counts, complete-pair estimate, bounds over 400, missingness by arm and suite, thresholds | `analysis_v2.analyze` | yes |
| E. Secondary task-weighted fresh-state interval | `analysis_v2.secondary_interval`, `stratified_interval` | arithmetic yes; coverage no |
| E. R3 coverage check | `r3_coverage.run_check`; result hash-bound in `check_r3_record` | **not run** (needs authorisation) |
| E. No post-hoc changes | thresholds are constants; the R3 record is bound to the analysis file hash | partly |
| F. Qualification records, UNVERIFIED until checks pass | `qualification.py`, `v3_qualification/*.json` | yes |

## Known limits of this code (unverified assumptions)

- **Trace-time instrumentation.** `server_openpi_v3` decides which compiles carry callbacks from a flag read at trace time. That relies on each `module_jit` producing its own trace, as the G02 prototype did. It has not been run.
- **Callback ordering.** The time-grid check compares multisets, so it does not depend on callback order. The count check does not depend on it either.
- **Time recurrence.** The pinned stock `pi0.py` recurrence and stopping condition have now been read independently of G02; see `STOCK_SAMPLER_SOURCE_AUDIT.md` for the commit, source hash and exact lines. Float32 assumptions, compiled callbacks/counting and numerical parity still require native verification.
- **LeRobot and LIBERO internals.** Attributes such as `base._env.env`, `parsed_problem`, `object_states_dict` and `_max_episode_steps` were read from a local hf-libero 0.1.4 copy. That copy is not the planned host's stack, and the OpenPI submodule's copy was not read.
- **Fresh-process retry.** `launch_v3.py` relaunches `runner_v2.py` with identical arguments on exit code 75 (cap 800 relaunches; other exit codes pass through). It is in `release.RUNTIME_FILES`, so a release binds it. Not run natively.
- **B904 style.** `ruff` flags re-raises without `from`, matching the house style of the frozen modules.
- **Second whole-schedule run.** It is not detected across output directories. `launches.jsonl` records relaunches in one directory and is not read by the analysis.
- **Attestation digest.** It detects corruption, not forgery: the server computes it. Gaps in sequence numbers are allowed. The order of time values is not checked.
- **Checks not written.** R9 parity, P2a preprocessing, SRC, UNITS, the SmolVLA normalization and action-mapping checks, ISO for Protocol A, and the CNT-T flags-on/off comparison have no code yet. Only qualification placeholders exist for them.
- **Release format.** The release must carry `v3_code_sha256` (the hash of every file in `release.RUNTIME_FILES`) and `qualification_sha256_by_model`. This is our proposal; Alba has not seen it.

## Independent review (one round, AI reviewer, read-only, nothing run)

17 findings. Addressed in code or docs:
- qualification evidence (1);
- code and qualification hashes bound in the release (2);
- the client reads the startup gate record and requires GATE_OK (3);
- a close failure can no longer erase an outcome (4);
- a malformed result is an instrument incident (5);
- input validity uses the input signature bound at Protocol A (6);
- the R3 pass is recomputed from the cells and bound to an authorisation (7);
- the SmolVLA fixtures step the single env, avoiding autoreset (8; LeRobot source not read, so the risk is avoided rather than verified);
- the π0.5 `done` claim is narrowed to the client horizon, and the ISO manifests are mandatory (9);
- the overhead mode records first-call time and memory, and the docs now say the flags comparison is a separate unwritten tool (10);
- accounting by arm and suite (11);
- HALTED is listed under UNINTERPRETABLE (12);
- the journal checksums are checked by the analysis (13);
- an unused helper was removed (14);
- the preallocate binding nothing asked for was removed (16).

Not addressed, and listed as limits above: 12 (whole-schedule rerun detection), 15 (attestation minor points) and 17 (checks not written).


## BenchStudio evidence handoff

Protocol E can optionally emit the BenchStudio `evaluator.json` consumed by
the v3 launch gate. Pass `--benchstudio-runtime` with a separately captured
native runtime identity mapping. Bind that file to the Protocol E authorization
as `benchstudio_runtime`. The exporter retains the native samples and the
Protocol E result digest, and refuses incomplete or unqualified fixture sets.
Supply one fixture per case and suite. Goal-reached traces must end normally
or continue to the declared horizon. Keep the Protocol E record and fixture
evidence for independent review before adding the exported file to a BenchStudio
qualification bundle.

This handoff uses JSON so the OpenPI Python 3.8 evaluator does not import
BenchStudio. Native qualification remains UNVERIFIED.


## Check Protocol E inputs before booking the native host

Run the following separately for `pi05` and `smolvla`:

```sh
python experiments/tooling/cross_policy_v3/evaluator_validation.py \
  --check-inputs --stack pi05 --fixtures PATH_TO_FIXTURES \
  --confirmation-manifest PATH_TO_PI05_MANIFEST \
  --confirmation-manifest PATH_TO_SMOLVLA_MANIFEST \
  --benchstudio-runtime PATH_TO_CAPTURED_RUNTIME_JSON
```

This imports neither native stack, writes no authorization, and emits the
missing cases/manifests plus exact file digests for review. Both confirmation
manifests are required for ISO before native execution, but this command can
report missing inputs without generating them. `--benchstudio-runtime` is
optional unless a BenchStudio evaluator export is needed.

Only after separate authorization, replace `--check-inputs` with
`--authorisation APPROVAL_JSON --output NEW_DIRECTORY` on the matching
pinned stack. The authorization must include the two ISO manifest hashes
under `confirmation_manifest_pi05` and `confirmation_manifest_smolvla`.
With the runtime argument, a qualifying run also emits `evaluator.json`.
