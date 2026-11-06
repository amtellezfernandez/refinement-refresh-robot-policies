# cross_policy_preflight_v3 (engineering preflight), revision 2: DRAFT / NOT RELEASED

> **DRAFT, NOT RELEASED.** Revision 2 of the unreleased v3 draft. It implements Alba's binding decisions on R6, R9, R4, N1 and CNT and the scientific design, received 6 November 2026 and relayed by Warren. It supersedes `../diagnostic_recovery_20261105/PROTOCOL_V3_DRAFT.md` (revision 1), which is kept unchanged as the record.
>
> **Nothing here authorises execution.** That covers GPU preflight, parity, confirmation, state generation, and use of any reserved identity. v2 (`experiments/protocols/cross_policy_preflight_v2.json`, `4fc520ec...`) and `cross_policy_confirmation_v1.json` (`9310639c...`) stay frozen and unchanged.

This document is **Protocol A: the engineering preflight**. The scientific confirmation is a separate protocol, **Protocol B** (`02_SCIENTIFIC_CONFIRMATION.md`). It has its own scope and is **not authorised for execution**. A passed Protocol A licenses only a request to Alba for a Protocol B release. It does not licence any claim about outcomes.

## A1. Question

Do the instruments, execution paths and mechanics of both models behave as specified on each model's own pinned host and toolchain? The checks use synthetic fixed actions, fresh environments and frozen diagnostic observations only.

The preflight answers **no** question about policy quality, success rates, or any effect of d or h.

## A2. Scope rules (binding)

| Rule | Content |
|---|---|
| No confirmation identities | No preflight step may read, request, generate or roll out an identity, state slot or policy seed from the Protocol B population. This covers the seeds 810001 and 810002, and any slot in either model's confirmation state manifest. A request for one aborts the run (A8) |
| No outcome-quality claims | No preflight step computes, prints or stores a task-success value. Rollout steps, where they exist, use synthetic fixed actions with no policy in the loop. No policy-driven episode is run |
| Fresh environments | Every simulator check builds a fresh environment. A reused environment is history-dependent (E01–E03) and is used only as the named negative control in A5 |
| Diagnostic-only observations | Policy inputs are the two synthetic counter cases and three frozen diagnostic observations of `06_DIAGNOSTIC_POPULATION.md`. All five are excluded from confirmation |
| Per model, per host | Each model's checks run on, and are reported for, that model's own pinned stack (`07_HOST_GAPS_REGISTER.md`). No result transfers between models or hosts |

**Removed from v2's preflight** (see `03_V2_TO_V3_DESIGN_DIFF.md`):
- the R22/R25/R52/R55 smoke checks;
- the R22 = R52 and R25 = R55 first-action hash identities;
- the 4 × 10 × 10 native-parity episodes per model, which read success rates;
- the per-task contact-sheet inspection of those episodes.

## A3. Pinned identities (verify before and after; any change aborts)

| Object | Identity | Status |
|---|---|---|
| Repository base | published base `ba14f26` (`warren/pi05-diagnostic-recovery-20261106`). The local working commit this draft was written on is unavailable on this branch. Frozen v2 files 27/27 match `run001_cpu/SHA256SUMS` on `ba14f26` (checked 6 Nov) | pinned |
| pi0.5 checkpoint | `gs://openpi-assets/checkpoints/pi05_libero/`, 16 files, hashes as P0. `norm_stats.json` `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` | pinned |
| SmolVLA checkpoint | revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, `model.safetensors` `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` | pinned |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479`, clean | pinned |
| Per-model stacks | `07_HOST_GAPS_REGISTER.md`. Every entry marked UNKNOWN there must be resolved, or carried as a declared gap, before release | **partly UNKNOWN** |
| Frozen tools | `server_openpi.py` `c3eca9a4...`, `client_openpi.py` `767cad39...`, `common.py` `afccd5e9...`, `launch_guard.py` `718edbd1...` | pinned |
| New tools (not written) | `run_preflight_v3.sh`, `count_compiled_pi05.py`, `parity_stock_reference.py`, `make_diagnostic_population.py`, a success-free mechanics harness (CHUNK, ENV, RESET, UNITS), and the Protocol B schedule and state generators. Each is hashed into this table before release | **absent from the repository and the MacBook; RTX workspace UNSEARCHED** |
| Diagnostic population | five items, hashes bound in `06_DIAGNOSTIC_POPULATION.md` once generated | **not generated** |

## A4. Execution paths

- **pi0.5 candidate path:** stock `create_trained_policy(pi05_libero)` + `KeyedPolicy` (documented `num_steps` and `noise` only), served compiled.
  - `KeyedPolicy` is a **candidate** for the official path, **not approved** (R4; see `05_PARITY_AND_ACTION_UNITS.md` §P5).
- **pi0.5 independent reference:** stock OpenPI `Policy.infer` with no wrapper, no server and no websocket, in a separate process (`05` §P1).
- **pi0.5 XLA flags:** `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"` for the candidate and for the reference.
  - Their timing cost is measured in A5 (CNT-T) before they are adopted for Protocol B.
- **SmolVLA:** the v2 path (pinned LeRobot `SmolVLAPolicy` + `LiberoEnv`) with the `run_paired.py` determinism settings. Its count is as in v2: denoise steps = generations × d.

## A5. Checks

Order inside `run_preflight_v3.sh` (not yet written). Each gate must **branch** before the next budgeted step.

| # | Check | Inputs | Pass condition | Detects |
|---|---|---|---|---|
| G0 | Release | release file | Alba's release bound to this document's sha256 exists | running without authorisation |
| PROV | Provenance | none | every A3 hash matches; per-model stack lock files match the register | silent stack drift |
| ISO | Isolation | diagnostic manifest; both confirmation state manifests (when they exist) | no diagnostic state hash, seed or identity collides with either confirmation population | contamination of confirmation |
| NORM | pi0.5 normalization | `norm_stats.json` | file hash matches; loaded `Unnormalize` statistics equal the file; quantile mode on | wrong or unofficial normalization |
| CNT | pi0.5 compiled count (G02 design; `09_CNT_G02_REVIEW.md`) | SC1, SC2 | CNT-0..CNT-REP and CNT-IND at d = 2, 5, 10; in-run mutants rejected | wrong executed step count or step size |
| CNT-T | Deterministic-flag timing | SC1 | measured and recorded, no threshold. 100 warm-up + 100 measured compiled calls per d, with and without the flags, in separate processes | unmeasured cost of determinism (R5) |
| SCNT | SmolVLA count | SC-equivalent synthetic input | denoise steps = generations × d at d = 2, 5, 10 | wrong SmolVLA step count |
| R9 | pi0.5 action parity, stock vs candidate | DO1–DO3 at d=10 | `05` §P2: identical shape and dimension order, all finite, **bitwise-identical values** | wrapper changes the action path |
| SENS | d reaches the integrator | DO1 at d = 2, 5, 10, same noise | the three chunks are pairwise bitwise different | d ignored |
| CHUNK | Native chunk use at pi0.5's native setting (d=10, h=5). This is a preflight setting, not a Protocol B arm | the d=10 chunk for DO1 is saved by the R9 step as a **frozen array** and replayed as synthetic fixed actions. There is no live policy in the loop. Fresh env | the plan logic executes exactly chunk[0:5] in order, then issues its next request at tick 5. Checked by element hashes over 6 ticks. Needs a new hashed harness, because the frozen `client_openpi.rollout` computes `success` (line 127), which A8 forbids | wrong chunk slicing or replanning |
| ENV | Fresh-env determinism | S0, 20 synthetic fixed actions, 2 fresh envs per model stack | final sim states bitwise equal | non-deterministic fresh env |
| ENV-NC | Reused-env control | same, 1 reused env | must differ from fresh. If it does not, ENV is not shown able to fail | blind ENV check |
| RESET | Reset-mismatch termination | S0 with a wrong expected hash | attempt stops before any policy query | reset check cannot fire |
| UNITS | Action-unit response | fresh env, unit probes (E02 `U` design) | signs and axes match the declared convention in `05` §P3. Magnitudes are recorded, not gated | sign or axis error |

**Which checks apply to which model.**

| Model | Checks that apply |
|---|---|
| pi0.5 | G0, PROV, ISO, NORM, CNT, CNT-T, R9, SENS, CHUNK, ENV, ENV-NC, RESET, UNITS |
| SmolVLA | G0, PROV, ISO, SCNT, ENV, ENV-NC, RESET, plus **S-SRC**: a source read recording the SmolVLA success-signal path (`info["is_success"]`), whether success terminates the episode, and the per-suite `_max_episode_steps` |

A SmolVLA PREFLIGHT_PASS therefore certifies only those checks. It does not certify SmolVLA normalization, action units or evaluator parity, which v2 never established either. Whether that is sufficient to release SmolVLA into Protocol B is **Alba's decision** (cover note, item 7).

**Certified configuration = executed configuration.** Protocol B must run each model in the exact configuration Protocol A certified: the same hashes, the same flags, the same host and the same renderer. In particular:
- if Alba declines the deterministic XLA flags after CNT-T, Protocol A must be re-run for pi0.5 without them;
- R9's cross-process bitwise criterion is then **not expected to hold**, because D01 measured about 5e-3 drift under default flags;
- that case needs Alba's ruling **before** any run (cover note, item 8).

## A6. Authored values (not measured costs)

These values were chosen, not measured, and none is reported as a behavioural cost:
- the seeds 20261105 (SC1 and the fixed-action sequence), 20261106 (SC2), 20261108 (DO2 action sequence) and 7 (`env.seed`, stock);
- the 20 fixed actions, the 100 + 100 timing calls, the 2 fresh envs, and d ∈ {2, 5, 10};
- the noise key `["fixed-noise"]`, and the DO3 alternate prompt.

## A7. Selection and gates

```selection-and-gates
counted_run: the first execution of run_preflight_v3.sh under Alba's v3 release whose start is recorded in the scheduler-independent `started` file before any output is read
rerun_policy: none for a gate failure; an infrastructure crash before CNT is UNINTERPRETABLE, gets a new run ID, and the crashed run is kept
post_data_free: NONE
gate: G0 | Alba's release file bound to this document's sha256 exists | enforced_at: run_preflight_v3.sh (NOT WRITTEN; must branch before any GPU use) | before: all GPU work | qualified_on: matching release -> pass; absent or hash-mismatched release -> fail (fire drill owed)
gate: CNT | CNT-0..CNT-REP and CNT-IND pass at d=2,5,10 | enforced_at: count_compiled_pi05.py (NOT WRITTEN; exits non-zero before R9) | before: R9 parity and any env check | qualified_on: stock and P -> pass; M1/M2/M3/M5/M6/K/W1 -> fail (G02 prototype, 0/162 wrong); fire drill owed
gate: R9 | bitwise stock = candidate on DO1-DO3 at d=10 | enforced_at: parity_stock_reference.py comparator (NOT WRITTEN) | before: Protocol B release request | qualified_on: candidate vs itself -> pass; candidate with M1 sampler at d=2 vs stock d=10 -> fail (to be exercised)
gate: ENV | 2 fresh envs bitwise equal and ENV-NC differs | enforced_at: run_preflight_v3.sh (NOT WRITTEN) | before: Protocol B release request | qualified_on: fresh -> pass (E02 3/3); reused -> fail (E01, E02)
selection_risk: low: no scientific estimate is drawn; every input is synthetic or a frozen diagnostic observation outside the confirmation population
stop_on_confirmation: if every check passes, CLOSE the preflight as PREFLIGHT_PASS and hand the record to Alba; add no probes and run no outcome step
verdicts: PREFLIGHT_PASS | PREFLIGHT_FAILED | INSTRUMENT_DEFECT | UNINTERPRETABLE
review_rounds_max: 2
```

**Disclosed open lint findings.** The gates cite tools that do not exist yet. Before release, each must exist, must branch at the cited line, and must pass a fire drill: the gate is forced to fail, and the budgeted step must never start.

## A8. Abort conditions

The run aborts on any of the following:
- any gate fails;
- `RESOURCE_EXHAUSTED` anywhere;
- a hash in A3 changes;
- any step requests a confirmation identity, seed or state;
- the GPU reservation is lost;
- any step computes or emits a task-success value.

## A9. Strongest claims

- **If it passes:** "On [host], with [stack], the instrumented checks in A5 passed: the pi0.5 compiled sampler executed d steps of size 1/d at d = 2, 5, 10 on two synthetic inputs, and the candidate path's d=10 actions were bitwise identical to stock OpenPI on three frozen diagnostic observations."
- **If it fails:** "Check X failed in the counted run."
- **Not claimed either way:**
  - policy quality, or any success rate;
  - any R22/R25 effect;
  - parity on observations outside DO1–DO3;
  - renderer equivalence between hosts or between models;
  - identical physical states across MuJoCo versions;
  - SmolVLA instrument validity;
  - approval of `KeyedPolicy` (R4 is Alba's).

## A10. Decisions implemented (Alba, 6 Nov, relayed)

| ID | Ruling implemented here | Where |
|---|---|---|
| R6 | Per-model native environment documented; no renderer-controlled cross-model comparability; GPU EGL, WSL OSMesa/llvmpipe and others distinguished; unresolved facts UNKNOWN | `07` |
| R9 | Independent stock-native reference, no wrapper; frozen diagnostic observations; d=10; bitwise; h=5 only for chunk use; full action specification; cross-host tolerance procedure specified, not run | `05` §P1–P4 |
| R4 | `KeyedPolicy` is a candidate; approval evidence listed; not approved | `05` §P5 |
| N1 | Separate frozen model-native state populations; R22/R25 paired within model only; fresh env per attempt; no cross-version physical-state claim | `02` §B3 |
| CNT | G02 reviewed; requirements specified; not run; not approved | `09` |
| Design | Delta_m, 400 identities × 2 conditions per model, 1,600 total | `02`, `04` |
