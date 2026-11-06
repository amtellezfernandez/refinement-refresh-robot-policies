# Protocol B: cross_policy_confirmation_v2, scientific confirmation. DRAFT / NOT AUTHORISED

> **NOT AUTHORISED FOR EXECUTION.** This protocol is scoped separately from the engineering preflight (Protocol A, `01_PREFLIGHT_V3_DRAFT_REV2.md`) and from the evaluator validation (Protocol E, `13_EVALUATOR_VALIDATION.md`).
>
> Execution, for a model, needs all of the following:
> 1. a PREFLIGHT_PASS under Protocol A for that model;
> 2. EVALUATOR_QUALIFIED under Protocol E for that model's stack;
> 3. every item of that model's qualification record (`01` A3a) marked PASSED by a separately authorised check;
> 4. that model's confirmation state manifest generated and bound;
> 5. the R3 coverage check (B6.4) **run** with the primary outcomes unseen, and its result recorded;
> 6. a release from Alba bound to this document's sha256, to that model's schedule hash, to that model's qualification record hash, and to the sha256 of every runtime file of the v3 tooling (the executed classifier, runner, adapters, server and analysis are the certified ones).
>
> It replaces the design content of the frozen `cross_policy_confirmation_v1.json` (`9310639c...`), which stays unchanged on disk. It does not edit that file.
>
> It implements Alba's decisions of 6 November (`12_DECISION_REGISTER_20261106.md`, D1–D8). Items marked **PROPOSED** are drafting choices that those decisions leave open.

## B1. Research question (unchanged hypothesis)

The question, for each model m in {SmolVLA, pi0.5} separately: in the LIBERO evaluation environment native to model m, does reducing denoising steps from d=5 to d=2, at a fixed h=2, change task success?

The question keeps v1's hypothesis: is SmolVLA's d=2 over d=5 result at h=2 specific to SmolVLA? v1's secondary h question is **removed**, along with h=5, R52 and R55.

**Prediction (D6):** NONE_RECORDED. The SmolVLA discovery direction (d=2 above d=5) is context only, not a preregistered prediction. **The analysis is two-sided for each model.**

## B2. Estimand

For each model m:

    Delta_m = success_m(d=2, h=2) − success_m(d=5, h=2)

- **R22** = (d=2, h=2). **R25** = (d=5, h=2).
- h = the number of consecutive actions executed from the latest chunk before a new chunk is requested (v1 `h_definition`).
- The **primary target** is the 400 frozen identities of model m's own state population (B3), each attempted under both conditions. On this target Delta_m is a finite-population quantity: the R22 − R25 success difference of this cohort, in model m's own environment (`07_HOST_GAPS_REGISTER.md`).
- The **secondary target** (B6.3) is the expected R22 − R25 difference for fresh states on the same 40 tasks, with equal task weights.
- **Never pooled across models.** Delta_SmolVLA and Delta_pi0.5 are reported side by side. Whether their signs agree is described, not tested.

**There are no other conditions.** There is no NATIVE arm, no R52, no R55, no h=5 condition and no quality-preservation margin.

## B3. Population (N1)

| Item | Rule |
|---|---|
| Identity | `model / suite / task_id / state_slot / policy_seed` |
| Factors | 4 suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) × 10 tasks × 5 state slots × 2 policy seeds (810001, 810002) = **400 per model** |
| State sets | **Two separate frozen model-native populations.** SmolVLA states are generated in the SmolVLA stack (LeRobot `LiberoEnv`, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1). pi0.5 states are generated in the OpenPI client stack (OpenPI LIBERO submodule, robosuite 1.4.1, MuJoCo 3.2.3) |
| State generation | Per v1's R7 rule (fresh generated slots, BDDL placement sampler, keyed seed, serialized state + SHA-256). The seed key **adds the model name**, so the two populations are independent draws. Tooling change owed: `generate_states.py` keys by cohort only |
| Validation at generation | Settling, task validity, serialization round-trip, reproducibility, disjointness from all archived cohorts and from the diagnostic population. Failed draws are replaced **only before the manifest is frozen**, and every rejection is logged |
| Pairing | R22 and R25 of one identity use the same frozen state (same SHA-256), the same policy seed, the same noise keys (`[identity, tick]`) and the same environment seed. **Pairs exist only within a model** |
| Fresh env | Every attempt builds a fresh environment, restores the frozen state, then runs the stack's settle procedure: 10 no-op wait steps for pi0.5, LeRobot's reset for SmolVLA. The settled-state and first-observation hashes must match the pair partner's **before any policy query** |
| Not claimed | Physical equivalence of states across the two stacks. They differ in MuJoCo, robosuite and LIBERO packaging. A SmolVLA-stack state that restores in the client stack (N1 probe, 2 Nov) is a serialization fact, not a physics fact |

## B4. Binary success: definition and source

Y = 1 if, and only if, the environment's success signal becomes true at any control step within the episode horizon. Otherwise Y = 0.

| Model | Source in code | Horizon | Status |
|---|---|---|---|
| pi0.5 | `done` from LIBERO `OffScreenRenderEnv.step` in the OpenPI LIBERO submodule; the client breaks on `done` | `MAX_STEPS[suite]` after 10 wait steps: 220 / 280 / 300 / 520 for spatial / object / goal / 10, matching stock `examples/libero/main.py` | **That `done` equals the BDDL goal conjunction, with no other path, is UNVERIFIED.** It is established only by Protocol E (E5.5) plus the pinned source read (G11) |
| SmolVLA | `info["is_success"]`, OR-accumulated (`runner_smolvla.py:118`; `run_paired.batch_one_success`, which fails closed if the field is missing) | `_max_episode_steps` from LeRobot `LiberoEnv` | **Horizon values per suite and whether success terminates the episode are UNKNOWN.** Both are D7 qualification items, recorded by Protocol E (E5.3, E5.4) |

The two success sources are different code paths with possibly different horizons. That is acceptable only because no cross-model pooling or comparison of levels is made.

## B5. Attempt classification

Every attempt record is append-only and is written ahead of the attempt. Each attempt ends in exactly one class:

| Class | Condition | Y | Run continues? |
|---|---|---|---|
| **SCI_SUCCESS** | the success signal became true | 1 | yes |
| **SCI_FAILURE** | the horizon was reached, or the env terminated, without success. Includes any policy behaviour: dropping objects, collisions, stalls. **For SmolVLA, also a non-finite policy action under valid inputs (D7):** the action is not applied and the episode ends at that tick | 0 | yes |
| **INFRA_PRE** | failure **before the first policy query**: env build error, state restore error, settled or first-observation hash mismatch (`check_reset`), server unreachable | missing; one retry (B7) | yes |
| **INFRA_POST** | failure **after the first policy query** that is not an instrument incident: process crash, OOM, websocket loss, host fault, watchdog timeout | missing; **no retry** | yes, up to the B9 halt |
| **INSTRUMENT_INCIDENT** | **pi0.5 (D5):** a served chunk whose executed-step attestation is missing, unparseable, not bound to this request identifier and tick, or not equal to d. **SmolVLA (D7):** a broken action contract (wrong shape, dtype or dimension count) or unclear provenance (no generation record behind an executed action; provenance hash mismatch), an input that fails validity at a policy query, or a generation whose counted denoise steps differ from d (the v2 count, checked per generation before the action is applied rather than once per episode). The action is **never applied**. An incident record is written for investigation | missing | **no: the model run stops for review** |
| **PROTOCOL_STOP** | pi0.5 only, unchanged from v1 `stop_rule_pi05`: non-finite action, wrong chunk shape or a chunk shorter than h, the server not echoing the configured d, or d unsupported (`unsupported_refinement_setting`) | not a missing value | **no: the model's phase stops** |

Scientific failures are outcomes and are never excluded. An infrastructure failure is never recoded as a scientific failure, and a scientific failure is never recoded as an infrastructure failure. An instrument incident is never recoded as either.

**Valid inputs (SmolVLA, D7).** At each policy query the batch passed to the policy has the expected keys and shapes, every numeric field is finite, and the attempt's reset hashes matched. A non-finite action with valid inputs is SCI_FAILURE. A non-finite action with invalid inputs is an INSTRUMENT_INCIDENT.

**π0.5 non-finite actions and wrong shapes** stay PROTOCOL_STOP as written in v1. D7 does not cover π0.5, so the two models classify these events differently. This is open question 1 in `12` §5.

**The classifier** is written as a function of record fields: `termination_reason`, the exception class (`ResetStateMismatch`, transport error, `RESOURCE_EXHAUSTED`, `unsupported_refinement_setting`, attestation failure, contract failure), and whether a first-query marker exists. It is fixed and hashed before release. It exists as code in the implementation draft PR (`cross_policy_v3/classify.py`) and has not been run on any real attempt.

**Timeout.** "Timeout" means a wall-clock watchdog only, set at **at least 3×** the longest horizon's expected wall time from the owed per-rollout measurement (`04` T4). A long episode that reaches its step horizon is SCI_FAILURE, never a timeout.

**Outcome-correlated missingness.** OOM or timeouts may concentrate on long `libero_10` episodes, so missingness may depend on outcomes. The worst-case bounds and the per-arm, per-suite missingness tables of B6.5 are the safeguard.

**Conflict with the frozen loop.** `attempt_loop.py:7-9` stops the whole run at the first failed attempt and never retries. B5, B7 and B9 require continuing past missing attempts, plus one INFRA_PRE retry. The Protocol B runner is therefore new code. The frozen loop is not modified.

## B6. Analysis (D1, D2)

Every analysis is run per model, two-sided, and only after all 800 attempts of that model have a final class (B9).

### B6.1 Paired outcomes

For identity i of model m: D_i = Y_i(R22) − Y_i(R25), which takes a value in {−1, 0, +1}. A pair is **complete** when both of its attempts end in SCI_SUCCESS or SCI_FAILURE.

The four paired outcome counts are over complete pairs:

| | R25 = 1 | R25 = 0 |
|---|---|---|
| **R22 = 1** | n11 | n10 |
| **R22 = 0** | n01 | n00 |

N_c = n11 + n10 + n01 + n00. Always reported: all four counts, overall and per suite.

### B6.2 Primary result: the exact cohort contrast

- **With all 400 pairs complete**, the primary result is the **exact cohort contrast**:

      Delta_m(cohort) = (n10 − n01) / 400

  It is the R22 − R25 success difference of these 400 identities, computed without sampling error, together with the four counts. No interval is attached to it, because it is not an estimate of this cohort's difference: it is that difference.
- **With any pair incomplete**, the cohort contrast is not known exactly. The primary result is then the pair of worst-case bounds (B6.5). The **complete-pair estimate**, (n10 − n01) / N_c, is reported separately, labelled `complete_pair_estimate`, and is never presented as the cohort contrast.

### B6.3 Secondary inference: fresh states on the same 40 tasks

This answers a different question from B6.2: what difference would fresh states, drawn by the same generator on the same 40 tasks, show on average?

- **Reported only if** all 400 pairs are complete **and** the R3 coverage check (B6.4) passed. Otherwise it is recorded as not reported, with the reason, and nothing replaces it.
- **Estimator.** For task t and state slot s, average the two seeds: Ď_ts = (D_ts,810001 + D_ts,810002) / 2. Task mean m_t = (1/5) Σ_s Ď_ts. Estimate = (1/40) Σ_t m_t, with equal task weights. With complete data it equals the exact cohort contrast.
- **Within-task variance.** s_t² = (1/4) Σ_s (Ď_ts − m_t)². SE² = (1/40²) Σ_t s_t² / 5.
- **Interval (95%).** Estimate ± c × SE, with **c = t₀.₉₇₅ at 160 degrees of freedom** (= Σ_t (5 − 1)); c ≈ 1.9749. **PROPOSED**: D1 does not name a reference distribution, and the R3 check evaluates this interval exactly as specified here.
- Tasks are fixed, not sampled. The interval says nothing about other tasks, suites or benchmarks.

### B6.4 R3 coverage check (preregistered; required for B6.3)

R3 found that v1's percentile bootstrap undercovered. The B6.3 interval is reported only if a synthetic coverage check, run **before release with the outcomes unseen**, passes.

- **Generative model (PROPOSED).** Each pair's joint outcome is one of {11, 10, 01, 00}, with base probabilities from baseline success b = P(R25 = 1), Delta Δ and discordance q = P(R22 ≠ R25): π10 = (q + Δ)/2, π01 = (q − Δ)/2, π11 = b − π01, π00 = 1 − π11 − π10 − π01. Task-level heterogeneity: π_t ~ Dirichlet(κ π) with κ = (1 − ρ)/ρ (none when ρ = 0). State-level heterogeneity inside a task: π_ts ~ Dirichlet(κ π_t), with the same ρ. The two seeds of a state are independent draws from π_ts. The 40 × 5 × 2 layout matches B3.
- **Target in each replicate:** the fresh-state difference on that replicate's 40 tasks, (1/40) Σ_t (π_t,10 − π_t,01). The check measures coverage of the B6.3 question, conditional on the tasks.
- **Grid (PROPOSED):** b ∈ {0.20, 0.50, 0.80, 0.95}; q ∈ {0.02, 0.10, 0.30}; Δ ∈ {−0.15, −0.05, 0, +0.05, +0.15}; ρ ∈ {0, 0.25, 0.5}.
- **Feasibility rule:** a cell is evaluated only if q ≥ |Δ| and all four π lie in [0, 1]. Infeasible cells are listed, not evaluated.
- **Replicates:** 2,000 per feasible cell (Monte Carlo SE about 0.005 at 0.95). Seed 20261107. **PROPOSED**.
- **Criterion:** empirical coverage ≥ 0.93 in **every** feasible cell. Otherwise the check fails, and B6.3 is not reported. Any replacement interval would need a new decision by Alba before release, made from synthetic output only.
- **Binding:** the check's result record names the sha256 of the analysis code and of the grid. The analysis refuses an R3 record whose hashes differ.
- **Disclosed expectation, not a prediction about outcomes.** At low discordance (q = 0.02) many tasks will have no discordant pair in a replicate, so s_t² = 0 for those tasks and the interval can be very short. Cells of this kind may fail the criterion. The check is designed to detect exactly this, and its result is reported whichever way it goes.
- This is a CPU computation on simulated data. It exists as code in the implementation draft PR and **has not been run**.

### B6.5 Missingness, bounds and the per-model verdict (D2)

**Always reported**, per model:
- missing attempts by arm (R22, R25) and by suite, after permitted retries;
- INFRA_PRE retries, INFRA_POST attempts, instrument incidents and incomplete pairs, by arm and by suite.

**Worst-case bounds**, always over all 400 identities (denominator 400, never N_c):
- Lower bound L: each missing R22 outcome set to 0 and each missing R25 outcome set to 1, then Σ D_i / 400.
- Upper bound U: each missing R22 outcome set to 1 and each missing R25 outcome set to 0, then Σ D_i / 400.
- With no missing outcome, L = U = the exact cohort contrast.

**Thresholds (D2).** The model's result is UNINTERPRETABLE if either holds:
- more than 20 of the 400 pairs are incomplete;
- |missing R22 attempts − missing R25 attempts| > 10, counted after permitted retries.

**Direction.** Below both thresholds, a direction is claimed **only if the bounds exclude zero**: L > 0 gives DELTA_POSITIVE (R22 above R25), U < 0 gives DELTA_NEGATIVE. Otherwise the result is DELTA_INCONCLUSIVE.

| Verdict | Condition |
|---|---|
| DELTA_POSITIVE | no stop, no incident, thresholds not breached, L > 0 |
| DELTA_NEGATIVE | no stop, no incident, thresholds not breached, U < 0 |
| DELTA_INCONCLUSIVE | no stop, no incident, thresholds not breached, L ≤ 0 ≤ U |
| UNINTERPRETABLE | a missingness threshold is breached, an instrument incident stopped the run, the run halted (more than 20 INFRA_POST attempts, or a gate, hash or reservation stop), or a pre-run gate failed |
| UNSUPPORTED_REFINEMENT_SETTING | PROTOCOL_STOP fired (pi0.5) |

DELTA_INCONCLUSIVE is not evidence of no effect. With complete data and n10 = n01 it means the cohort's R22 and R25 success counts were equal.

```verdict-table
dim: stop=no|yes
dim: incident=no|yes
dim: missingness=ok|breached
dim: bounds=above0|below0|contains0
rule: stop=yes, incident=*, missingness=*, bounds=* -> UNSUPPORTED_REFINEMENT_SETTING
rule: stop=no, incident=yes, missingness=*, bounds=* -> UNINTERPRETABLE
rule: stop=no, incident=no, missingness=breached, bounds=* -> UNINTERPRETABLE
rule: stop=no, incident=no, missingness=ok, bounds=above0 -> DELTA_POSITIVE
rule: stop=no, incident=no, missingness=ok, bounds=below0 -> DELTA_NEGATIVE
rule: stop=no, incident=no, missingness=ok, bounds=contains0 -> DELTA_INCONCLUSIVE
```

Missing outcomes are never treated as failures or successes in any reported estimate. They are imputed only inside the worst-case bounds, in the adversarial directions defined above.

## B7. Retry rules

- **INFRA_PRE:** at most **one** retry, in a fresh process and a fresh env. Both records are kept. If the retry also fails, the attempt is missing. No outcome information exists before the first policy query, so the retry selects nothing.
- **INFRA_POST:** **no retry.** The attempt is missing, permanently.
- **INSTRUMENT_INCIDENT:** **no retry.** The model run stops.
- **SCI_FAILURE:** never retried.
- **Partial execution:** an attempt is never replaced after any of its output has been read. A second execution of a counted attempt is recorded, and it invalidates that attempt (it becomes missing). It never chooses between the two.

## B8. Randomisation, order and determinism

- **Identity order:** a keyed permutation of the 400 identities per model. The key is sha256("cross_policy_confirmation_v2", model, "order"), and it is fixed in the schedule file before release.
- **Condition order within an identity:** balanced and keyed. Within each task, the 10 identities are sorted by sha256(model, identity, "cond"). The first 5 run R22 first and the other 5 run R25 first. Each suite therefore has exactly 50 R22-first and 50 R25-first identities.
- **Adjacency:** the two attempts of a pair run consecutively, in one model process, with a fresh env each. This balances slow drift across conditions.
- **Common random numbers:** both conditions draw the same noise per (identity, tick).
  - pi0.5 noise has shape (chunk, action_dim), independent of d, so the noise arrays are equal.
  - SmolVLA uses `make_noise(policy, ident, tick)`.
- **Determinism (D8):** pi0.5 runs with `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`, exactly as certified in Protocol A. SmolVLA runs with the `run_paired.py` settings, exactly as certified in Protocol A.
- **Host (D4):** both models run on the RTX 5090 WSL host under OSMesa, each in its own pinned stack, in the configuration bound in its qualification record.
- **Schedule generation** is new tooling (implementation draft PR, `cross_policy_v3/make_schedule_v2.py`). It reads the model's state manifest and emits the 800 rows directly. `make_primary_schedule.py` derives pairs from the frozen 2,000-row five-condition schedule, which this design removes, so it is not used.

## B9. Stopping and exclusion rules

- **No interim analysis.** There is no early stop for efficacy or futility. Outcomes are not read until all 800 attempts of a model have a final class.
- **Stops:**
  - PROTOCOL_STOP (the model's phase ends);
  - an INSTRUMENT_INCIDENT (the model run stops for review, and the incident is investigated before anything is changed);
  - any frozen-hash change, gate failure, or reservation loss.
  - After any of the last three stops, the result is UNINTERPRETABLE unless the run resumes under a recorded amendment that Alba approves before any outcome is read.
  - INFRA_POST attempts exceeding 20: the run halts for diagnosis before more outcomes are produced. This counts **attempts** and is an operational halt. The B6.5 threshold counts **incomplete pairs** and is an analysis verdict. They are deliberately separate, and either can fire alone.
- **Exclusions:** only before the manifest is frozen (B3 validation). After the freeze, no identity is excluded. Reset-hash mismatches are INFRA_PRE, not exclusions.
- **Archived cohorts:** the invalid 160-attempt LeRobot pi0.5 campaign stays permanently excluded. The SmolVLA discovery and confirmation identities, and the stale-observation diagnostic, are disjoint by construction and checked by hash.

## B10. Selection and gates

```selection-and-gates
counted_run: per model, the first execution of the Protocol B runner under Alba's release whose start is recorded in the scheduler-independent `started` file and whose schedule sha256 matches the release
rerun_policy: INFRA_PRE once per attempt, fresh process and env, both records kept; INFRA_POST, INSTRUMENT_INCIDENT and SCI outcomes never
post_data_free: NONE; estimator, bounds, thresholds, secondary interval, R3 criterion and verdict table fixed here; the R3 check runs before release on synthetic data only
gate: G0-B | Alba's release bound to this document and the model's schedule hash | enforced_at: experiments/tooling/cross_policy_v3/runner_v2.py check_release (implementation PR; never run) | before: any confirmation attempt | qualified_on: matching release -> pass; mismatched schedule hash -> fail (fire drill owed)
gate: PRE | Protocol A PREFLIGHT_PASS, Protocol E EVALUATOR_QUALIFIED and every qualification item PASSED for this model | enforced_at: experiments/tooling/cross_policy_v3/qualification.py require_qualified (implementation PR; never run) | before: any confirmation attempt | qualified_on: all PASSED -> pass; any UNVERIFIED -> fail (unit-tested on synthetic records only)
gate: ATT | pi0.5 per-request executed-step attestation equals d and is bound to this request and tick | enforced_at: experiments/tooling/cross_policy_v3/attestation.py verify (implementation PR; never run) | before: application of each served chunk's first action | qualified_on: correct attestation -> pass; missing, wrong count, wrong request or tick -> fail (unit-tested on synthetic attestations only)
gate: RESET | pair partner settled and first-observation hashes equal | enforced_at: experiments/tooling/cross_policy/attempt_loop.py:24 | before: first policy query of each attempt | qualified_on: matching reference -> pass; wrong reference -> raise ResetStateMismatch (real-simulator qualification owed in Protocol A RESET; never run)
selection_risk: material
selection_anchor: per-model schedule sha256 and runner command hash sent to Alba before launch; the earliest `started` record after that message counts
multiple_executions: a second started execution of any planned attempt makes that attempt missing; a second started run of the whole schedule makes the model's result NO VALID RESULT
stop_on_confirmation: report Delta_m per model with its verdict and CLOSE; no added conditions, no h=5 follow-up under this protocol
verdicts: DELTA_POSITIVE | DELTA_NEGATIVE | DELTA_INCONCLUSIVE | UNINTERPRETABLE | UNSUPPORTED_REFINEMENT_SETTING
review_rounds_max: 2
```

## B11. Strongest claims under each outcome

- **DELTA_POSITIVE or DELTA_NEGATIVE for model m, complete data:** "On 400 frozen model-native identities, in [model m's native stack] on the RTX 5090 WSL host under OSMesa, success at d=2, h=2 minus success at d=5, h=2 was exactly X (n11, n10, n01, n00 = …)." If B6.3 is reported: "For fresh states on the same 40 tasks, the 95% interval was [a, b]."
- **DELTA_POSITIVE or DELTA_NEGATIVE with missing pairs:** "Over all 400 identities, the difference lies in [L, U] under worst-case imputation of k missing outcomes; the complete-pair estimate on N_c pairs was X."
- **DELTA_INCONCLUSIVE:** "No direction was established for model m on this cohort in this environment (bounds [L, U])."
- **Never claimed:**
  - a pooled or cross-model effect;
  - simulator or renderer equivalence between models;
  - identical physical states across models;
  - anything about h=5, R52 or R55;
  - quality preservation;
  - generality beyond LIBERO, these 40 tasks, these checkpoints, or this host;
  - SmolVLA instrument validity for the historical 1,120 attempts.
