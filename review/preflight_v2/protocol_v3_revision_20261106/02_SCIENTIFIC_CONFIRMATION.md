# Protocol B: cross_policy_confirmation_v2, scientific confirmation. DRAFT / NOT AUTHORISED

> **NOT AUTHORISED FOR EXECUTION.** This protocol is scoped separately from the engineering preflight (Protocol A, `01_PREFLIGHT_V3_DRAFT_REV2.md`).
>
> Execution needs all of the following:
> 1. a PREFLIGHT_PASS under Protocol A for the model concerned;
> 2. both confirmation state manifests generated and bound;
> 3. the R3 coverage study (B6.4) completed with the primary outcomes unseen;
> 4. a release from Alba bound to this document's sha256 and to each model's schedule hash.
>
> It replaces the design content of the frozen `cross_policy_confirmation_v1.json` (`9310639c...`), which stays unchanged on disk. It does not edit that file.
>
> Items marked **PROPOSED** are drafting choices for Alba's review. Everything else implements her 6 November decisions.

## B1. Research question (unchanged hypothesis)

The question, for each model m in {SmolVLA, pi0.5} separately: in the LIBERO evaluation environment native to model m, does reducing denoising steps from d=5 to d=2, at a fixed h=2, change task success?

The question keeps v1's hypothesis: is SmolVLA's d=2 over d=5 result at h=2 specific to SmolVLA? v1's secondary h question is **removed**, along with h=5, R52 and R55.

**Prediction:** NONE_RECORDED. The SmolVLA discovery direction (d=2 above d=5) is context, not a preregistered numerical prediction. If Alba wants a numerical prediction per model, she supplies it before release.

## B2. Estimand

For each model m:

    Delta_m = success_m(d=2, h=2) − success_m(d=5, h=2)

- **R22** = (d=2, h=2). **R25** = (d=5, h=2).
- h = the number of consecutive actions executed from the latest chunk before a new chunk is requested (v1 `h_definition`).
- The **target population** is the 400 frozen identities of model m's own state population (B3), each attempted under both conditions.
- Delta_m is a finite-population average treatment effect over those 400 identities, in model m's own environment (`07_HOST_GAPS_REGISTER.md`).
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

When the episode ends differs by model:
- **pi0.5:** the client breaks on `done` (`client_openpi.py:125`).
- **SmolVLA:** the loop breaks only on `terminated` or `truncated` (`runner_smolvla.py:119`), while `success` accumulates by OR (line 118). Whether LeRobot `LiberoEnv` terminates on success is **UNVERIFIED** (register G10).

| Model | Source in code | Horizon | Status |
|---|---|---|---|
| pi0.5 | `client_openpi.rollout`: `obs, _, done, _ = env.step(...)`; `success = bool(done)` (`client_openpi.py:124-127`). `done` comes from LIBERO `OffScreenRenderEnv.step` in the OpenPI LIBERO submodule | `MAX_STEPS[suite]` after 10 wait steps: 220 / 280 / 300 / 520 for spatial / object / goal / 10, matching stock `examples/libero/main.py` | **That `done` equals the BDDL goal-predicate check, with no other termination path, is UNVERIFIED in this revision.** Owed: a source read of pinned LIBERO `ControlEnv.step` / `_check_success` before release |
| SmolVLA | `runner_smolvla.rollout`: `success = success or batch_one_success(info)` over `info["is_success"]` (`runner_smolvla.py:118`; `experiments/tooling/run_paired.py:95-107`, which fails closed if the field is missing) | `base._max_episode_steps` from LeRobot `LiberoEnv` | **Horizon values UNKNOWN** in this revision. Owed: read them from the pinned stack and record them per suite |

The two success sources are different code paths with possibly different horizons. That is acceptable only because no cross-model pooling or comparison of levels is made.

## B5. Attempt classification

Every attempt record is append-only and is written ahead of the attempt. Each attempt ends in exactly one class:

| Class | Condition | Y |
|---|---|---|
| **SCI_SUCCESS** | the success signal became true | 1 |
| **SCI_FAILURE** | the horizon was reached, or the env terminated, without success. This includes any policy behaviour: dropping objects, collisions, stalls | 0 |
| **INFRA_PRE** | failure **before the first policy query**: env build error, state restore error, settled or first-observation hash mismatch (`check_reset`), server unreachable | missing; retry allowed (B7) |
| **INFRA_POST** | failure **after the first policy query**: process crash, OOM, websocket loss, host fault, a per-request count attestation mismatch (if adopted, see `09`), timeout | missing; **no retry** |
| **PROTOCOL_STOP** | non-finite action, wrong chunk shape, or d unsupported (`unsupported_refinement_setting`, v1 `stop_rule_pi05`) | the model's phase **stops**; not a missing value |

Scientific failures are outcomes and are never excluded. An infrastructure failure is never recoded as a scientific failure, and a scientific failure is never recoded as an infrastructure failure.

The classifier is to be written as a function of these record fields:
- `termination_reason`;
- the exception class: `ResetStateMismatch`, transport error, `RESOURCE_EXHAUSTED`, `unsupported_refinement_setting`;
- whether any query record exists.

It is fixed and hashed before release. **It is not written yet.**

**Timeout.** "Timeout" means a wall-clock watchdog only, set at **at least 3×** the longest horizon's expected wall time from the owed per-rollout measurement (`04` T4). A long episode that reaches its step horizon is SCI_FAILURE, never a timeout.

**Outcome-correlated missingness.** OOM or timeouts may concentrate on long `libero_10` episodes, so missingness may depend on outcomes. The worst-case bounds of B6.6 and the per-suite, per-condition missingness table are the safeguard.

**Conflict with the frozen loop.** `attempt_loop.py:7-9` stops the whole run at the first failed attempt and never retries. B5, B7 and B9 require continuing past missing attempts, plus one INFRA_PRE retry. The Protocol B runner must therefore be new code. The frozen loop is not modified.

## B6. Analysis

### B6.1 Primary estimator (within-identity, paired)

For identity i of model m: D_i = Y_i(R22) − Y_i(R25), which takes a value in {−1, 0, +1}.

The 2×2 table counts the complete pairs:

| | R25 = 1 | R25 = 0 |
|---|---|---|
| **R22 = 1** | n11 | **n10** |
| **R22 = 0** | **n01** | n00 |

The estimate is Delta_hat_m = (n10 − n01) / N_c, where N_c = n11 + n10 + n01 + n00 is the number of complete pairs. N_c = 400 if nothing is missing.

### B6.2 Discordant pairs

- The estimate depends only on the discordant counts n10 and n01. Concordant pairs enter only the denominator.
- Always report all four cells, the discordance rate (n10 + n01)/N_c, and the per-suite tables.
- Secondary: an exact two-sided McNemar test (binomial on n10 out of n10 + n01, p = 0.5). It is reported, not used for the verdict.

### B6.3 Paired uncertainty interval (PROPOSED: Alba to confirm)

Identities are not independent: the two seeds share a state, and the five slots share a task. The interval therefore respects the design.

- **Primary (PROPOSED):** a 95% cluster-robust paired interval. The clusters are the 40 tasks. The interval is Delta_hat ± t_{0.975, 39} × SE.
  - SE² = (40/39) Σ_t (S_t − Delta_hat · n_t)² / N_c².
  - S_t is the sum of D_i over the complete pairs in task t, and n_t is the number of those pairs.
  - Clustering at the outermost level nests both state and seed.
- **Sensitivity (reported, not used for the verdict):**
  - the Tango score interval for a paired difference at identity level, which ignores clustering;
  - the state-level paired bootstrap of v1 (20,000 draws, seed 20261102), kept only for continuity with the frozen v1 record.

### B6.4 R3 calibration requirement

R3 found that v1's percentile bootstrap undercovered. **Before release, and with the outcomes unseen**, a synthetic coverage study must run.

- **Design:** simulate paired binary outcomes with task, state and seed random effects across a grid of baseline success (0.2–0.95), discordance (0.02–0.30), Delta (−0.15 to +0.15) and intra-task correlation (0–0.5).
- **Criterion:** the primary interval's empirical coverage must be ≥ 0.93 in every grid cell. If it is not, Alba picks the replacement procedure **before release**, from the study's output.
- **Freeze:** the grid, the replicate count (PROPOSED: 2,000 per cell, Monte Carlo SE about 0.005 at 0.95) and the criterion are frozen with this document.
- **Infeasible cells:** combinations that cannot exist, for example baseline 0.95 with discordance 0.30 and a large Delta, are excluded by a stated feasibility rule. The marginal success probabilities must lie in [0, 1].
- This is a CPU computation on simulated data, and it is **not run here**.

### B6.5 Per-model verdict (PROPOSED wording)

| Verdict | Condition |
|---|---|
| DELTA_POSITIVE | the primary interval lies entirely above 0 |
| DELTA_NEGATIVE | the primary interval lies entirely below 0 |
| DELTA_INCONCLUSIVE | the primary interval contains 0 |
| UNINTERPRETABLE | a missingness threshold (B6.6) is breached, a pre-run gate failed, or a frozen hash changed |
| UNSUPPORTED_REFINEMENT_SETTING | PROTOCOL_STOP fired |

DELTA_INCONCLUSIVE is not evidence of no effect. It is reported as "no difference established at this N in this environment".

```verdict-table
dim: interval=above0|below0|contains0|none
dim: missingness=ok|breached
dim: stop=no|yes
rule: stop=yes, interval=*, missingness=* -> UNSUPPORTED_REFINEMENT_SETTING
rule: stop=no, missingness=breached, interval=* -> UNINTERPRETABLE
rule: stop=no, missingness=ok, interval=none -> UNINTERPRETABLE
rule: stop=no, missingness=ok, interval=above0 -> DELTA_POSITIVE
rule: stop=no, missingness=ok, interval=below0 -> DELTA_NEGATIVE
rule: stop=no, missingness=ok, interval=contains0 -> DELTA_INCONCLUSIVE
```

### B6.6 Missingness (prespecified)

- **Primary analysis:** complete pairs only. A pair is complete when both of its attempts end in SCI_SUCCESS or SCI_FAILURE.
- **Always reported:** the counts of INFRA_PRE retries, INFRA_POST attempts, and incomplete pairs, by condition and by suite.
- **Always reported, worst-case bounds:** recompute Delta_hat with every missing outcome imputed adversarially.
  - Lower bound: missing R22 → 0 and missing R25 → 1.
  - Upper bound: the reverse.
- **Thresholds (PROPOSED authored values):** the model's verdict is UNINTERPRETABLE if either of the following holds:
  - more than 20 of 400 identities (5%) have incomplete pairs;
  - missing attempts differ between R22 and R25 by more than 10.
- Missing outcomes are never treated as failures in the primary analysis, never treated as successes, and never imputed.

## B7. Retry rules

- **INFRA_PRE:** at most **one** retry, in a fresh process and a fresh env. Both records are kept. If the retry also fails, the attempt is INFRA_PRE_FINAL, which is missing. No outcome information exists before the first policy query, so the retry selects nothing.
- **INFRA_POST:** **no retry.** The attempt is missing, permanently.
- **SCI_FAILURE:** never retried.
- **Partial execution:** an attempt is never replaced after any of its output has been read. A second execution of a counted attempt is recorded, and it invalidates that attempt (it becomes missing). It never chooses between the two.

## B8. Randomisation and order controls

- **Identity order:** a keyed permutation of the 400 identities per model. The key is sha256("cross_policy_confirmation_v2", model, "order"), and it is fixed in the schedule file before release.
- **Condition order within an identity:** balanced and keyed. Within each task, the 10 identities are sorted by sha256(model, identity, "cond"). The first 5 run R22 first and the other 5 run R25 first. Each suite therefore has exactly 50 R22-first and 50 R25-first identities.
- **Adjacency:** the two attempts of a pair run consecutively, in one model process, with a fresh env each. This balances slow drift across conditions.
- **Common random numbers:** both conditions draw the same noise per (identity, tick).
  - pi0.5 noise has shape (chunk, action_dim), independent of d, so the noise arrays are equal.
  - SmolVLA uses `make_noise(policy, ident, tick)`.
- **Determinism:** pi0.5 runs with the deterministic XLA flags only if CNT-T has been recorded and Alba accepts their cost. SmolVLA runs with the `run_paired.py` settings.
- **Stage:** schedule generation is new tooling. `make_primary_schedule.py` derives the pairs from the frozen 2,000-row five-condition schedule, which this design removes. The new generator emits the 800 rows directly and is hashed into the release.

## B9. Stopping and exclusion rules

- **No interim analysis.** There is no early stop for efficacy or futility. Outcomes are not read until all 800 attempts of a model have a final class.
- **Stops:**
  - PROTOCOL_STOP (the model's phase ends);
  - any frozen-hash change, gate failure, or reservation loss (the run halts and is UNINTERPRETABLE unless resumed under a recorded amendment approved by Alba before outcomes are read);
  - INFRA_POST attempts exceeding 20 (the run halts for diagnosis before more outcomes are produced). This counts **attempts** and is an operational halt. The B6.6 threshold counts **incomplete identities** and is an analysis verdict. They are deliberately separate, and either can fire alone.
- **Exclusions:** only before the manifest is frozen (B3 validation). After the freeze, no identity is excluded. Reset-hash mismatches are INFRA_PRE, not exclusions.
- **Archived cohorts:** the invalid 160-attempt LeRobot pi0.5 campaign stays permanently excluded. The SmolVLA discovery and confirmation identities, and the stale-observation diagnostic, are disjoint by construction and checked by hash.

## B10. Selection and gates

```selection-and-gates
counted_run: per model, the first execution of the Protocol B runner under Alba's release whose start is recorded in the scheduler-independent `started` file and whose schedule sha256 matches the release
rerun_policy: INFRA_PRE once per attempt, fresh process and env, both records kept; INFRA_POST and SCI outcomes never
post_data_free: NONE; estimator, interval, missingness thresholds, verdict table and reporting fixed here; the R3 replacement, if needed, is chosen before release from synthetic data only
gate: G0-B | Alba's release bound to this document and both schedule hashes | enforced_at: launch_guard.py successor (NOT WRITTEN) | before: any confirmation attempt | qualified_on: matching release -> pass; mismatched schedule hash -> fail (fire drill owed)
gate: PRE | Protocol A PREFLIGHT_PASS recorded for this model | enforced_at: Protocol B runner (NOT WRITTEN) | before: any confirmation attempt | qualified_on: PASS record -> pass; FAILED or absent -> fail
gate: RESET | pair partner settled and first-observation hashes equal | enforced_at: experiments/tooling/cross_policy/attempt_loop.py:24 | before: first policy query of each attempt (called at client_openpi.py:87 and runner_smolvla.py:83) | qualified_on: matching reference -> pass; wrong reference -> raise ResetStateMismatch (real-simulator qualification owed in Protocol A RESET; never run)
selection_risk: material
selection_anchor: per-model schedule sha256 and runner command hash sent to Alba before launch; the earliest `started` record after that message counts
multiple_executions: a second started execution of any planned attempt makes that attempt missing; a second started run of the whole schedule makes the model's result NO VALID RESULT
stop_on_confirmation: report Delta_m per model with its verdict and CLOSE; no added conditions, no h=5 follow-up under this protocol
verdicts: DELTA_POSITIVE | DELTA_NEGATIVE | DELTA_INCONCLUSIVE | UNINTERPRETABLE | UNSUPPORTED_REFINEMENT_SETTING
review_rounds_max: 2
```

## B11. Strongest claims under each outcome

- **DELTA_POSITIVE or DELTA_NEGATIVE for model m:** "In [model m's native environment, host, renderer], on 400 frozen model-native identities, success at d=2, h=2 minus success at d=5, h=2 was X (95% CI [a, b], task-clustered)."
- **DELTA_INCONCLUSIVE:** "No difference was established for model m at this N in this environment (estimate X, CI [a, b])."
- **Never claimed:**
  - a pooled or cross-model effect;
  - renderer-controlled comparability between models;
  - identical physical states across models;
  - anything about h=5, R52 or R55;
  - quality preservation;
  - generality beyond LIBERO, these checkpoints, or these hosts;
  - SmolVLA instrument validity for the historical 1,120 attempts.
