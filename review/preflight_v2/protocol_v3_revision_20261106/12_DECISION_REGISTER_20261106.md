# Decision register: Alba's eight protocol decisions of 6 November 2026

**Source.** Alba's eight decisions of 6 November 2026, answering the eight questions in `00_COVER_NOTE_FOR_ALBA.md` (first draft of this revision), as relayed by Warren. Her own text is not held in this repository. Where her own wording later differs from this register, her wording governs.

**Relation to earlier registers.** `review/preflight_v2/DECISION_REGISTER.md` (2 November) is frozen evidence and is **not edited**. Where this register differs from it, or from any PROPOSED item in revision 2 as first published, this register governs. Superseded requirements are removed from the current text of `01`–`09` and listed in §3 below, so that nothing superseded remains in force by accident.

**Scope.** Documentation and specification only. Nothing here authorises execution: no GPU job, diagnostic, state generation, parity run, counted preflight, confirmation run, or use of a reserved identity.

## 1. The decisions and where each is implemented

| # | Decision (6 Nov) | Implemented in |
|---|---|---|
| D1 | **Primary analysis.** For each model separately, the primary estimand is the R22 − R25 success difference over the 400 frozen paired identities. With complete pairs, report the **exact cohort contrast** and the **four paired outcome counts**. Secondary inference: a 95% interval for **fresh states on the same 40 tasks**, with equal task weights, the two seeds averaged within each state, and within-task variance estimated. It is reported only with complete data, and only if the preregistered R3 coverage check passes. **The task-clustered t₃₉ interval is not used.** | `02` B6.1–B6.4 |
| D2 | **Missingness.** UNINTERPRETABLE if more than 20 pairs are incomplete, or if the absolute difference in missing attempts between the two arms exceeds 10 after permitted retries. Missingness is always reported by arm and by suite. Worst-case bounds use all 400 identities in the denominator. Below the thresholds, a direction is claimed only if the bounds exclude zero. The complete-pair estimate is labelled separately. | `02` B6.5–B6.6 |
| D3 | **Evaluator validation.** The 4 × 10 × 10 episode-level parity exercise is replaced by a small, separate, **model-free** check for **both** stacks. It uses version-matched saved actions or scripted fixtures outside the confirmation population, and covers goal reached, goal unmet, and horizon/termination. BDDL goal predicates are inspected independently against simulator state and compared with the recorded outcome. π0.5 `done` semantics are established explicitly. | `13_EVALUATOR_VALIDATION.md` (Protocol E) |
| D4 | **Hosts and renderers.** Planned host: **RTX 5090, WSL, OSMesa**, for both models. Each model keeps its own pinned native simulator and evaluator stack. Before qualification, bind the actual renderer string, the driver, the package locks and the asset hashes. Protocol A and Protocol B use the same certified configuration for each model. The 4060 Ti NVIDIA EGL host is **excluded**. No simulator equivalence between models is claimed. | `07` H1–H3; `01` A3, A5 |
| D5 | **Per-request count.** For π0.5, executed denoising steps are attested on **every served chunk**, bound to the request identifier and the execution tick. Required: the startup G02 gate, independent count verification, an action-output preservation check, and an overhead measurement. A missing or incorrect count **prevents action application, stops the model run, and is recorded as an instrument investigation**. | `09` C2, C4; `02` B5; `01` A5 |
| D6 | **Numerical prediction.** NONE_RECORDED. The SmolVLA discovery direction is context only. The analysis is two-sided for each model. | `02` B1, B6 |
| D7 | **SmolVLA qualification.** Required before SmolVLA enters Protocol B: an independent evaluator check; horizons by suite; success termination by suite; normalization verification; action mapping through the evaluator boundary. A non-finite policy action under valid inputs is a **scientific failure**. A broken action contract or unclear provenance is an **instrument incident**, and the run stops for review. | `01` A5 (S-QUAL); `13`; `02` B5 |
| D8 | **Determinism and parity.** The deterministic XLA flags are used for stock native π0.5, for the candidate implementation, and in the same configuration in Protocol B. Required: same-host bitwise action parity, matched noise, a preprocessing comparison from identical raw observations through both adapters, and controller clipping and coordinate-frame checks against pinned source. **If this is infeasible, stop and return a revised proposal before collecting new parity data.** | `05` P1–P3, P6; `01` A4–A5 |

## 2. Items these decisions settle

| Item | Status before | Status now |
|---|---|---|
| Cover-note question 1 (primary interval) | PROPOSED t₃₉ task-clustered interval | Settled by D1; the t₃₉ interval is removed |
| Question 2 (missingness thresholds) | PROPOSED | Settled by D2 |
| Question 3 (episode-level evaluator parity) | open | Settled by D3: replaced by Protocol E |
| Question 4 (hosts) | G1, G2 UNKNOWN | Settled by D4: one planned host; G2 closed as excluded |
| Question 5 (per-request attestation) | optional | Settled by D5: required |
| Question 6 (numerical prediction) | NONE_RECORDED | Settled by D6: NONE_RECORDED, two-sided |
| Question 7 (SmolVLA preflight scope) | open | Settled by D7: five qualification items required |
| Question 8 (deterministic XLA flags) | open; cost measurement first | Settled by D8: flags adopted for stock, candidate and Protocol B |

## 3. Superseded requirements removed from the current text

| Removed | Was in | Replaced by |
|---|---|---|
| Primary task-clustered paired interval, Delta_hat ± t₀.₉₇₅,₃₉ × SE | `02` B6.3 | D1: exact cohort contrast; secondary fresh-state interval |
| Interval-based verdicts (interval above / below / containing 0) | `02` B6.5 | D2: worst-case bounds with denominator 400 |
| Exact McNemar test as a secondary result | `02` B6.2 | Not part of D1; removed |
| Tango score interval and v1 state-level bootstrap as sensitivity analyses | `02` B6.3 | Not part of D1; removed |
| "Missing attempts differ by condition by more than 10" (counted before retries were defined as permitted) | `02` B6.6 | D2: after permitted retries, absolute difference |
| Attestation mismatch classed INFRA_POST (missing, run continues) | `02` B5; `09` C2 | D5: instrument incident; action not applied; model run stops |
| Per-request attestation "optional" | `09` C2 | D5: required |
| Deterministic XLA flags conditional on CNT-T and on Alba's acceptance of their cost | `01` A4–A5; `02` B8; `05` P1 | D8: adopted. CNT-T is still measured and recorded, as information only |
| Unresolved choice of host per model (G1); Alba's GPU-EGL reference host (G2) | `07` H3 | D4: RTX 5090 WSL OSMesa for both; 4060 Ti EGL excluded |
| Cross-host tolerance procedure as part of the parity plan | `05` P4 | D4 + D8: single host, bitwise; P4 is retained only as the rule that applies if a host change is ever proposed, which would need a new decision |
| "SmolVLA normalization, units and evaluator parity are not tested" | `01` A5 | D7: required qualification items |
| Episode-level evaluator check "flagged for Alba" | `03` | D3: Protocol E |

## 4. Not changed by these decisions

- The estimand's definition, the two conditions (R22 = d=2, h=2; R25 = d=5, h=2), the 400 × 2 = 800 attempts per model, and 1,600 in total.
- Separate model-native state populations (N1), pairing only within a model.
- `KeyedPolicy` remains a **candidate** (R4). D8 sets the parity criteria; approval remains Alba's, after evidence exists.
- The π0.5 PROTOCOL_STOP rule of v1 (`stop_rule_pi05`) for an unsupported refinement setting.
- All frozen files and checksums.

## 5. Decisions on the four implementation questions

1. **π0.5 outcomes.** Apply the same rule as SmolVLA. An attested chunk of the expected floating-point shape that contains a non-finite action is a scientific failure only if valid inputs and a non-finite raw model output are verified for that request. A malformed chunk, failed attestation, or unclear attribution stops the run as an instrument incident before applying the action. Keep PROTOCOL_STOP for unsupported refinement settings.
2. **R3.** Freeze the existing baseline, discordance, and Delta grid. Vary task heterogeneity and within-state seed dependence independently over {0, 0.25, 0.5} at each level. Use 2,000 replicates per feasible cell, seed 20261107, and minimum empirical coverage 0.93. Freeze the grid and code before a separately authorised synthetic check.
3. **Secondary interval.** Use Student t with 160 degrees of freedom as the prespecified candidate. Report it only for complete data if R3 passes. The exact cohort contrast remains primary.
4. **Protocol E fixtures.** Prefer version-matched saved actions. Alternatively use a scripted trajectory frozen as an action file, with its script source and hashes recorded, on each pinned native stack. Use diagnostic states disjoint from confirmation. Validate success and termination through the evaluator with an independent BDDL goal check. A goal teleport alone does not qualify the rollout path.

The proposed release fields `v3_code_sha256` and `qualification_sha256_by_model` are accepted for review. Authorization still binds the exact protocol and schedule and requires independent approval before execution.
