# Protocol E: model-free evaluator validation, both stacks. DRAFT / NOT AUTHORISED

> **NOT AUTHORISED FOR EXECUTION.** This protocol implements decision D3 (`12_DECISION_REGISTER_20261106.md`). It replaces the v2 4 × 10 × 10 native-parity episodes. It is separate from Protocol A (`01`) and Protocol B (`02`). No policy runs in it and no confirmation identity is touched.
>
> Status: specification only. The harness exists as code in the separate implementation draft PR (`experiments/tooling/cross_policy_v3/evaluator_validation.py`); **it has never been run**. No fixture has been created.

## E1. Question

For each model's own pinned stack, does the recorded episode outcome agree with the task's BDDL goal, evaluated independently against simulator state, in the three cases goal reached, goal unmet and horizon reached?

It answers **no** question about any policy, any success rate or any effect of d or h.

## E2. Stacks

| Stack | Outcome as recorded by the stack | Episode end |
|---|---|---|
| π0.5 client (OpenPI LIBERO submodule, robosuite 1.4.1, MuJoCo 3.2.3) | `done` returned by `OffScreenRenderEnv.step`; the client sets `success = bool(done)` (`client_openpi.py:124-127`) | the client loop breaks on `done`, or stops at `MAX_STEPS[suite]` after 10 wait steps |
| SmolVLA (LeRobot `LiberoEnv`, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1) | `info["is_success"]`, read by `run_paired.batch_one_success`, OR-accumulated (`runner_smolvla.py:118`) | the loop breaks on `terminated` or `truncated`, or stops at `_max_episode_steps` |

Both stacks run on the planned host (D4): RTX 5090, WSL, OSMesa, with the renderer string, driver, package locks and asset hashes bound in the qualification record first. **No equivalence between the two stacks is claimed or tested.**

## E3. Fixtures (outside the confirmation population)

A fixture is: `{stack, suite, task_id, initial_state, action_source, expected_case}`.

- **Initial state:** an official LIBERO initial state of the task, by index, loaded from the stack's own benchmark files. The confirmation population consists only of freshly generated slots, so official states are outside it by construction. The ISO check (`01` A5) also confirms that no fixture state hash, seed or identity collides with either confirmation manifest.
- **Action source:** either
  - **scripted:** a fixed, written-down action sequence, such as the stock no-op `[0, 0, 0, 0, 0, 0, −1]` repeated; or
  - **saved:** a version-matched saved action array, bound by its sha256, recorded in the same stack version (D3).
- **Cases per stack (PROPOSED, small):** for each of the four suites, one task (task 0 unless the fixture needs another):

| Case | Action source | Expected independent goal truth | Expected evaluator outcome |
|---|---|---|---|
| GOAL_UNMET | scripted no-op, 20 steps after settling | false at every step | no success at any step; no termination before step 20 |
| HORIZON | scripted no-op until the stack's horizon | false at every step | no success; the episode ends exactly at the stack's horizon for that suite, and the horizon value is recorded |
| GOAL_REACHED | saved, version-matched action array | becomes true at some step k | success at step k, and termination behaviour at k recorded |

That is 12 fixtures per stack, 24 in total.

**Goal-reached fixtures do not exist yet for either stack.** No policy may produce them under this protocol. Admissible sources are a version-matched replay of a recorded demonstration, or a scripted sequence, provided its source and sha256 are bound before the counted run. If no admissible goal-reached fixture can be produced for a stack, that stack's evaluator qualification cannot pass, and this is reported rather than worked around. This is open question 4 in `12` §5.

## E4. Independent inspection

At every step of every fixture the harness records two things, separately:

1. **The recorded outcome:** what the stack itself reports (E2).
2. **The independent goal truth.** The harness reads the task's BDDL file from disk, binds its sha256, and parses the `:goal` section with its own parser, without `BDDLUtils`. It evaluates each goal atom against the live simulator state through LIBERO's atomic predicate functions (`eval_predicate_fn`) on the env's `object_states_dict`, and forms the conjunction itself.

As a cross-check, the harness asserts that its parsed goal atoms equal the env's own `parsed_problem["goal_state"]`. A difference stops the fixture as an instrument incident.

**What is and is not independent.** Independent of the evaluator: the goal parse, the conjunction, the per-step reading, and the wiring from goal truth to `done`, `is_success`, `terminated`, `truncated` and the horizon. **Not independent:** the geometry of each atomic predicate (for example `On`'s contact and height thresholds). It is shared with the evaluator, and it is in the trusted base of this check. Re-implementing predicate geometry would be a new instrument; that is not proposed.

## E5. Pass conditions, per stack

1. At every step of every fixture: recorded success = independent goal truth. For π0.5, `done` = goal truth.
2. GOAL_UNMET: both false throughout, and no early termination.
3. HORIZON: both false throughout. The episode ends exactly at the stack's horizon. The horizon is recorded per suite (for SmolVLA, `_max_episode_steps`, which closes gap G10).
4. GOAL_REACHED: the goal truth first becomes true at step k, and the recorded success first becomes true at the same step k. Termination at k is recorded per suite (for SmolVLA this records whether success terminates the episode, D7).
5. **π0.5 `done` semantics, established only if** conditions 1–4 hold for every π0.5 fixture **and** the source read of the pinned OpenPI LIBERO `step` is recorded in the qualification record (gap G11). The statement established is then: "In the pinned π0.5 stack, `done` is true exactly when the BDDL goal conjunction holds; reaching the horizon does not set it." Anything less leaves G11 open.

A source read of the locally installed hf-libero 0.1.4 (the SmolVLA stack's version, read on the MacBook, not on the planned host) shows `BDDLBaseDomain.step` overwriting robosuite's `done` with `_check_success()`, and `_check_success` as the conjunction of goal predicates. That is a **lead** for the SmolVLA stack only. It is not a qualification, and it says nothing about the OpenPI submodule's copy.

## E6. Verdicts (per stack)

```verdict-table
dim: agreement=all|some_disagree
dim: fixtures=complete|goal_reached_missing|fixture_invalid
dim: incident=no|yes
rule: incident=yes, agreement=*, fixtures=* -> UNINTERPRETABLE
rule: incident=no, agreement=some_disagree, fixtures=* -> EVALUATOR_DISAGREES
rule: incident=no, agreement=all, fixtures=goal_reached_missing -> EVALUATOR_NOT_QUALIFIED
rule: incident=no, agreement=all, fixtures=fixture_invalid -> EVALUATOR_NOT_QUALIFIED
rule: incident=no, agreement=all, fixtures=complete -> EVALUATOR_QUALIFIED
```

- `fixture_invalid`: a goal-reached fixture whose independent goal truth never becomes true. It is evidence about the fixture, not about the evaluator.
- `EVALUATOR_DISAGREES` is an instrument finding. It blocks that stack from Protocol B. Its cause is investigated before anything is changed.

## E7. Selection and gates

```selection-and-gates
counted_run: per stack, the first execution of evaluator_validation.py under Alba's Protocol E authorisation whose fixture manifest sha256 matches the authorisation
rerun_policy: none for a disagreement; an infrastructure crash before the first fixture step is UNINTERPRETABLE, gets a new run ID, and the crashed run is kept
post_data_free: NONE
gate: E-AUTH | authorisation file bound to this document's sha256 and the fixture manifest sha256 | enforced_at: experiments/tooling/cross_policy_v3/evaluator_validation.py (implementation PR; never run) | before: any env construction | qualified_on: matching authorisation -> pass; absent or mismatched -> fail (fire drill owed)
selection_risk: low: model-free fixtures with recorded expected cases; no confirmation identity
stop_on_confirmation: if every fixture agrees, record EVALUATOR_QUALIFIED for that stack and CLOSE; add no fixtures
verdicts: EVALUATOR_QUALIFIED | EVALUATOR_NOT_QUALIFIED | EVALUATOR_DISAGREES | UNINTERPRETABLE
review_rounds_max: 2
```

## E8. Not claimed

- Correctness of LIBERO's atomic predicate geometry.
- Anything about policy-driven episodes.
- Equivalence of the two stacks' physics, renderers or evaluators.
- Validity of the historical SmolVLA campaign.
