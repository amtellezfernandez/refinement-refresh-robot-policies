# Protocol and schedule status, 2 November 2026

A schedule is a plan and is not execution evidence. **State hashes are not available in any schedule, because generated initial states do not yet exist.** Every row has `state_sha256: null`, and each `schedule_sha256` will change when states are bound. All three protocols have `launch_permitted: false`, last changed at commit `9f36fdf`. They are draft specifications awaiting review, and are not finally frozen.

## Common to all three

| Item | Value |
|---|---|
| SmolVLA | `HuggingFaceVLA/smolvla_libero`, revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, `model.safetensors` `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`, LeRobot `b6ec0060779550c0a157ae34feb89e0cf86012a8` |
| pi0.5 | `gs://openpi-assets/checkpoints/pi05_libero/`, normalization `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`, OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Population | 4 suites, 10 tasks, 5 state slots, 2 seeds: 400 identities per model |
| Preflight status | **PREFLIGHT INCOMPLETE** for both models; the bounded check of 1 November is **PREFLIGHT FAILED**. See `06_PREFLIGHT_STATUS_AND_CORRECTIONS.md` |
| Completed checks | Checkpoint identities; normalization hash; load; finite chunks at d=2, 5, 10; smoke evaluators at NATIVE, R22, R25 |
| Missing checks | Native parity; 100 + 100 profiling; videos; R52 and R55; action units; reset reuse across conditions; generated states |

## `cross_policy_confirmation_v1`

| Item | Detail |
|---|---|
| Purpose | Is the d=2 over d=5 result at h=2 specific to SmolVLA, and how does h=5, d=2 compare with h=2, d=2? |
| Specification | `experiments/protocols/cross_policy_confirmation_v1.json`, sha256 `9310639c3dbf19875bf8cb472b5cdc68551aa2038169441565a9e971b74d7221` |
| SmolVLA schedule | `experiments/runs/cross_policy_confirmation_v1/smolvla/schedule.json`, 2,000 rows, sha256 `df716ae4f6f8dce472f9dddc51186c61b87f3c920cb749a98fd11f9685c6b179` |
| pi0.5 schedule | `experiments/runs/cross_policy_confirmation_v1/pi05/schedule.json`, 2,000 rows, sha256 `a3e74e563315cf7f62f2a4a1b6a09b8c3874f77d1885de031bd382529c3c97dc` |
| Conditions | NATIVE, R22, R25, R52, R55; seeds 810001, 810002 |
| Preflight requirements | The `mandatory_preflight` list in the protocol |
| Run status | PLANNED; NOT STARTED; 0 of 4,000 |
| Available evidence | Schedules and preflight evidence; no attempt journal |
| Blocking conditions | R6 rendering; R7 states; R9 parity threshold; R3 bootstrap; R4 wrapper; R12 pi0.5 native; mandatory preflight; Alba's release |
| Next permitted action | Alba rules on the blocking decisions. No rollout |

## `observation_age_confirmation_v1`

| Item | Detail |
|---|---|
| Purpose | Effect of supplied observation age alone, at h=1, d=5: FRESH vs STALE |
| Specification | `experiments/protocols/observation_age_confirmation_v1.json`, sha256 `1756e7b7dbd427b307df70fbf5d38f724b34263f5b05d1c2bbac487624602e84` |
| SmolVLA schedule | `experiments/runs/observation_age_confirmation_v1/smolvla/schedule.json`, 800 rows, sha256 `6f5d676ceec315fd2e5f8e23c50be1e4f2c200d68d749e493341c79ad9f85c45` |
| pi0.5 schedule | `experiments/runs/observation_age_confirmation_v1/pi05/schedule.json`, 800 rows, sha256 `40357f9977d9a8c1ec7b4ac90205b08b596b100146eed57e98e8f3601bda86e4` |
| Conditions | FRESH, STALE; seeds 820001, 820002, proposed and unconfirmed |
| Preflight requirements | No separate list; shares model checks; no stale-path check has run |
| Run status | PLANNED; NOT STARTED; 0 of 1,600 |
| Available evidence | Schedules; `first_action_divergence.py`, tested on synthetic data only |
| Blocking conditions | R2 seeds; R16 first-action divergence meaning; R7; runs after confirmation; Alba's release |
| Next permitted action | Alba rules on R2 and R16. No rollout |

## `asynchronous_confirmation_v1`

| Item | Detail |
|---|---|
| Purpose | Whether lower computation stays useful under asynchronous execution: NATIVE, R22, R52 |
| Specification | `experiments/protocols/asynchronous_confirmation_v1.json`, sha256 `0121199ddad710856d336610a500b593b491dfe3b2eab3c022fed879ae9b187d`; status `BLOCKED_NO_OFFICIAL_RUNTIME_FOR_THIS_DESIGN` |
| SmolVLA schedule | `experiments/runs/asynchronous_confirmation_v1/smolvla/schedule.json`, 1,200 rows, sha256 `5bde0b9efb3e963c42be05a0532ce638e0885730782d7ad5a9f59ca6c08d9d2a` |
| pi0.5 schedule | `experiments/runs/asynchronous_confirmation_v1/pi05/schedule.json`, 1,200 rows, sha256 `336de1127737d802ff08bd5fd89a73f27cda3938a4b2d0cc63ab19fe95203e7e` |
| Identities | The same as confirmation, as Alba's email specifies |
| Preflight requirements | Not defined; no runner exists |
| Run status | PLANNED; NOT STARTED; 0 of 2,400 |
| Available evidence | Schedules only; synchronous smoke timing is not asynchronous evidence |
| Blocking conditions | R1: no official runtime found for both policies in simulation; both earlier studies complete and audited |
| Next permitted action | Alba names the runtime, accepts a custom harness, or narrows or drops the cohort |

The R-numbered decisions are defined in `review/CROSS_POLICY_CONFIRMATION_AUDIT.md`, section 4, and R16 in `review/GPU_PREFLIGHT_RESULT.md`, section 5.
