# Post-submission cross-policy safety repairs — 2 November 2026

This document is **code review guidance, not a preflight verdict**. The original
1 November counted engineering run remains `PREFLIGHT_FAILED`. The mandatory
preflight remains **INCOMPLETE**. There are **no new confirmation outcomes**.

## Provenance and boundaries

- Change branch: `audit/preflight-schedule-reset-20261102`, based on Warren's
  `warren/cross-policy-confirmation` handover branch.
- The archived `release/submitted.pdf`, previous SmolVLA journals, prior run
  directories, frozen predictions, and original `run_paired.py` are untouched.
- The invalid 160-attempt LeRobot PyTorch pi0.5 campaign remains excluded.
- SmolVLA 1,120-attempt results remain
  **ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING**.
- No preflight, confirmation, observation-age, asynchronous, or third-policy
  rollout has been run as part of these repairs.
- This branch contains **no launch release**.

## Confirmed new-harness defects and corrections

| Defect | Repair | Verification still needed |
| --- | --- | --- |
| Both new rollout runners only recorded mismatched resets, rather than stopping the attempt | Compare paired settled-state **and first-observation hashes before inference**, fail the attempt if different | Real reset tests in mandatory preflight |
| Launch release checked only the protocol hash | Bind each model to both the full schedule file SHA-256 and canonical row digest; validate identity population, conditions, counts, state binding and paired state hashes | Create release only after independent preflight review |
| The current full schedule contains five experimental conditions, while only R22/R25 is authorized next | Deterministically derive separate 800-attempt-per-model primary schedules; refuse other arms on the first release | Generate after states are bound, verify manifest and hashes |
| Analysis could treat a missing contrast arm as a completed result, and did not reject a completed attempt with a failed reset check | Return `not_planned` if an arm is absent; exclude unmatched or reset-unverified attempts from success analysis | Compare against raw attempt journals |
| New SmolVLA runner did not reproduce all settings in the original `run_paired.py` loop | Restore conditional AMP and deterministic cuDNN/TF32 settings and seed setup | Verify policy precision, action traces, actual solver NFE and original native parity |
| pi0.5 request d was not echoed back from its server | Echo server-configured `num_steps`; client rejects missing/mismatched echo | **Echo is not proof** of JAX internal denoising-loop count; independently instrument or validate against pinned OpenPI source |
| Counted client failed to import LIBERO | Add a CPU-only import/source-root check for the actual client interpreter | Warren must install/pin a compatible LIBERO import and run the check *before* allocating GPU; do not mix Python 3.8 with Python 3.12 binary site-packages |

Warren's **counted** OpenPI error was `ModuleNotFoundError: No module named 'libero'`.
The later rerun worked with `PYTHONPATH` set. That identifies a new-client
environment defect, not an established defect in the original SmolVLA
experiment. The old runner already checks paired reset hashes before execution,
cross-checks action producer/consumer records, and counts denoising calls at the
model boundary. These checks still do not independently validate all historical
instrumentation.

## Mandatory next steps for Warren (before any science rollout)

1. Use the **actual client interpreter** to run
   `python experiments/tooling/cross_policy/check_openpi_client_environment.py --expected-libero-root <pinned-client-LIBERO-source>`.
   Record interpreter, import path, dependency lock, dataset hashes and output.
   Do not patch the frozen `run_gpu_preflight.sh` retrospectively or reinterpret
   `alba-preflight-2`; author a **new, versioned** full-preflight entry point.
2. Reconcile `R4`: inspect the pinned OpenPI `Policy.infer` and
   `Pi0.sample_actions` together with the new server. Confirm actual `d=2/5/10`
   solver execution, complete integration interval, normalization and action
   units. The response echo alone is insufficient.
3. Reconcile `R6`: run with a native NVIDIA EGL renderer if an equivalent
   reference stack is required, or explicitly preregister OSMesa as a separate
   evaluation environment. The one-state pixel/action diagnostic is **not**
   an evaluator-parity acceptance test.
4. Reconcile `R7`: the author's original protocol requires **freshly generated
   state slots**, rather than an unapproved substitution of official slots.
   Validate every generated state's settling, task validity, serialization,
   reproducibility and disjointness with archived cohorts. Propose a versioned
   protocol amendment **before results** if different states are necessary.
5. Reconcile `R9`: preregister native-parity acceptance criteria against the
   reference evaluator before looking at outcomes. Report per-task success,
   first actions, units, images/videos and reset reproducibility over all four
   suites. Do not infer parity from successful smoke episodes.
6. Complete the required 100-warmup/100-measured profiling, full action-chunk
   measurements, action-shape/finite checks, R22/R25/R52/R55 smoke checks, and
   per-condition reset comparisons. Audit a new versioned verdict. The prior
   `PREFLIGHT_FAILED` remains unchanged.
7. Once state manifests are bound, derive **separate** primary schedules for
   `smolvla` and `pi05` with `make_primary_schedule.py`. Each must contain
   **400 identities × 2 conditions = 800 planned attempts**; total across both
   models is **1,600**, correcting the 800-across-both wording in Alba's email.
8. Send preflight commit, protocol hashes, per-model primary schedule hashes,
   native-parity report, failure records and manifest **before** anyone creates
   `review/releases/cross_policy_confirmation_v1.release.json`.
   A matching release must include `approval_scope=PRIMARY_R22_R25_ONLY`,
   `approved_conditions=["R22","R25"]`,
   `approved_attempts_per_model=800`, `protocol_sha256`,
   `schedule_sha256_by_model`, and `schedule_file_sha256_by_model`.
   This example is a **format**, not an authorization. Do not create a release
   on this repair branch.

## Verification limits

The automated CPU-only tests exercise schedule derivation, release binding,
write-ahead journal rules, synthetic paired analysis and the OpenPI server
configuration echo. They cannot demonstrate real GPU parity, native OpenPI
normalization, correct physics or confirmed robot-policy outcomes.

The 160 invalid pi0.5 attempts are not in the repository. Their missing raw
records cannot be reconstructed from this repair branch. The SmolVLA evidence
must stay conditional until the original shared evaluator and journaling path
has been independently audited.

**Post-submission only. No accepted PDF or historic evidence is modified.**
