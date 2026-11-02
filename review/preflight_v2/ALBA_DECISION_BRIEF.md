**Superseded by `run001_cpu/PREFLIGHT_V2_STATUS.md` after Alba's latest email. Kept as a record.**

# Decision brief for Alba: cross-policy preflight v2

2 November 2026. Based on your PR #1. Time estimates are extrapolated from one easy task. No confirmation or GPU work has started; only CPU-only checks with no policy weights have run.

**We need four answers to start the preflight (A1 to A4), and the invalidation reason as early as you can give it.**

Your PR settles, and we will not ask again: 800 attempts per model and 1,600 in total, not 800; freshly generated states; a calibrated bootstrap chosen before outcomes; CUDA timing and pi0.5 XLA arithmetic reported as gaps.

## A. Yours, needed before preflight

**A1. R9: What must native parity meet, and against which reference?** Your PR asks for criteria against the reference evaluator, preregistered before outcomes. No pi0.5 reference exists in the repository.
- **Options:**
  - You set numeric criteria and a reference per model. This gives a mechanical verdict; a wrong reference could fail a sound stack.
  - We draft them for your approval before the run. This is quicker, and the choice stays yours.
- **Engineering test:** none can set this.

**A2. R6: Native Linux EGL, or OSMesa preregistered as a separate evaluation environment?**
- **Options:**
  - EGL: same renderer as your runs. It needs another host, not yet identified, and more engineering.
  - OSMesa: ready on the RTX. It is not pixel-comparable, and is reported as a separate environment.
- **Engineering test:** informs only. One state's first-action result is below the noise range, which you note is no parity test.

**A3. N1: One fresh state set, or one per model?** SmolVLA pins MuJoCo 3.8.1 and the pi0.5 client 3.2.3. R22 vs R25 stays paired within each model either way. This affects what the cross-model sign comparison means.
- **Options:**
  - One set generated in the SmolVLA stack: shared states, but pi0.5 runs them under another simulator version.
  - One set per model: each native to its stack, with no shared states.
  - Align versions: one physics, but departs from a pinned lock.
- **Engineering test:** partly. One generated state per task restored exactly in the client stack (2 tasks). A CPU probe could compare settling across stacks.

**A4. R4: Does the noise wrapper count as the official pi0.5 path?** The stock server cannot pair noise by identity and tick.
- **Options:**
  - Accept and disclose: noise stays paired.
  - Stock server, one per d: closest to stock, but noise is unpaired, which weakens the design.
- **Engineering test:** not for this definition. It does cover execution, below.

## B. Yours, as early as possible

**B1. Why was the earlier pi0.5 campaign invalid, where are its records, and could the defect reach `run_paired.py`, LeRobot or SmolVLA?** Only you hold this. It might add a preflight check, and SmolVLA validity stays pending until then.

## C. Yours, before confirmation only

| Item | Question | Options |
|---|---|---|
| R10 | How do failed attempts enter the test? | Incomplete (current); recorded retry for infrastructure faults; count as unsuccessful |
| R3 method | Which calibrated method? | We propose one with a synthetic coverage study, for your approval |
| N4 | Independent outcome verification? | None; or replay saved actions with no policy, which catches recording errors only |
| R13, SmolVLA | Report calibrated arithmetic, within model? | Yes; or a gap, as for pi0.5 |
| R8 | Identity overlap with the invalid campaign | Share its seeds and states; or disclose the limit |

## D. Engineering we own

**Prepared on a local branch, not yet on GitHub or merged:**
- Your PR merged with our later work, keeping your R22/R25 route and release format, and retiring our separate cohort.
- `run_gpu_preflight.sh` restored byte-for-byte to the counted run's version.
- Client on OpenPI's own LIBERO tree: 40 of 40 tasks identical to the previous config.
- Fresh pi0.5 environment per attempt, because reuse broke exact reset on `libero_object`.
- Your environment check: one line changed, because `libero.__file__` is `None` in this client and raised `TypeError`.

**Still to do:**
- Measure the denoising count with real weights. So far, pinned OpenPI on a dummy model made 2, 5 and 10 loop calls.
- First real run of the new code.
- Implement the calibrated bootstrap, after you approve a method.

Details: `RECONCILIATION_20261102.md`, `DECISION_REGISTER.md`, `PREFLIGHT_V2_SPEC.md`, `N2_PROBE_20261102.md`.

## Reply format

A1: / A2: / A3: / A4: / B1:
