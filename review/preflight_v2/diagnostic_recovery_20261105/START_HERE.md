# pi0.5 preflight diagnostic recovery, 5-6 November 2026

**Status:** non-counting diagnostics complete; **protocol v3 DRAFT / NOT RELEASED**.

- Branch `warren/pi05-diagnostic-recovery-20261106`, based on `warren/preflight-v2-review` at `4248b58`. It adds only this folder.
- For review. Not merged, no PR, and v3 is not approved.
- No run001, counted preflight, parity, episode, confirmation identity or outcome analysis.
- Frozen evidence unchanged: 27/27 checksums.
- GPU work ran under exclusive reservation R018, now released.

## Recommendation

**Execution route: GO, for Alba's review. Protocol v3 release: BLOCKED** on Alba's rulings, and on tooling not yet written.

1. **Why the frozen eager count runs out of memory.** Found and causally confirmed in A01; reproduced byte for byte in A02.
   - In eager mode every forward pass leaves a per-layer copy of the scanned weights live in a JAX cache: 99.99995 % of 15.75 GiB at d=2.
   - `jax.clear_caches()` frees all of it. Removing and restoring the release makes the failure go and come back.
   - It is live-buffer retention causing real over-demand, not an intrinsically large model demand.
   - With the release, the frozen counter counts 2/5/10. → `01`
2. **A counting gate on the served compiled path works and rejects every wrong implementation tried.** (G02: 3 processes, 2 input cases, **0 wrong verdicts out of 162**.)
   - How it works: a data-dependent in-program count, plus bitwise equality with a reference whose step count is a static constant, plus d±1 exclusion.
   - Rejected implementations:
     - ignores `num_steps`;
     - runs d+1 steps;
     - right count with the wrong step size;
     - redundant head calls;
     - skipped network evaluations;
     - evaluations that are made but discarded;
     - a stale backbone on one step.
   - The stock sampler and a positive control pass.
   - The `received_num_steps` echo is set by the wrapper itself, so it is evidence of nothing. → `03`
3. **The wrapper's output is bitwise equal to stock pi0.5 at native d=10** (two inputs). The stock and wrapper HLO differ (constant vs traced `num_steps`). → `04`
4. **Simulator.**
   - A fresh env per attempt is deterministic.
   - **A reused env is history-dependent, and stock's evaluator and the frozen parity mode both reuse envs.**
   - On this host EGL and OSMesa are the same software renderer, so **GPU-EGL equivalence cannot be tested here**.
   - Action units are consistent. The measured motion is about 0.9 cm per step at full command. → `05`
5. **Reproducibility.** Compiled outputs drift about 5e-3 across processes under default XLA flags. Two deterministic flags make them bitwise identical (D01, G02). → `06`
6. **The earlier LeRobot pi0.5 campaign.**
   - No records exist, so the defect cannot be attributed.
   - It shares no pi0.5 component with the current route.
   - SmolVLA can be neither cleared nor implicated. → `02`

## Decisions only Alba can make (`PROTOCOL_V3_DRAFT.md` section 9)

- **R6:** native Linux GPU EGL, or OSMesa preregistered as a separate environment.
- **R9:** parity and action-unit criteria, including whether the parity reference is stock's reused-env procedure.
- **R4:** the wrapper as the official path.
- **N1:** the state stack.
- **CNT:** compiled-path counting in place of v2's eager count, and whether to add per-request attestation.

## Not established

- **Inputs and hardware:** synthetic inputs only, one host, one GPU, JAX 0.5.3.
- **Probe-keyed cheating:** an implementation that behaves correctly only on the probe input would pass the gate; per-request attestation is proposed.
- **Shared code:** a bug in model code that the reference shares with stock is invisible to the gate.
- **Fire drill:** the gate's non-zero exit path was never triggered.
- **Speed:** the speed cost of the deterministic flags and of attestation is unmeasured.
- **Simulator:** the cause of reused-env history dependence is bounded but not named. GPU-EGL equivalence is unmeasured.
- **Not written yet:** `run_preflight_v3.sh` and `count_compiled_pi05.py`.
- **Review:** both review rounds were by AI reviewers, not a human.

## Files

- `RUN_LEDGER.md`: every diagnostic, with failed approaches and my own defects.
- `01`-`06`: the phase reports.
- `PROTOCOL_V3_DRAFT.md`.
- `07_INDEPENDENT_REVIEW.md`: two rounds and every disposition.
- `drivers/`, `runs/<ID>/` (raw outputs and the exact job command per run), `SHA256SUMS`.
