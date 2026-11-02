# Engineering fixes, 2 November 2026

Branch `warren/cross-policy-confirmation`. Local commits on top of pushed `6ad02ec`: `31748c8`, `24288bd`, `caebdd9`. Not pushed.

Final test result at `caebdd9`: **51 tests, OK**, Python 3.12 with NumPy through `uv`. The 50 tests at `24288bd` also pass from a clean `git archive` export. `build_protocols.py --check` reports `checked 9 files`. A `git diff` of `experiments/protocols` and `experiments/runs` against `6ad02ec` is empty.

**No fix has been exercised with a real model, simulator or GPU.** Tests use stubs and fakes, and the first real exercise will be the full preflight.

| # | Defect | Commit | Failing first? | Scientific semantics changed? |
|---|---|---|---|---|
| 1 | Client cannot import LIBERO | `31748c8` | Yes, same traceback | No |
| 2 | Reset-state mismatch recorded, not enforced | `24288bd` | Yes | No; see note |
| 3 | A failed attempt does not stop the run | `24288bd` | Yes | No; see note |
| 4 | Release does not bind the schedule | `24288bd` | Yes | No |
| 5 | Analysis omits required outputs | `24288bd` | Yes | No |
| 6 | Provenance loses the commit in a bundle | `24288bd` | Yes | No |
| 7 | pi0.5 server writes no provenance | `caebdd9` | Yes | No |

## 1. Harness path error

- **Failure:** `alba-preflight-2`: `ModuleNotFoundError: No module named 'libero'` at `preflight_pi05_client_small.py` line 29.
- **Root cause:** `envs/client` does not install LIBERO. OpenPI's example takes it from `third_party/libero`, and `run_gpu_preflight.sh` never put that on `PYTHONPATH`. The rerun worked because its job command set it by hand.
- **Fix:** one variable on the client line of `run_gpu_preflight.sh`.
- **Test:** `test_preflight_harness.py` runs the real wrapper and real client script in a fake workspace, with stubs for `nvidia-smi`, `flock`, the environments and the server. It also checks that the import resolves to the workspace's package.

| State | Result |
|---|---|
| Unfixed | `FAILED (failures=1)`, with the original traceback |
| Fixed | `OK` |
| Old script restored | `FAILED (failures=1)` |
| Fix re-applied | `OK` |

**Limit:** the fake package stands in for LIBERO. On the RTX, the job record of the rerun is the only real evidence that this path works.

## 2. Reset-state reuse

- **Defect:** `settled_state_matches_block` was written, the attempt still completed, and `analyze.py` never read it. The reference was also lost on restart.
- **Fix:** `attempt_loop.py`. The block reference is rebuilt from the journal, and a mismatch raises before any policy query. `analyze.py` independently recomputes the block hash and marks a contrast `incomplete`, and a missing hash never counts as a match.
- **Mutation check:** removing the check from `client_openpi.rollout` fails `test_real_client_rollout_refuses_before_any_policy_query`. Removing the analysis branch fails two tests.
- **Note:** this is the specification's "exact reset-state reuse". It adds another route to an `incomplete` test under the current R10 rule, which Alba has not ruled on.
- **Limit:** the same check in `runner_smolvla.rollout` is not directly tested, because that needs torch and LeRobot. pi0.5 uses a real client rollout with a mocked environment.

## 3. Stop on failure

- **Defect:** both runners caught each exception and continued. One server crash would journal every remaining planned attempt as failed in seconds, and those records can never be replaced. pi0.5's `unsupported_refinement_setting` stop rule was also not enforced.
- **Fix:** journal the failure, then stop with `AttemptStopped`.
- **Note:** nothing is deleted, retried or replaced. Stopping is operational, and for pi0.5 it is the protocol's rule. A restart skips the failed row, which stays failed and keeps its test `incomplete`.

## 4. Release binds the schedule

- **Defect:** `launch_guard.py` compared only the protocol hash.
- **Fix:** the release must list the exact schedule file in `schedule_sha256s`, and rollouts with no schedule are refused. One existing test was updated to the new release format.
- **Limit:** the release file can be written by whoever runs the campaign, so this guards against accidents only.

## 5. Analysis outputs

- **Added:** `paired_state_differences`; `timing` per condition, with host, CUDA and rollout kept separate; `UNAVAILABLE` where a quantity was not recorded.
- **SmolVLA rollouts:** `rp.Counters` now records generation host and CUDA intervals, as the historical runner did. **This has not been run.**
- **pi0.5:** per-generation memory is now journaled.
- **Not implemented, and marked in the output:** arithmetic per chunk, step and episode. This waits for R13 and is a stop condition, because pi0.5 has no counting method.

## 6-7. Provenance

- `provenance.git_commit` reads `BUNDLE_COMMIT`, because the RTX bundle has no `.git`.
- `server_openpi.py` writes `provenance.json` with versions, checkpoint tree and normalization hash when given `--output`.
- Parity journals now reference the real provenance hash rather than a hash of the output path.

## New infrastructure

Not defects, and none has been run:

| File | Purpose |
|---|---|
| `run_full_preflight.sh` | Mandatory preflight for both models. Refuses, exit 6, without Alba's release (tested) |
| `conditions` mode in both runners | All five conditions once on the smoke identity: valid shapes and exact reset reuse. Summary logic is tested |
| `audit_shared_instrumentation.py` | Read-only historical self-consistency checks |

Seeds 790011 and 790012 and state `official0` are the existing authored smoke identity, outside every confirmation identity.

## Not repaired

| Item | Why |
|---|---|
| Rendering choice in the scripts | R6 |
| `generate_states.py` | Never run; R7 is open, and it may need changes after N1 |
| CUDA time for pi0.5 | No JAX equivalent (R5) |
| Independent outcome verification | Not in the frozen protocol (N4) |
