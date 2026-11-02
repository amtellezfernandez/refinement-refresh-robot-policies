# Preflight v2 status: PREFLIGHT_INCOMPLETE

Protocol `experiments/protocols/cross_policy_preflight_v2.json`. CPU stage `cross_policy_preflight_v2.run001.cpu`; GPU stage `cross_policy_preflight_v2.run001`, **not started**. 2 November 2026.

**This is not a pass and not a failure.** No GPU work, policy rollout or confirmation attempt ran in this stage. The 1 November run stays PREFLIGHT_FAILED, with its records untouched. The invalid pi0.5 campaign stays excluded. The submitted PDF and frozen schedules are unchanged.

## What happened to the CPU stage

| Job | Outcome |
|---|---|
| `alba-v2-run001-cpu-prov` | Done, exit 0, 09:53:02Z to 09:53:14Z |
| `alba-v2-run001-cpu-weights` (pi0.5 normalization and d count on real weights) | **Failed, exit 143 (SIGTERM) at 09:54:37Z, with no result** |
| `alba-v2-run001-cpu-states` (generated states) | **Failed, exit 143 (SIGTERM) at 09:54:37Z, with no result** |

Both were terminated in the same second. About two minutes later the host's WSL returned `Wsl/Service/E_UNEXPECTED`, and the RTX became unreachable. The cause is not established. Earlier in that WSL session's kernel log there was a global out-of-memory kill of a 45.7 GiB Python process; its time relative to these jobs is not established. Under the frozen rule these two jobs are recorded as failed and are not treated as passed.

## Your required checks

| # | Check | SmolVLA | pi0.5 | Evidence |
|---|---|---|---|---|
| 1a | Exact checkpoint provenance | **PASS** on hash | **PASS** on hash | `model.safetensors` `71d9563c...f52f`; pi0.5 16 files hashed, listing below. CPU job |
| 1b | Dependency provenance | **PASS** | **PASS** | Installed lists: 160 (SmolVLA), 240 (OpenPI server), 125 (client) packages, with hashes below |
| 2 | Official pi0.5 normalization | n/a | **PARTIAL** | File hash `b3a44bb2...bd84` matches; config uses asset `physical-intelligence/libero`, quantile mode. **Loaded policy's statistics: NOT RUN** (job terminated) |
| 3 | Native evaluator parity, four suites | **NOT RUN** | **NOT RUN** | Needs GPU. No criterion specified |
| 4 | Action units and shapes | **NOT RUN** | **NOT RUN** | Needs GPU and a unit criterion |
| 5 | d=2, 5, 10 reach the implementation | **NOT RUN** in v2 | **PARTIAL** | Pinned OpenPI `Pi0` on a dummy model: 2, 5, 10 loop calls (`alba-v2-probe-7b`). **Real weights: NOT RUN.** SmolVLA journals are self-consistent on this count, which is not independent validation |
| 6 | First-action hashes, matched resets | **NOT RUN** | **NOT RUN** | Needs GPU |
| 7 | Reset reuse across conditions | **PARTIAL** | **PARTIAL** | Simulator only, no policy, sampled tasks: settled state and first observation repeat exactly (`probe-4`, `-5`, `-7c`). pi0.5 needs a fresh environment per attempt. Across conditions: NOT RUN |
| 8 | R22, R25, R52, R55 smoke | **NOT RUN** | **NOT RUN** | Needs GPU. The 1 November smoke covered NATIVE, R22, R25 in a failed run |
| 9 | Video inspection | **NOT RUN** | **NOT RUN** | Needs GPU, then a person |
| 10 | Per-call and rollout timing | **NOT RUN** | **NOT RUN** | Needs GPU |
| 11 | Generated initial-state hashes | **NOT RUN** | **NOT RUN** | Job terminated |
| 12 | Reset mismatch terminates the attempt | **PARTIAL** | **PARTIAL** | Unit and mutation tests. pi0.5 uses the real rollout with a mocked environment. Real simulator: NOT RUN |
| 13 | Release bound to protocol and schedule hashes | **PARTIAL** | **PARTIAL** | Implemented and tested synthetically. No release or bound schedule exists |

**Also run:** the client environment check on the real interpreter passes on OpenPI's own tree, and correctly fails on the old SmolVLA-environment config. Client on OpenPI's tree: 40 of 40 tasks identical to the old config (`N2_PROBE_20261102.md`).

**Your invalidation reason targets checks 2 and 5.** Both are implemented, but neither is complete with real weights.

## Open before the GPU stage

| Item | State |
|---|---|
| **Attempt count** | Your email says "Total: 800 attempts across both models". Its factors and PR #1 give 800 per model, 1,600 in total. Frozen schedules unchanged; `make_primary_schedule.py` and the guard follow PR #1 |
| Parity criterion, action-unit acceptance | Not in your email or PR |
| Renderer | PR #1 allows EGL, or OSMesa preregistered as separate. Only OSMesa is implemented |
| State stack, wrapper | Not addressed. Current code: SmolVLA-stack states and the server wrapper |
| GPU release | Warren's; not yet given |
| RTX host | WSL fault; needs recovery. Code must also be transferred, as the RTX still holds `code-bb86691` |

## Machine time, extrapolated from limited observations

GPU-stage preflight 6-26 hours, then R22 vs R25: about 6-28 hours at 1,600 attempts. It cannot finish in the current two-hour window. Not commitments.

## CPU stage hashes

From the job log, before the host failed. The files themselves are on the RTX and have not been retrieved.

| Artifact | SHA-256 |
|---|---|
| Client check as run | `50eeea52032fc88f8871616d13fa68ce39f52528e38113504b88c9aaf771fac2` |
| Package list 1 of 3 (displayed order) | `01dbda77deba2b609c1891512129893fde5dc2b8b75055a7e14b373a07d24731` |
| Package list 2 of 3 | `73fd9a1f87ce4a04103128bac814fc7cb03c35fbbb96a7fbaf90935282db8504` |
| Package list 3 of 3 | `ac110efdda699cc712b27ab3dac119169dbb3bdd927cea93ce2688e1ebeda73d` |
| Checkpoint listing 1 of 2 | `42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99` |
| Checkpoint listing 2 of 2 | `e273ca51cb5e143a22e9fe2e4733e9f25f7643a7714b114a7cfbe5bb7ed1ed51` |
| SmolVLA `model.safetensors` | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |
| pi0.5 `norm_stats.json` | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |

Runtime identity at the time: OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`; existing RTX bundle `bb866919673defea572e6aaf90208e921c8e2c2c`; GPU memory 720 MiB (idle) after the job.

## Limits

No check above is evidence of evaluator parity, policy quality or any R22 vs R25 result. SmolVLA stays ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING. New harness code has not run with real weights or on the GPU. Synchronous timing is not asynchronous or physical-robot latency evidence.
