# R22/R25 execution plan, 2 November 2026

**Not authorised and not ready.** Commands are for review and none has been run. `<RID>`, `<COMMIT>` and `<RELEASED_COMMIT>` are filled in at the time.

## What the frozen design specifies

`experiments/protocols/cross_policy_confirmation_v1.json`: primary contrast R22 minus R25, h=2, d=2 vs d=5, 400 paired identities per model, seeds 810001 and 810002, models analysed separately.

## Comparison requirements

| Requirement | Status | Evidence | Remaining |
|---|---|---|---|
| Identical task definitions | By design within model | Both arms share `episode_identity`; `task_bddl_sha256` recorded by the state generator | Not run |
| Matched initial states | **BLOCKED** | Rows share `state_key`, all `state_sha256` null | R7, N1, generation |
| Exact reset reuse | Enforced in code | `attempt_loop.py`; `analyze.py` recomputes | Not exercised on a simulator |
| Specified checkpoints | SmolVLA: revision and source hashes checked at load. pi0.5: normalization hash at load; tree hash in server provenance | `run_paired.py`, `server_openpi.py` | Run |
| Same horizon within model | By design | SmolVLA `_max_episode_steps`; pi0.5 `MAX_STEPS` (220, 280, 300, 520) | **Differs across models** where SmolVLA's limit differs, so cross-model is signs only (R11) |
| Same scoring within model | By design | SmolVLA `info` success; pi0.5 `done` | **Differs across models** |
| Prespecified assignment | Yes | Hashed order in schedules; condition order hashed within block | Schedule hash changes on binding, order does not (tested) |
| Same noise key across arms | By design | Keyed by identity and tick; both arms generate on the same ticks at h=2 until termination | Not exercised |
| Independent outcome verification | **NOT IMPLEMENTED** | Not in the frozen protocol | N4 |

## Remaining confounders

| Confounder | Effect on R22 vs R25 | Handling |
|---|---|---|
| MuJoCo differs by model (N1) | None within model; matters for cross-model reading | Alba |
| OSMesa vs EGL against historical evidence (R6) | None within the new campaign; not pixel-comparable with old | Alba |
| Earlier SmolVLA population differs | New states, seeds, suites and tasks, so old (2,2) vs (2,5) is not replicated like-for-like | Report separately; never pool |
| Environment reuse in the pi0.5 client | `env.seed` once per environment, so attempt order could matter | Settled-state check catches state differences only; rendering or other RNG is not covered. Check at preflight |
| Other CPU load | Timing only | Exclusive reservation; report |
| Wrapper differs from stock server (R4) | Same in both arms | Alba |
| Failure and stop rule (R10, N5) | Can leave the test incomplete | Alba |
| Interval undercoverage (R3) | Primary interval narrower than nominal | Alba, before data |
| Shared LeRobot defect | Could affect SmolVLA both historically and here | Audit once the reason is known |

## Subset question (N3)

Schedules hold all five conditions in hashed order, and the runners have no condition filter. Running only R22 and R25 needs Alba's choice and a new frozen schedule. I did not add a filter, because it would change the attempt schedule.

Machine time, extrapolated from limited observations: R22 vs R25 both models 6-28 h; all five conditions 17-76 h.

## Commands, in order (do not run yet)

Each `job run` needs the GPU release. The transfer needs Warren's single-use approval.

Full preflight, after items 1-10 in `NEXT_PREFLIGHT_READINESS.md`:

```bash
job run rtx alba-full-preflight-1 --ram 24 --vram 12 --reservation <RID> \
  --purpose "Alba mandatory preflight, no confirmation identities" -- \
  'ALBA_GPU_RESERVATION=<RID> bash $HOME/alba-cross-policy-20261101/code-<COMMIT>/experiments/tooling/cross_policy/run_full_preflight.sh $HOME/alba-cross-policy-20261101 $HOME/alba-cross-policy-20261101/full-preflight-<COMMIT>'
```

States, on the chosen host, after R7 and N1 (CPU):

```bash
$W/envs/smolvla/bin/python experiments/tooling/cross_policy/generate_states.py \
  --cohort cross_policy_confirmation_v1 --output experiments/runs/cross_policy_confirmation_v1/states
python3 experiments/tooling/cross_policy/build_protocols.py
```

SmolVLA, after Alba's release:

```bash
ALBA_GPU_RESERVATION=<RID> $W/envs/smolvla/bin/python experiments/tooling/cross_policy/runner_smolvla.py attempts \
  --phase confirmation --checkpoint $W/ckpt/smolvla_libero \
  --schedule experiments/runs/cross_policy_confirmation_v1/smolvla/schedule.json \
  --states experiments/runs/cross_policy_confirmation_v1/states/initial_states.npz \
  --output $W/runs-<RELEASED_COMMIT>/smolvla
```

pi0.5, one model at a time. The server runs under the preflight guard phase, and the client enforces the release:

```bash
ALBA_GPU_RESERVATION=<RID> XLA_PYTHON_CLIENT_PREALLOCATE=false $W/envs/openpi/bin/python \
  experiments/tooling/cross_policy/server_openpi.py serve --checkpoint $W/ckpt/checkpoints/pi05_libero \
  --output $W/runs-<RELEASED_COMMIT>/pi05_server --port 8000 &
ALBA_GPU_RESERVATION=<RID> PYTHONPATH=$W/src/openpi/third_party/libero $W/envs/client/bin/python \
  experiments/tooling/cross_policy/client_openpi.py attempts --phase confirmation \
  --schedule experiments/runs/cross_policy_confirmation_v1/pi05/schedule.json \
  --states experiments/runs/cross_policy_confirmation_v1/states/initial_states.npz \
  --output $W/runs-<RELEASED_COMMIT>/pi05 --port 8000
```

`analyze.py --run-root` expects `<model>/schedule.json` and `<model>/attempts`, so the outputs must be arranged under one run root before analysis.

## Validation checks before any result is read

1. `launch_guard` refused before release, by a recorded fire drill.
2. Every planned row has exactly one record; none is `planned` or `running`.
3. Status counts; all failures listed with reasons.
4. `reset_state_failures` empty; `analyze.py` status not `incomplete`.
5. Each record's `provenance_sha256` matches a saved provenance file with the released commit.
6. Schedule file hash is in the release; protocol hash matches.
7. Generations match `ceil(steps / h)`; for SmolVLA, `denoise_steps = generations x d`.
8. Journal checksums recorded before analysis, as the audit requires.
9. Models analysed separately, signs only across models.

## Independent outcome verification (proposal for N4; not implemented)

Replay each saved action array from its bound initial state, in a fresh process with no policy, and compare success and episode length with the journal. This covers recording errors. It shares the simulator, and does not check whether the policy produced correct actions.

Rollout outcomes must not be read before steps 1-8 pass.
