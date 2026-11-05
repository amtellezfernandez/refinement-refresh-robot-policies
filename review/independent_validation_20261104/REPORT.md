# Independent validation of the SmolVLA confirmation evidence (4 November 2026)

**Verdict: READY FOR ALBA REVIEW. NO ARCHIVED PAPER-EVIDENCE REPLAY HAS BEEN RUN.** The audit and
harness are built and tested. The protocol is frozen at commit `c92193f` (PROTOCOL.md sha256
`ede3eb8570487b7e2eda8e47f39c6afc1d12e40bece590e8c4590d34341ea737`), with one pre-approval harness
amendment (`protocol/PROTOCOL_AMENDMENT_01.md`). This is not an independent confirmation of the paper's outcomes.

Branch `warren/independent-replay-validation`, based on Alba's `da87239`.
None of Alba's files were modified. Everything new is under `review/independent_validation_20261104/`.

## A. Independent evidence audit

`audit/independent_audit.py` uses only the standard library and imports none of the repository's analysis
code. It re-derives outcomes from per-step flags and re-encodes every action and initial-state
array to recompute its hash.

Result: **22/22 checks pass** (`audit/independent_audit_result.json`). Negative controls: **11/11
planted defects detected** (`audit/audit_mutation_results.json`). The first control run failed 1 of 10.
The failure was my expected set, because the dropped record was one of the two repeats. It also exposed an
audit crash on a missing record, which is now fixed. That run is kept in `test_runs/`.

| Paper claim | Our check | Result | Tier |
|---|---|---|---|
| 1,120 confirmation attempts, 7 configs × 8 tasks × 10 states × 2 seeds | plan recount + canonical selection re-derived | 1,120, exact cover, matches archived canonical file | recomputed from raw |
| 139/160 (d=2) vs 122/160 (d=5) at h=2 | per-step success flags | 139 vs 122; 24 vs 7 discordant pairs | recomputed from traces |
| +10.6 pp | | 0.10625 | recomputed |
| 95% interval 4.4 to 16.9 (task-conditional) | own bootstrap, 2 RNG seeds | [4.375, 16.875] both seeds | independent implementation |
| 138/160 at (5,2) | | 138 | recomputed |
| Secondary and prediction tables (+2.5, +0.6, −1.3, +1.3, +3.1, +0.6) | condition counts 137/136/135/139/122/138/133 | all point estimates match | recomputed; intervals not recomputed |
| Per-task table | | all 8 rows match | recomputed |
| 1,122 raw episodes, 172,243 steps | | match | recomputed |
| Paired initial states | hash per block, per task-state | identical within block and across seeds; 80 distinct | recomputed |
| No stop before success / failures run to horizon | per-step flags | success only at final step; failures = 280 / 300 | recomputed |
| Two repeats excluded | | both repeats give the same outcome and step count as the kept attempt (158 and 280) | recomputed |
| 25/40 native, 28/40 best in discovery | episode records | match | **episode summaries only; discovery traces not audited** |
| 5.9%, 82.1%, 60.7% arithmetic | | **not checked** | not applicable |
| Outcome = LIBERO `check_success()` | pinned wrapper fetched, sha256 matches archive | source-level | static |
| Task files and init states | our install vs archived hashes | 8/8 BDDL, 8/8 init-state arrays match | static |

Two manifest observations:
- The confirmation manifest records `numpy_version` 2.2.6 and `torch_version` 2.7.1 for the
  imported modules. Its package list shows numpy 1.26.4 and torch 2.6.0, so two
  distributions were visible. The replay environment matches the imported versions.
- The LIBERO asset tree is not hashed in the archive. Only one asset revision existed before the
  campaign (`0b3ea86`), so I infer it was used; this is not recorded.

What the audit does **not** establish: that the instrument was right. A recount from the same
records shares their source. The replay is the step that re-derives outcomes by another route.

## B. Replay feasibility verdict: FEASIBLE, with one open host question

- **Sufficient:** the archive holds the post-settle initial state, every float32 action in order, the
  per-tick success flags, the environment seed, and the task and init-file hashes.
- **Hidden state:** the archived state is not a complete checkpoint. The Panda gripper's
  `current_action` (integrates 0.01 per step) and OSC `goal_ori` (updated only on non-zero
  rotation) are outside the MuJoCo state vector. The harness therefore regenerates them
  through the wrapper's own reset path. Measured on fixtures: injecting the state without the
  settle steps moved the success tick in 2/15 trajectories.
- **Reset semantics:** reset, termination, control frequency (20 Hz), the 280/300 horizons and the
  no-op settle action all come from the pinned wrapper source. Clipping happens inside robosuite, and
  raw float32 actions are fed unchanged (some archived values reach −1.09).
- **Determinism:** same-platform replay is bit-identical. Two full reruns, render on vs off, and
  fixture regeneration in a fresh process all gave identical terminal-state hashes.
- **Open:** cross-platform fidelity. The original was x86_64 Linux and the replay host is arm64
  macOS. This is unknown until Stage 0. If Stage 0 fails, the protocol stops and needs an
  x86_64 Linux host.

## C. Replay harness

`replay/`:
- `libero_replay.py`: reset path and stepping.
- `replay.py`: input validation, verdicts and the governance gate.
- `run_protocol.py`: stage orchestration and stage gates.
- `extract_bundles.py`: archive to bundles, plus the frozen sample.
- `make_fixtures.py`, `test_harness.py`, `drill_protocol.py`.

Environment: a separate virtualenv (pinned, `protocol/replay_env_freeze.txt`; setup in `README.md`), assets at
`0b3ea86`, tree manifest sha256 `ad609f41…a503`.

All 1,120 archived bundles pass static input validation: 0 defects, 0 duplicate keys, 0
duplicate action streams. The gate refused them (exit 5, no output written).

## D. Adversarial test results

Fixtures: 20 trajectories recorded on this machine from tasks outside every paper task set
(Object 2, Goal 1). There are 15 successes from LIBERO human demonstrations played open-loop, plus
5 failures (zero, random, and one demo that failed open-loop).

- Harness suite: **run 003, 36/36** at the freeze; **run 004, 40/40** after amendment 01 (4 new gate-refusal tests).
  - Run 001 failed 3/36: a renumbered missing step was not detected.
  - Run 002 failed 3/36: my forgeries became inconsistent after the stream-hash check was added.
  - Both failed runs are kept.
- Protocol drills: **5/5.** Stage-logic test for CHAOTIC_INCONCLUSIVE, with the probe stubbed: **4/4** (its first run failed on a stub defect and is kept).
  - Clean data: all stages run.
  - A forged start stops at Stage 0.
  - A planted mislabel gives DISAGREEMENT, and Stage 2 never starts.
  - Drift plus a mislabel: the adjudication probe finds the archived outcome unreachable (0/10).
  - A missing release refuses with exit 5.
- Measured power limits:
  - A fully consistent deletion forgery was detected in 2/15.
  - A consistent one-tick action change was detected in 0/15 at 1e-7, 1/15 at 1e-4, and 5/15 at 0.3.
  - Replay cannot authenticate action streams. Hashes and the two separate records carry that.

## E. Frozen protocol

`protocol/PROTOCOL.md`; the lint reports the form complete; the cited gate lines branch.
- Stage 0: reset fidelity on 80 task-states.
- Stage 1: 57 attempts, stratified by task × condition × outcome, seeded and hashed.
- Stage 2: all 320 primary attempts.
- Arms: reset_path is primary, inject is secondary.
- RESET_TOL is 1e-6, set before any archived comparison.
- Verdict table covers every case. No reruns count, and stop on confirmation.

Disclosure: during setup, one reset of Object 7, state 24 printed three numbers that agree with
the archived start at printed precision. No action was applied, and no formal comparison has been run.

## F. Unresolved dependencies

1. Alba's approval (release bound to the protocol hash).
2. An x86_64 Linux host, needed only if Stage 0 fails on arm64. None is available to me at present.
3. The asset revision is inferred, not recorded.
4. numpy and BLAS differ by platform. robosuite's OSC uses numpy linear algebra, and Stage 0 tests this.
5. The CHAOTIC_INCONCLUSIVE branch is tested with a stubbed probe only, because no fixture flips under 1e-9 start noise. A missing release file now refuses cleanly (amendment 01).
6. Discovery traces and the arithmetic claims are not audited.

## G. Three ranked opportunities

1. **Blind human check of the discordant pairs.** The effect rests on 31 discordant pairs (24 vs 7).
   Stage 2 already records terminal predicates. Render each terminal frame, then have a person label
   success with the condition hidden. This is the one check that does not use the same BDDL predicate
   as the instrument. Cost: minutes of compute plus about an hour of a person's time. Needs approval.
2. **Outcome stability of fixed action streams.** Replay each primary attempt with 10 perturbations
   of 1e-9. If the discordant pairs are concentrated among chaos-sensitive attempts, part of the
   paired gap is simulator sensitivity. If not, the gap is insensitive to it. About 2 CPU-hours. Needs approval.
3. **Outcome variance across policy noise.**
   - Post hoc: the two noise seeds share each start, and agree in 69/80 states at d=2 but only 52/80 at d=5.
   - Proposed: a preregistered study with more seeds per state would test whether fewer denoising
     steps reduce outcome variance, which is a mechanism candidate for the gain.
   - Cost: new GPU rollouts, so it needs Alba and a GPU.
   - This is a hypothesis, not a finding.

Post-hoc descriptive results (`audit/descriptive_reanalysis_POSTHOC.json`, not preregistered):
- The gap is 8.75 to 11.25 pp for every horizon cap from 150 to 300.
- Where both succeed, mean success tick is 123.6 vs 123.9.
- Task-random t interval: [5.0, 16.3] pp.
- Cluster sign-flip permutation by task-state: p = 0.004.
- Task sign test: 7 tasks positive, 0 negative, 1 zero (Object 5, 20/20 in both).
- McNemar p = 0.003 treats the 160 pairs as independent, which they are not: seeds share starts.

## H. Can proceed without approval

Re-running the audit and the harness tests; more fixtures; extending the audit to the discovery
traces; descriptive re-analysis of archived records; building an x86_64 environment if a host becomes
available. I have chosen to keep Stage 0 behind the gate, although it applies no actions, because it
compares against archived evidence and its result shapes the run.

## I. Awaiting Alba's approval

The current implementation requires an external release bound to the exact
protocol, archive bundles, sample manifest, executable files and protocol
amendments. A Stage 0-only decision must use `scope: [0]`; this does not
authorize Stages 1 or 2. No release file exists and no archived reset or action
replay has run. The current runtime-provenance edits have not yet been
qualified.
## Implemented / executed / checked / unverified

- **Implemented:** the audit, 11 audit mutations, the harness, the fixtures, 36 harness tests, 5 drills, the protocol, and the post-hoc re-analysis.
- **Executed:** all of the above, on one arm64 macOS workstation.
- **Independently checked:** the claims marked *recomputed* in A. Task-file hashes. Harness detection of every gated defect.
- **Unverified:**
  - outcome-instrument validity (the replay is not run);
  - cross-platform fidelity;
  - the discovery traces;
  - the arithmetic claims;
  - whether the BDDL predicate matches human judgement.


## I. Runtime provenance correction (5 November 2026)

Review found that `experiments/tooling/run_paired.py` made a single-value
package map from every visible distribution record. Duplicate package
metadata could therefore overwrite one another, while separate top-level
`numpy.__version__` and `torch.__version__` fields described the modules
actually imported. The archived confirmation manifests show this conflict:
imported NumPy/Torch are recorded as 2.2.6/2.7.1, while the package map says
1.26.4/2.6.0. The archive does not retain module paths or all distribution
candidates, so this cannot be resolved retroactively from the manifest alone.

The PR now changes future experiment manifests to retain duplicate distribution
candidates, record imported module versions and file paths, and hash the
resolved LIBERO assets tree. The replay gate also verifies imported NumPy,
Torch and MuJoCo versions and their installation roots alongside distribution
metadata. These code changes are on the current PR branch but are **not covered
by the earlier local qualification results**.

The original experiment's LIBERO asset tree was not hashed; its revision remains
unknown. The replay asset tree is pinned and checked independently. A passing
Stage 0 on arm64 can therefore establish reset agreement only in the declared
replay environment, not identity with the original assets or full
cross-architecture trajectory equivalence. GitHub Actions is not required by
the protocol; the new source still needs review and local validation before any
stage-scoped approval.
