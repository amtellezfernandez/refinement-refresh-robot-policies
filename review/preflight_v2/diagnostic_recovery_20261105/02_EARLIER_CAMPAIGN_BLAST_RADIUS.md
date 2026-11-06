# Phase C: the invalid earlier pi0.5 campaign, blast radius

**Scope.** Provenance and source only. No outcome was read, re-analysed or reconstructed. Diagnostic `C01`/`C02` (read-only source listings on the RTX, no model load).

## What the campaign was

| Item | Evidence | Status |
|---|---|---|
| Identity | "The invalid 160-attempt **LeRobot PyTorch** pi0.5 campaign" (Alba, `review/POST_SUBMISSION_CROSS_POLICY_REPAIRS_20261102.md:13`, commit `a34cc08`, her account) | Stated by Alba |
| Raw records | "The 160 invalid pi0.5 attempts are not in the repository. Their missing raw records cannot be reconstructed" (same file, line 96). `git log --all -S pi05` finds no commit before `9f36fdf` (1 Nov, ours); `review/CROSS_POLICY_CONFIRMATION_AUDIT.md` R8 | **Records absent** from the repository and from every branch |
| Stated invalidating reasons | "official normalization could not be established and requested denoising steps were not reliably verified" (Alba's email as relayed, `review/preflight_v2/DECISION_REGISTER.md:7`) | Stated by Alba |
| Host | Unknown. Not on the RTX workspace (`find ~ -iname "*pi05*"` lists only this project's 1-5 Nov jobs). Plausibly Alba's 4060 Ti | **UNSEARCHED** off the estate |

## The invalidating mechanism

**Established:** the campaign was invalidated because two properties could not be **verified**: (1) that the loaded normalization was the official one, and (2) that the requested number of denoising steps actually executed. That is an *unverified instrument*, not a demonstrated defect.

**Not established:** whether either property was actually wrong. Without the records, the run configuration or the LeRobot revision, no defect can be demonstrated or excluded.

**Context, not bound to the campaign.** A LeRobot checkout on the RTX (`~/lerobot`, `c441ebb`, committed 2026-10-08, clean) has a PyTorch pi0.5 port with `normalization_mapping` STATE/ACTION = `QUANTILES` (`configuration_pi05.py:92-96`) and `num_inference_steps = 10`, overridable per call (`modeling_pi05.py:945-950`). Nothing binds this checkout to the campaign, so it says nothing about what that campaign ran.

## Which measurements are affected

Every outcome of the 160 attempts. Both unverified properties act on every action: normalization maps every model output to controller units, and the step count is the experimental variable d itself. No subset of attempts can be protected by a property that holds for some attempts and not others, because nothing about any attempt is recorded.

## Do any claims survive

From that campaign: **none that can be established.** A claim needs records; there are none.

## Where the defect sits: stock model, wrapper, instrumentation or harness

**Unattributable** from records. Both stated reasons sit at the **policy-execution layer** (loaded normalization; sampler step count), not the simulator or journal. What can be said is negative:

| Layer of the *current* route | Shared with the invalid campaign? |
|---|---|
| pi0.5 implementation | **No.** Current route: OpenPI JAX `215abfb`, checkpoint `gs://openpi-assets/checkpoints/pi05_libero`. Invalid campaign: LeRobot PyTorch |
| Normalization | **No.** Current: OpenPI `Unnormalize`, quantiles, loaded stats equal `norm_stats.json` `b3a44bb2...` exactly (probe_r0, 5 Nov, `rtx_recovery_20261105/`, outside this package; same author, so not independent) |
| Step-count instrument | **No.** Current: a compiled-path count that passed a single-process coded prototype gate (G01) on synthetic inputs, one host. It is not yet a coded v3 tool (`03_COUNTING_METHOD.md`) |
| Client and evaluator | **Unknown.** Whether the campaign used the shared LeRobot evaluator and journal that SmolVLA used is not recorded |

## Blast radius

1. **The 160 attempts:** all excluded, permanently (Alba's ruling; unchanged).
2. **The current OpenPI JAX route:** **not inherited.** Different implementation, different normalization path, verified by this session's diagnostics (author-run; reviewed by one independent read-only reviewer). Its own validity rests on this package, not on the old campaign.
3. **SmolVLA (1,120 attempts):** **cannot be cleared or implicated from these records.** If the pi0.5 campaign shared the LeRobot evaluator and journal, a harness-level defect could reach SmolVLA. Alba's label stands: ARITHMETICALLY REPRODUCED, INSTRUMENT VALIDITY PENDING. Nothing here changes it.
4. **Identity overlap with confirmation:** cannot be certified (R8), because the campaign's seeds and states are unknown. The confirmation cohorts use generated states and seeds 810001/810002/820001/820002, so overlap requires the old campaign to have drawn the same keyed seeds. That is unlikely and still **unverified**.

## What would close it

Alba's raw journals, the LeRobot revision and run configuration. With those, a source-and-config audit, with no outcome analysis, could attribute the defect and settle (3) and (4).
