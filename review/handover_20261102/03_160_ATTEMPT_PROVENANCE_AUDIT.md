# 160-attempt provenance audit, 2 November 2026

## Finding

On the available evidence, the two 160s refer to distinct campaigns.

- **Manuscript 139/160 and 122/160:** SmolVLA attempts, with 160 per configuration, inside a study of 1,120 unique attempts.
- **Invalid 160:** described in Alba's email as a LeRobot PyTorch pi0.5 campaign total. No record of it is available here.

**Label for the SmolVLA figures: ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING ALBA CONFIRMATION.**

Dataset identity and arithmetic are established against the journals. Experimental validity is not. The reason the pi0.5 campaign was invalid is unknown, and a defect in shared LeRobot instrumentation might also affect SmolVLA. This audit makes no decision to keep or remove the figures. That decision is Alba's.

## Provenance table

| Campaign | Model | Run ID | Protocol | Attempts | Artifact | Full SHA-256 | Validation status | Record overlap | Instrumentation independently validated? |
|---|---|---|---|---:|---|---|---|---|---|
| Confirmation, scheduled phase | SmolVLA `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | `confirmation` | `protocol.json`, `confirmation_plan.json` | 799 | `confirmation/episodes.jsonl` in archive | `03a6a6ed5080ac9fb757d1a049531fe757f6070ae6f9a4a2606e3c654d40e0a5` | Arithmetic reproduced | None within canonical set | No |
| Confirmation, primary completion | Same | `primary_deadline` | Above plus deadline amendment | 92 | `primary_deadline/episodes.jsonl` in archive | `6f9c1bee5e7757c95960d33059d7bde47f2d0b4bb759b8ce75a005be3edba4e8` | Arithmetic reproduced | 2 records repeated by the next phase | No |
| Continuation, partial | Same | `confirmation_continuation_partial` | Above plus dedup amendment | 6, of which 2 excluded repeats | `confirmation_continuation_partial/episodes.jsonl` in archive | `3f7f30663f42611f024a65d352d0fd75d676c5fbee4b583529991b1e222c8860` | Arithmetic reproduced | 2 repeats, identical outcome and action stream | No |
| Continuation, rest | Same | `confirmation_continuation_rest` | Above | 225 | `confirmation_continuation_rest/episodes.jsonl` in archive | `71a6367b632cd950aeb12f98f0b5edc267c1dabf2ad338da788013caedeb825d` | Arithmetic reproduced | None | No |
| Confirmation, canonical | Same | `refinement_refresh_final_001` | Above | 1,120 unique of 1,122 raw | `confirmation_canonical_episodes.jsonl` in archive | `ad1bc9ba7fd32d22455703b6fe2eeca2ab95f115e0eb03923598a04e9a44d916` | ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING ALBA CONFIRMATION | 0 identities with discovery or pilot | No |
| Confirmation archive | Same | Contains all of the above | | | `experiments/runs/refinement_refresh_final_001/confirmation_completed.tar.gz` | `36c1d5f2dcaef85664ec7a1bfe7409c849b3be17831566e783d24a00ea4e8df0` | Hash checked on `main` | | |
| Discovery | Same | `refinement_refresh_final_001/discovery` | `protocol.json`, `discovery_plan.json` | 480 | `experiments/runs/refinement_refresh_final_001/discovery/episodes.jsonl` | `7ebdec030428652533a0d68e92a4c99d864f037ee0802e14bdc2735ce01697c1` | Selected counts reproduced | 0 identities with confirmation | No |
| Pilot | SmolVLA | `paired_pilot_001` | Frozen pilot schedule | 80 | `experiments/runs/paired_pilot_001/episodes.jsonl` | `1d95b51af219e876e3eb79f610f54a1092a240ce6dee9211b87f9c1f20c7189c` | Not re-audited | All 20 pilot identities reused by discovery, as separate attempts; 0 with confirmation | No |
| **Invalid pi0.5 campaign** | pi0.5, per Alba's email | **Not available** | **Not available** | 160, per Alba's email | **Not available** | **Not available** | INVALID; excluded; never relabelled | **Not testable** | Mechanism unknown |

Every "Same" model row has `model.safetensors` `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`, RTX 4060 Ti, LeRobot `b6ec0060779550c0a157ae34feb89e0cf86012a8`.

## Recount

`recount_confirmation.py` uses the standard library, reads the archive directly and does not call the repository's analyzer.

| Check | Result |
|---|---|
| Raw and unique | 1,122 and 1,120 |
| Repeats | 2 at `libero_object`, task 7, state 29, seed 3001, at (2,2) and (2,5); outcome and action stream identical |
| (2,2) | 139/160; 21 task-horizon failures; mean 149.0 steps |
| (2,5) | 122/160; 38 task-horizon failures; mean 166.3 steps |
| Paired difference | +10.625 points |
| Paired wins and losses | 24 and 7 |
| Bootstrap, 20,000 draws, own code and seed | [+4.375, +16.875] |
| Actions | 171,805; primary 50,451 |
| Matching initial observation, primary pairs | 160 of 160 |
| Other settings | (1,2) 137; (1,5) 136; (1,10) 135; (5,2) 138; (5,5) 133 |

## Model identity

- All 1,122 raw records carry a `runtime_manifest_sha256` equal to the hash of their phase manifest.
- All four manifests record `model.safetensors` `71d9563c...` and chunk size 50. The pi0.5 chunk size is 10.
- A separate 1 November download of the pinned SmolVLA revision has the same hash (`experiments/runs/cross_policy_confirmation_v1/preflight_cpu_evidence/smolvla.sha256`, line 3).
- `git log --all -S pi05` finds that string only in the November preparation commits.

**Limits of this evidence.** The manifests are the evaluator's own statement, so the hash shows which weights it recorded. The tar's internal times are Alba's host's file times. GitHub independently witnesses the archive only from the 13 September push.

## What is not established

- **Instrument validity.** The recount shares its source with the journals. Agreement checks arithmetic and binding, not whether the journals are a true record of correct policy behaviour.
- **Upstream parity.** `experiments/runs/development_parity_001/parity.json` records status `passed` for one native episode (Spatial task 0, state 0, seed 1000, h=1, d=10) against the pinned upstream evaluator. That is one episode at native settings, on LeRobot code itself. It does not rule out a shared LeRobot defect and does not cover h=2.
- **Attempt-level comparison with the invalid campaign.** Impossible without its records.

## What was searched for the invalid campaign

All repository refs, MacBook filenames, agent transcripts and Warren's Gmail: nothing found. The RTX filesystem was not searched directly, because that access is blocked. Alba's experiment host (`/mnt/NVME1/projectx-robopad-proposal2/results/`, named in `review/REVISION_AUDIT.md`) is not reachable from here.

## What Alba must confirm

1. The identity and location of the invalid pi0.5 campaign, and that it contains no SmolVLA attempts.
2. The reason it was ruled invalid.
3. Whether that defect could affect `run_paired.py`, the pinned LeRobot code or the SmolVLA runs.

If it could, the figures should not be relied on until that has been checked. No manuscript number was changed, and no invalid attempt was relabelled or reused.
