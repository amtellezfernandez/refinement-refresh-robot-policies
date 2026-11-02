# Immediate handover, 2 November 2026

Branch, commit and attachments are in `01_READ_ME_FIRST.md`, which is in the ZIP only. Paths below are repository paths. In the ZIP, the same files are under `ARTIFACTS/`, as listed in `08_ARTIFACT_MANIFEST.csv`.

## Status vocabulary

| Term | Meaning here |
|---|---|
| PLANNED | Specified and scheduled. No attempt exists. A schedule is not execution evidence |
| PREFLIGHT INCOMPLETE | Required checks have not all been run |
| PREFLIGHT FAILED | A frozen gate failed in the counted run |
| EXECUTED | Attempts ran and records exist |
| VALIDATED | Executed, and the instrument has been independently validated. **Nothing here has this status** |
| INVALID | Ruled invalid; stays excluded |
| NOT STARTED | No attempt has run |

## Status of every experiment

| Experiment | Status | Evidence |
|---|---|---|
| SmolVLA discovery, 480 attempts | EXECUTED; counts reproduced | `experiments/runs/refinement_refresh_final_001/discovery/` |
| SmolVLA confirmation, 1,120 unique attempts | EXECUTED; ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING ALBA CONFIRMATION | `confirmation_completed.tar.gz` on `main` |
| SmolVLA paired pilot, 80 attempts | EXECUTED; separate from main counts | `experiments/runs/paired_pilot_001/` |
| Stale-input diagnostic, 40 attempts | Reported in the working paper; stale arm raw records not available; provenance insufficient | None available |
| `stale_instrumented_004` rerun | UNKNOWN; last recorded as active on 13 September | None available |
| LeRobot PyTorch pi0.5 campaign, 160 attempts | INVALID, per Alba's email | None available |
| Bounded engineering preflight, 1 November | PREFLIGHT FAILED | `experiments/runs/cross_policy_confirmation_v1/preflight_gpu_evidence/` |
| Mandatory preflight, both models | PREFLIGHT INCOMPLETE | See `06_PREFLIGHT_STATUS_AND_CORRECTIONS.md` |
| `cross_policy_confirmation_v1`, 4,000 attempts | PLANNED; NOT STARTED | Schedules only |
| `observation_age_confirmation_v1`, 1,600 attempts | PLANNED; NOT STARTED | Schedules only |
| `asynchronous_confirmation_v1`, 2,400 attempts | PLANNED; NOT STARTED; blocked | Schedules only |
| Third policy | Candidates listed; NOT STARTED | `experiments/runs/third_policy_candidates.json` |

## Where to read next

| Question | Document |
|---|---|
| Are the two 160s the same records? | `03_160_ATTEMPT_PROVENANCE_AUDIT.md` |
| What can go in the camera-ready paper? | `04_CAMERA_READY_EVIDENCE_GATE.md` |
| What is each protocol's state? | `05_PROTOCOL_AND_SCHEDULE_STATUS.md` |
| What did the preflight show, and what was corrected? | `06_PREFLIGHT_STATUS_AND_CORRECTIONS.md` |
| What remains, and how long might it take? | `07_REMAINING_WORK_AND_ESTIMATES.md` |
| How do I check this? | `REPRODUCE.md` |

## What Alba needs to decide today

1. Whether 139/160 and 122/160 can be relied on, given what she knows about the invalid pi0.5 campaign (section "What Alba must confirm" in `03_160_ATTEMPT_PROVENANCE_AUDIT.md`).
2. Which version is the camera-ready base, and whether to keep the stale-input result.

No new pi0.5 or R22/R25 result is available for camera-ready inclusion.
