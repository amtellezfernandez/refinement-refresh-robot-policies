# Earlier pi0.5 campaign: revised interpretation

This revision supersedes the interpretive wording of `../diagnostic_recovery_20261105/02_EARLIER_CAMPAIGN_BLAST_RADIUS.md` and of `START_HERE.md` item 6. Both files are kept unchanged as the record. Nothing about the campaign's outcomes was read, re-analysed or reconstructed.

## Established

1. **The instrument was not verified.** Two properties could not be established for the 160-attempt LeRobot PyTorch pi0.5 campaign (Alba, PR #1 and her 2 Nov message as relayed):
   - that the loaded normalization was the official one;
   - that the requested number of denoising steps actually executed.
2. **Required records are missing.** The raw attempt records, the run configuration and the LeRobot revision used are not in the repository or on any branch. They are also not on the RTX workspace (`find`, 5 Nov). Locations outside our machines are UNSEARCHED.
3. **Consequence:** the campaign is permanently excluded (Alba's ruling). No claim from it can be established, because a claim needs records.

## Not established

| Item | Why not |
|---|---|
| **An actual underlying defect** | No record shows that either property was wrong. "Could not be verified" is not "was wrong". There may have been no defect |
| **A root cause** | There is no observed defect to attribute, and no records to attribute one from |
| **Reach through shared LeRobot or evaluator components** | Whether the campaign used the LeRobot evaluator and journal that SmolVLA used is not recorded. Even if it did, no defect is established that could propagate |
| **SmolVLA outcome instrumentation** | Neither cleared nor implicated by this campaign. SmolVLA stays **ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING** (Alba). That label rests on its own pending validation, not on this campaign |

## Wording corrections to the 5 to 6 Nov package

| Location | Was | Corrected reading |
|---|---|---|
| `START_HERE.md:40` | "No records exist, so the defect cannot be attributed." | "No records exist. Neither a defect nor its absence can be established." |
| `02_…:30` heading | "Where the defect sits: stock model, wrapper, instrumentation or harness" | "Which layers of the current route share components with the unverified campaign" |
| `02_…:45` | "a harness-level defect could reach SmolVLA" | "if a defect existed in a shared harness layer, it could have reached SmolVLA. No such defect is established" |
| `02_…:50` | "could attribute the defect" | "could establish whether either unverified property was in fact wrong, and if so where" |
| `02_…` blast-radius item 2 | "verified by this session's diagnostics" | "examined by this session's diagnostics (author-run, AI-reviewed). The current route's validity rests on Protocol A, which has not run" |
| `02_…` table, step-count instrument | "a single-process coded prototype gate (G01)" | stale: the current prototype is G02. Neither is a coded v3 tool |

The "shares no component" statements for the pi0.5 implementation and normalization stand: OpenPI JAX against LeRobot PyTorch. They describe the current route. They say nothing about whether the old campaign had a defect.
