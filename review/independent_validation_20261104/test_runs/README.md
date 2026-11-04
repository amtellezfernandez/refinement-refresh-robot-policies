# Test-run ledger (all runs kept, synthetic fixtures only)

| File | Harness | Result | Note |
|---|---|---|---|
| `run001_harness_test_results_FAILED_3of36.json` | pre-freeze | 33/36 | a renumbered missing step with every hash rewritten was not detected in 3/5; led to the episode-level stream-hash check and to reporting consistent forgeries as a measured limit |
| `run002_harness_test_results.json` | pre-freeze | 33/36 | test-construction defect: after the stream-hash check, three "consistent" forgeries no longer rewrote that hash |
| `run003_harness_test_results.json` | frozen (`c92193f`) | 36/36 | |
| `run004_harness_test_results.json` | amendment 01 | 40/40 | adds 4 gate-refusal tests |
| `stage_logic_run001_FAILED_stub_defect.json` | amendment 01 | 3/4 | the "unreachable" stub still put the archived outcome in the ensemble |
| `stage_logic_results.json` | amendment 01 | 4/4 | CHAOTIC_INCONCLUSIVE branch, probe stubbed |
| `protocol_fire_drill.json` | amendment 01 | 5/5 | drill C requires exit 5 |

Audit controls (`../audit/audit_mutation_results.json`): 11/11. The first control run failed 1/10
(`audit_mutation_run001_FAILED_1of10.json`). The dropped record was one of the two repeats, so my expected set was wrong. That run also exposed a crash on a
missing record, which is fixed.

Redaction: in runs 001 to 003 the field `environment.assets_path` held an absolute local path. It was
replaced by `<REPLAY_ENV>/libero-assets-0b3ea86` before publication. No other field was changed.
From run 004 the harness records only the directory name.

History: the original freeze commit existed only locally. Before the first push, history was rebuilt
into a freeze commit (`c92193f`) and a review commit. This dropped the redacted path from history, along with two working files that were never
meant for the repository: an unsent email draft and an internal decision summary. `PROTOCOL.md` is byte-identical
across the rebuild (sha256 `ede3eb85…ea737`).
