# Contradictions between existing documents (found while reading; none edited)

| # | Documents | Contradiction | Disposition |
|---|---|---|---|
| X1 | `cross_policy_confirmation_v1.json` `expected_attempts` (2,000 per model, 4,000 total) vs PR #1 and v2 (800 per model, 1,600) | five-arm total vs primary-only total | Alba's binding design: 800 per model, 1,600 total. The frozen v1 file is left unchanged and superseded by Protocol B |
| X2 | Alba's 2 Nov email ("800 attempts across both models") vs PR #1 ("1,600, correcting the 800-across-both wording") | total count | PR #1 governed already; Alba's 6 Nov design confirms 1,600 |
| X3 | confirmation_v1 primary interval (state-slot percentile bootstrap) vs R3 (Warren's synthetic check found undercoverage; Alba: preregister a calibrated method) | interval method | replaced (`02` B6.3–B6.4) |
| X4 | v2 R7 single generated state set (`generate_states.py` keyed by cohort) vs Alba N1 (separate model-native populations) | state population | N1 governs; tooling change listed |
| X5 | 2 Nov N1 probe ("a SmolVLA-stack state restores exactly in the client stack") vs N1 ruling (no cross-version physical-state claim) | not a contradiction in content, but the probe is easily read as equivalence | `02` B3 states that restoration is a serialization fact only |
| X6 | v3 rev 1 §5 (parity reuses one env per task, stock order) vs Alba R9 (action-level parity on frozen observations; fresh envs in preflight) | parity procedure | R9 governs; episode-level parity removed (flagged in `03`) |
| X7 | `rtx_recovery_20261105/REPORT.md:23` / `gpu.txt` driver 610.88 vs WSL `libnvidia-gpucomp.so.610.57.01` | driver version | unresolved; register G9 |
| X8 | v3 rev 1 §2 lists only client MuJoCo 3.2.3 and SmolVLA 3.8.1; `openpi.lock.txt` also installs MuJoCo 2.3.7 in the server env | incomplete pin | register G7 |
| X9 | `02_EARLIER_CAMPAIGN…` lines 30, 45, 50 and `START_HERE.md:40` speak of "the defect" vs the same file's line 16 ("not a demonstrated defect") | presupposed defect | corrected reading in `08` |
| X10 | `02_EARLIER_CAMPAIGN…` cites G01 as the step-count prototype vs `03_COUNTING_METHOD.md` (G02 current) | stale reference | `08` |
| X11 | v3 rev 1 §7 quotes "measured displacement about 0.0088 m per step" beside the commanded ±0.05 m | risk of reading the measured figure as a unit | `05` P3 separates commanded from measured |
| X12 | v2 robosuite pins: SmolVLA stack 1.4.0 vs client stack 1.4.1 | not recorded in any protocol as a cross-stack difference | register H2 |
| X13 | `make_primary_schedule.py` requires the 2,000-row five-condition source; `analyze.py` primary = bootstrap | code encodes the removed design | listed as owed tooling in `03`; code untouched |
| X15 | `attempt_loop.py:7-9` (a failed attempt stops the run; nothing retried) vs Protocol B B5, B7, B9 (continue past missing attempts; one INFRA_PRE retry) | runner semantics | new Protocol B runner required; frozen loop untouched (`02` B5) |
| X16 | frozen `client_openpi.rollout` always computes `success` (line 127) vs Protocol A's rule that no step computes a success value | preflight harness | a success-free mechanics harness is required (`01` A3, CHUNK) |
| X14 | memory record, 4 Nov: "our branches … ARE on origin" vs the 2 Nov lines "unpushed" | status records, not protocol | not relevant to design; no action |

Checked and **not** a contradiction: `make_primary_schedule.py` encodes R22 = (h=2, d=2) and R25 = (h=2, d=5), consistent with confirmation_v1.
