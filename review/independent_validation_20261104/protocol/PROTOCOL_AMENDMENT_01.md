# Amendment 01 to the frozen replay protocol (pre-approval)

`PROTOCOL.md` is unchanged (sha256 `ede3eb8570487b7e2eda8e47f39c6afc1d12e40bece590e8c4590d34341ea737`,
frozen at commit `c92193f`). This file records one harness change made after the freeze and
before any approval or archived replay.

## Change

`replay/replay.py`, sha256 `6463a4d5…` → `6c5a75cee80df8718196ad36a60ed4df7de6fb47c5e91ffa228732f371a420d9`.

1. A missing, unreadable, malformed or non-object release file, or an unreadable protocol file,
   now gives a clean refusal (exit 5) instead of an exception (exit 1). The approval condition is
   unchanged. On any read failure the release is treated as empty, so it can never satisfy the
   protocol-hash equality.
2. The run summary records the asset directory name instead of its absolute local path.

The lines cited in PROTOCOL.md section 6 still hold the same statements: `replay.py:206`
(`if refusal:`), `run_protocol.py:176` and `run_protocol.py:185`. `run_protocol.py` and
`libero_replay.py` are byte-identical to the freeze.

No criterion, tolerance, sample, stage, verdict rule, prediction or host requirement changed.

## Tests added

- Harness suite, run 004: **40/40**. Four new gate refusals: missing release file, malformed
  JSON, non-object JSON, and missing protocol with a hash-less release.
- `replay/test_stage_logic.py`: **4/4**. The CHAOTIC_INCONCLUSIVE branch is tested with the
  physics probe replaced by a stub, so this covers classification and the Stage 2 refusal, not the probe.
  The real probe path is covered by drill E. The first run failed on a defect in the stub, which is kept as
  `test_runs/stage_logic_run001_FAILED_stub_defect.json`.
- Protocol drills: **5/5**. Drill C now requires exit 5.

## Binding

An approval should name both files: `{"protocol_sha256": "ede3eb85…ea737",
"amendments": ["<sha256 of this file>"], "approved_by": …, "decision": "APPROVED"}`.
The code checks `protocol_sha256` only. The binding of this amendment is therefore recorded, not enforced.
