# Amendment 02 — fail-closed enforcement repairs (pre-approval)

This is an **implementation amendment** to the immutable
`PROTOCOL.md` (sha256 `ede3eb8570487b7e2eda8e47f39c6afc1d12e40bece590e8c4590d34341ea737`).
Amendment 01 is retained verbatim. No archived action replay has been authorized
or performed by this change. The paper, past experiment archives, population,
frozen sample, scientific contrasts, reset tolerance `1e-6`, verdict definitions,
and stage ordering are unchanged.

## Why this amendment is necessary

Code review identified execution paths that did not enforce the written protocol:

1. A release approving only Stage 0 could still launch the default full three-stage
   orchestration because `scope` was not checked.
2. The exact 1,120-bundle population, 320 primary attempts, 80 task-state starts,
   and frozen 57-attempt sample were not validated before work.
3. The release bound only `PROTOCOL.md`, not the actual replay executables, the
   amendment(s), the archive bundle bytes or sample-manifest bytes.
4. A failed Stage 0 with `--through 0`, a failed Stage 1 with `--through 1`,
   or a failed Stage 2 could still yield process exit 0.
5. The promised counted-run identity and release hash were not durably written
   before the first reset.

## Implemented repair

- `replay/governance.py` independently checks the immutable protocol SHA, the
  explicit approver and decision, contiguous approval scope `[0]`, `[0,1]` or
  `[0,1,2]`, and that `--through` lies within that scope.
- It checks the exact frozen bundle file digest
  `99587b7e3ea594572f18ee3b933109eaf57dfa0ce4cc74deea83b3bc1b43f372`,
  1,120 unique identities, the 320 original primary keys and 80 paired starts.
  It reconstructs the frozen outcome-stratified Stage 1 draw, verifies all 29
  nonempty strata, and checks the 57 IDs and selection digest
  `3ffe84b0ef2d055a6c22667406c9d8eca370bf66ab7999c46e99a7114f73df56`.
- The release must also bind the exact sample manifest *file* SHA-256, all
  execution-source file SHA-256 digests, and both protocol amendments.
- The declared counted host remains arm64 macOS with rendering enabled; a
  different platform or runtime needs a new explicit protocol decision.
- `run_protocol.py` rejects malformed/out-of-scope releases before creating an
  output directory, records `counted_run_identity.json` with exclusive create,
  fsync and directory fsync before Stage 0, and checks Stage 0/1/2 outcomes
  against the frozen verdict rules and 139/160 versus 122/160 totals.
- `replay.py` refuses direct CLI execution on genuine archived paper evidence.
  It accepts an archived action replay only from the validated orchestrator
  with a stage-scoped runtime capability. The capability avoids accidental
  execution; it is not cryptographic authorization against a person who edits
  the Python code. Old synthetic fixture drills remain supported.

## Approval format (example only — NOT AN APPROVAL)

Create an external, uncommitted `release.json` **only after explicit approval**.
All SHA-256 fields must be computed from the reviewed branch and frozen inputs.

```json
{
  "approved_by": "Alba María Téllez Fernández",
  "decision": "APPROVED",
  "scope": [0],
  "protocol_sha256": "ede3eb8570487b7e2eda8e47f39c6afc1d12e40bece590e8c4590d34341ea737",
  "bundles_sha256": "99587b7e3ea594572f18ee3b933109eaf57dfa0ce4cc74deea83b3bc1b43f372",
  "sample_selection_sha256": "3ffe84b0ef2d055a6c22667406c9d8eca370bf66ab7999c46e99a7114f73df56",
  "sample_manifest_sha256": "<SHA256 of protocol/stage1_sample_manifest.json>",
  "render": true,
  "amendments_sha256": {
    "PROTOCOL_AMENDMENT_01.md": "<SHA256>",
    "PROTOCOL_AMENDMENT_02.md": "<SHA256>"
  },
  "code_sha256": {
    "governance.py": "<SHA256>",
    "run_protocol.py": "<SHA256>",
    "replay.py": "<SHA256>",
    "libero_replay.py": "<SHA256>",
    "extract_bundles.py": "<SHA256>"
  }
}
```

The field `scope: [0]` authorizes *only the 80-state Stage 0 reset check*.
Stage 1 requires a later release explicitly including stage 1. Stage 2
requires a later release explicitly including stage 2. No release file is
created in this PR and no stage may be run merely because the code exists.

## Evidence boundary

This change is **not** a modification of the 4 November frozen scientific
protocol. It improves implementation of its gates. It must be reviewed and
tested before any scientific approval. Do not relabel archived evidence as a
synthetic fixture, change the frozen protocol, choose another tolerance after
seeing outputs, or report fixture tests as independent confirmation.


## Runtime provenance repair (5 November 2026)

Review of the archived confirmation manifests found a silent ambiguity in
`experiments/tooling/run_paired.py`: the runtime manifest records
`numpy.__version__` and `torch.__version__`, but its package inventory builds
a single-value dictionary from every visible distribution. If multiple
`.dist-info` records are visible, one can overwrite another without warning.
The archive therefore reports imported versions 2.2.6 / 2.7.1 and inventory
versions 1.26.4 / 2.6.0 for NumPy / Torch. The archive does not retain module
paths or all candidate distribution records, so the mismatch cannot be
retroactively resolved from that manifest alone.

The PR now changes the runner to preserve duplicate distribution candidates,
leave ambiguous package versions unset in the legacy summary, and record each
critical imported module's version and file path. It also hashes the resolved
LIBERO assets tree in future experiment manifests. The replay gate checks the
imported NumPy, Torch and MuJoCo module versions and binds them to matching
distribution metadata under the same installation root, rather than trusting
a single metadata lookup. These changes apply to future manifests and replay
preflights; they do not rewrite archived records or change the frozen
scientific protocol. They have not yet been tested after this amendment.

The original LIBERO asset tree was not hashed by the experiment, so its exact
revision remains unknown. The replay's pinned asset tree is verified against
its own manifest. A passing Stage 0 on arm64 must therefore be reported as a
reset diagnostic in the declared replay environment; it cannot establish that
the original experiment used identical assets or prove cross-architecture
trajectory equivalence. This provenance correction does not require GitHub
Actions; the code and its local qualification evidence must still be reviewed.
