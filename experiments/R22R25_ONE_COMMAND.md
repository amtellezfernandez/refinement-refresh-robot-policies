# One command after qualification and release

The frontend lives here because R22/R25 is a scientific recipe. BenchStudio
provides `ProcessCommand` from `benchstudio.processes` for reusable process,
log and service mechanisms. This frontend is tested against BenchStudio commit
`081e12a8d0e92fbfd40a92147fcc23eab415c19f` in its
Python >=3.10 environment; keep each native environment pinned and separate.

The pi05 LIBERO client and OpenPI/JAX server are distinct environments. OpenPI
at the pinned commit 215abfb217dbac7d5f1273282331b9b1866c0479 declares Python
>=3.11 in its [pyproject.toml](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/pyproject.toml).
Its server cannot be launched with the prepared Python 3.8 LIBERO client.
Select both explicitly; `--pi05-python` remains an alias for CLIENT only.
The server's stdlib precheck runs before the first SmolVLA attempt and rejects
Python <3.11 before reading artifacts or importing a native framework. Each
role logs its actual interpreter/version. Minimum-version compatibility is not
proof that package locks, checkpoints, CUDA/JAX or native qualification match;
those still require pinned environments and retained native evidence.

The workspace defaults to this checkout's
`experiments/runs/cross_policy_confirmation_v2/`, as required by the existing
release gate. Omit `--workspace` in normal use; a custom path must resolve to
that same location. An arbitrary external workspace would be rejected even
with a valid release. This directory is outside the frozen evidence packages.

Place existing files in `WORKSPACE/smolvla/` and `WORKSPACE/pi05/`:
`manifest.json`, `schedule.json`, `initial_states.npz`, and pinned `checkpoint/`.
Keep these outside frozen evidence. An existing `r3_record.json` is forwarded
to that model's analysis; without one, the existing analysis withholds the
secondary interval. No R3 check or identity generation is performed here.

Preview (read-only, no subprocesses):

```sh
python experiments/run_r22r25.py --smolvla-python /path/to/smolvla/bin/python --pi05-client-python /path/to/libero-client/bin/python --pi05-server-python /path/to/openpi-server/bin/python
```

After both native qualifications and the separate campaign release exist, the
same command with `--execute` orchestrates the released round:

```sh
python experiments/run_r22r25.py --smolvla-python /path/to/smolvla/bin/python --pi05-client-python /path/to/libero-client/bin/python --pi05-server-python /path/to/openpi-server/bin/python --execute
```

The native verification stage checks existing qualification and release records,
manifest/schedule pairing and state archive hashes for BOTH models. It also checks
that every scheduled state's digest, BDDL, task language and provenance match the
native manifest, including its archive and generator bindings. Server-interpreter
compatibility is checked before any
model process starts. GPU reservation/environment variables required by the
existing native guards must already be present. Then it runs SmolVLA and its
offline analysis, starts the pi05 attesting server with deterministic XLA flags
and OSMesa, waits for its existing startup gate and local port, runs the pi05
client and its separate offline analysis, and shuts down its owned server. Logs and
receipts go to a new `operator-logs/launch-*` folder each invocation, including
failed input checks, without overwriting earlier logs. Neither model is pooled.

Execution stops on any failed command. If a stopped native run retained a summary,
its existing analysis produces the permitted accounting report before stopping;
this does not continue to the other model or retry a native attempt. Existing run/server/analysis outputs are
refused; use the native reviewed recovery path after interruption, not a fresh
workspace or a second campaign. Do not launch concurrent operators or other GPU
jobs. This is not a global single-use campaign registry or cross-host lock; the
existing duplicate-campaign limitation remains. Port readiness is a liveness
probe; the native pi05 client verifies gate metadata and per-request attestation.

**Current status:** orchestration tested only with fake subprocesses, not native
models/simulators. Native Protocol A/E, parity/action mapping, compiled counter
qualification and manifests remain outstanding. This command does not implement
those missing scientific checks, issue their permissions, execute E0, qualify a
stack, generate states, or release a campaign. Before real use, freeze/check the
frontend and native code hashes in the eventual release. Merging code is not a
campaign release. The native hash list includes this frontend and its verifier.

The pi05 client command monitors the owned server process after readiness.
A server exit while the client remains active stops the client, retains the
failure receipt, and prevents reporting normal campaign completion. Any existing
native stopped-run summary still goes through the same accounting-only rules;
no automatic native retry is introduced. This requires the updated
`ProcessCommand.execute(required_service=...)` from `benchstudio.processes`.
The installed recipe checks `configuration_valid` and `execution_authorized`
separately; neither the package import nor this frontend qualifies a native stack.
These code updates apply to a subsequent reviewed campaign release, not the
independently pinned E0 staging package.


