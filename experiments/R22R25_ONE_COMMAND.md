# One command after qualification and release

The frontend lives here because R22/R25 is a scientific recipe. BenchStudio
provides `bs.ProcessCommand` for the reusable process/log/service mechanisms.
Use a reviewed BenchStudio version that includes that API in the frontend's
Python >=3.10 environment; keep each native environment pinned and separate.

Place existing files in `WORKSPACE/smolvla/` and `WORKSPACE/pi05/`:
`manifest.json`, `schedule.json`, `initial_states.npz`, and pinned `checkpoint/`.
Keep these outside frozen evidence. An existing `r3_record.json` is forwarded
to that model's analysis; without one, the existing analysis withholds the
secondary interval. No R3 check or identity generation is performed here.

Preview (read-only, no subprocesses):

```sh
python experiments/run_r22r25.py --workspace /path/to/workspace --smolvla-python /path/to/smolvla/bin/python --pi05-python /path/to/openpi/bin/python
```

After both native qualifications and the separate campaign release exist, the
same command with `--execute` orchestrates the released round:

```sh
python experiments/run_r22r25.py --workspace /path/to/workspace --smolvla-python /path/to/smolvla/bin/python --pi05-python /path/to/openpi/bin/python --execute
```

The native verification stage checks existing qualification and release records,
manifest/schedule pairing and state archive hashes for BOTH models before any
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
