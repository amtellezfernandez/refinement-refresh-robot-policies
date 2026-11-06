# Reproducibility package

## Reproductions of the critical diagnostics (identical driver, fresh process)

| Original | Reproduction | Input | Result |
|---|---|---|---|
| A01 (memory root cause) | A02 | unseeded synthetic | **Reproduced.** Retained after d=2: 15.7476 GiB both times; gc releases nothing; `clear_caches` returns to the post-load level; hook counts 2/5/10; d=2 hook output bitwise equal to no-hook; retention returns on restore. Post-load level 6.419 vs 6.499 GiB (different random input; not investigated) |
| B01 (compiled counting, M1/M2, stock vs wrapper) | B02 | unseeded synthetic (a different random input) | **All verdicts reproduced:** every I1 count incl. M1 10/10/10 and M2 3/6/11; stock = wrapper bitwise; cb = plain bitwise at all d; all 12 I2 identifications; all within-process repeats bitwise. I2 margins 7.6 / 6.4 (B01: 5.8 / 4.4) |
| F01 (falsification M3/M5/M6) | F02 | **seeded**, same input | **All verdicts reproduced** (I1 counts, every bitwise relation, P = plain bitwise). **Numbers not reproduced:** distances and neighbour gaps differ in the third decimal |

## The cross-process discrepancy (D01)

| Arm | Processes | Same seeded input, compiled d = 2 / 5 / 10 |
|---|---|---|
| default XLA flags | 2 | **all three hashes differ**; first-action max difference 0.0049 / 0.0052 / 0.0068 (normalized controller units) |
| `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"` | 2 | **bitwise identical** at all d |

**Reading.** With default flags, compiled pi0.5 output is bitwise reproducible within a process but not across processes. This is consistent with per-process GPU autotuning or non-deterministic kernel choice. Both flags were changed together, so which one is responsible is not separated. n=2 per arm. The speed cost of the deterministic flags is **unmeasured**.

**Consequences.**
1. The counting gate is unaffected: all its comparisons are within one process.
2. Any cross-run claim of identical chunks (replay, audit, a paired identity run twice) needs the deterministic flags, or must accept about 5e-3 drift.
3. The v2 profile's fixed-noise hashes were neither seeded nor deterministic; they are not reproducible.

**v3 proposal:** set the two flags for the pi0.5 server, after a timing check (Alba's R5 asks for timing).

## How to re-run any diagnostic

Each GPU diagnostic ran as a job on the RTX under an exclusive GPU reservation. The exact job body for each run is `runs/<ID>/job_command.sh`. `drivers/mkjob.sh RUN_ID DRIVER RESERVATION TIMEOUT` regenerates it from a driver. Drivers too long for the Windows SSH command limit (about 8 KB) were first copied to the host in hash-checked chunks, and the job body then copies the staged file and checks its hash. The result is base64+gzip between `BEGIN_RESULT` and `END_RESULT` in `runs/<ID>/log`.

The job takes `gpu.lock`, runs a RAM watchdog, records `env.txt`, the frozen-code hashes and the OpenPI revision, and refuses a driver whose SHA-256 differs from the one embedded in the command. The simulator drivers (E01-E03) are CPU-only, use the `envs/client` interpreter, and have no GPU lock. Their exact commands are preserved in `runs/E0*/log`.

## Identities

| Item | SHA-256 |
|---|---|
| `drivers/pdr_a_memory.py` | `7470c08d35fe7ebc5911e2887c02cc62debede52e8e23d720f5faaf1def64738` |
| `drivers/pdr_b_compiled.py` | `15e28be446bc3f7915fc1989da99690faea3ce67276bb4203bee98dfc001279f` |
| `drivers/pdr_x_eager_vs_compiled.py` | `42c78e2566516df78347931906009fae790bd55b17ad89ef141e624ce04fc9fc` |
| `drivers/pdr_f_falsify.py` | see `SHA256SUMS` and the `driver_sha256` line in `runs/F01/log` |
| `drivers/pdr_d_xproc.py` | see `SHA256SUMS` and `runs/D01/log` |
| `drivers/pdr_e_sim.py` / `pdr_e2_sim.py` / `pdr_e3_sim.py` | `e0d3a1ff...` / `16ac4342...` / `c1793e2d...` |
| Frozen `server_openpi.py` (local = RTX `code-078e166`) | `c3eca9a49f35935f0bd49adfe72b2d5c9db5702e62f1eab9ab28e49e7e1832d9` |
| OpenPI on the RTX | `215abfb217dbac7d5f1273282331b9b1866c0479`, 0 modified files |
| `norm_stats.json` | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
