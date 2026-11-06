# R6: per-model native environment and host-specific unresolved-gaps register

**Ruling implemented:**
- Each model is evaluated in its own documented native environment.
- **No renderer-controlled cross-model comparability is claimed.**
- NVIDIA GPU EGL, WSL OSMesa/llvmpipe and any other rendering environment are distinguished.
- Facts not established from a primary record are marked **UNKNOWN**.

**Sources:**
- `preflight_cpu_evidence/*.lock.txt` and `gpu.txt` (1–2 Nov);
- `rtx_recovery_20261105/evidence/p4b`, `p4c` and `rtx_recovery_20261105/REPORT.md` line 23 (5 Nov);
- `diagnostic_recovery_20261105/runs/*/env.txt`.

Every fact below is **carried** from those records and was not re-measured on 6 November.

## H1. Rendering environments: three distinct kinds

| Kind | Where it exists | GL vendor / renderer | Status |
|---|---|---|---|
| **NVIDIA GPU EGL** | Alba's reference host (RTX 4060 Ti, by her account); possibly a future native Linux host | NVIDIA, presumably | **UNKNOWN on every machine we have access to.** Not available on the RTX WSL host: no NVIDIA EGL vendor file, only `50_mesa.json` (P4) |
| **WSL Mesa llvmpipe, requested as `MUJOCO_GL=egl`** | RTX WSL host | `Mesa` / `llvmpipe (LLVM 20.1.2, 256 bits)` (P4b: client 3.2.3 and SmolVLA 3.8.1 stacks) | software rendering; **not GPU EGL** despite the name |
| **WSL OSMesa / llvmpipe** | RTX WSL host (diagnostic runs used `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`) | Mesa software. `libosmesa6_25.1.7-1ubuntu2~24.04.2_amd64.deb`, sha256 `9c23e599...` | the renderer string under OSMesa was **not logged** (E01 review finding) |

P4c rendered one 64×64 frame **byte-identically** under `egl` and `osmesa` in all three stacks (same hash `67d353d6...`). E01 found the same at 256 px. On this host, therefore, "EGL" and "OSMesa" are not two independent renderers. **A 1–2 Nov comparison of OSMesa against Alba's GPU EGL gave a non-zero SmolVLA first-action L2 of 0.011–0.019 on one state**, so GPU EGL and llvmpipe are not shown to be equivalent. That single confounded comparison suggests they may differ.

## H2. Per-model native environment

| Field | SmolVLA | pi0.5 server | pi0.5 client (simulator) |
|---|---|---|---|
| Native simulator | LeRobot `LiberoEnv` over hf-libero | n/a (no simulation; see gap G7) | OpenPI LIBERO submodule `OffScreenRenderEnv` |
| MuJoCo | 3.8.1 | 2.3.7 installed (whether it is used is UNKNOWN) | 3.2.3 |
| robosuite | 1.4.0 | n/a | 1.4.1 |
| LIBERO package | `hf-libero==0.1.4` | n/a | editable `src/openpi/third_party/libero` (OpenPI submodule) |
| Renderer (as run on the RTX host) | llvmpipe (EGL request) / OSMesa | n/a | llvmpipe (EGL request) / OSMesa |
| Framework | torch 2.7.1+cu128, LeRobot editable `src/lerobot` (v2 pins `b6ec0060...`; the lock file records only the path) | JAX / jaxlib 0.5.3, flax 0.10.2, OpenPI `215abfb` | torch 1.11.0+cu113 (present; not used for policy) |
| Python | 3.12 (`envs/smolvla/lib/python3.12`) | 3.11 (patch UNKNOWN) | 3.8.20 |
| numpy | 2.2.6 | 1.26.4 | 1.22.4 |
| gym / gymnasium | gymnasium 1.4.0 | gymnasium 0.29.1 | gym 0.25.2 |
| transformers | 5.5.4 | 4.53.2 | 4.21.1 |
| Asset path | inside the `hf-libero` install: **path UNKNOWN** | `assets/physical-intelligence/libero/norm_stats.json` | `src/openpi/third_party/libero` |
| Asset hashes | **UNKNOWN** (never manifested) | `norm_stats.json` `b3a44bb2...`; checkpoint 16 files (P0) | 585 OpenPI asset files byte-identical in the client tree (N2 probe, 2 Nov); **a single bound manifest hash does not exist yet** |
| Env variables (diagnostic runs) | `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `OMP/MKL/OPENBLAS_NUM_THREADS=3`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `LD_LIBRARY_PATH=<workspace>/syslibs/...` | the same, plus `XLA_PYTHON_CLIENT_PREALLOCATE=false`; proposed `XLA_FLAGS` deterministic pair | `MUJOCO_GL`, `PYOPENGL_PLATFORM`, thread caps |

**Host (all three stacks as run so far):**
- hostname `DESKTOP-J7AGHMV`; Windows with WSL2 kernel `6.6.87.2`;
- distribution **UNKNOWN**: inferred as Ubuntu 24.04 from the `~24.04` libosmesa package name, not read from `/etc/os-release`;
- architecture **UNKNOWN**: inferred as x86_64 from the amd64 package; `uname -m` not recorded in these files;
- RTX 5090, 32,607 MiB, UUID `GPU-ba05e81b-15e9-b15d-44f4-efddf5fa1e5c`;
- driver 610.88 (`nvidia-smi`), CUDA UMD 13.3.

**Cross-stack differences that rule out any cross-model physical-state claim:**
- MuJoCo 3.8.1 versus 3.2.3;
- robosuite 1.4.0 versus 1.4.1;
- LIBERO packaged as `hf-libero` versus the OpenPI submodule;
- different reset procedures: keyed env seed plus LeRobot reset, versus `env.seed(7)` plus 10 no-op waits.

## H3. Unresolved gaps

| # | Gap | Affects | Status | Closes when |
|---|---|---|---|---|
| G1 | Which host and renderer each model's Protocol B runs on (Alba's ruling: each model in its own native environment; the concrete host is not yet named) | both | **UNKNOWN** | Alba names a host per model; this table is filled from that host |
| G2 | Alba's GPU-EGL reference host: OS, kernel, driver, MuJoCo, Python, package locks | reference | **UNKNOWN** (not on our machines) | Alba supplies lock files |
| G3 | Renderer string under `MUJOCO_GL=osmesa` on the RTX host | both | not logged | logged in PROV |
| G4 | SmolVLA LIBERO asset path and hashes (`hf-libero`) | SmolVLA | **UNKNOWN** | manifest produced in PROV |
| G5 | Single bound asset manifest for the client LIBERO tree | pi0.5 | per-file N2 result only | manifest hash in A3 |
| G6 | LeRobot commit actually checked out in `envs/smolvla` (lock shows an editable path) | SmolVLA | v2 pins `b6ec0060...`; checkout state **not re-verified** | `git -C src/lerobot rev-parse HEAD` recorded in PROV |
| G7 | Whether the pi0.5 server process imports or initialises MuJoCo 2.3.7 or a GL context | pi0.5 | **UNKNOWN** | import trace in PROV |
| G8 | Distribution and architecture recorded first-hand | both | inferred | `/etc/os-release`, `uname -m` in PROV |
| G9 | WSL library file named `libnvidia-gpucomp.so.610.57.01`, while `nvidia-smi` reports driver 610.88 | both | **unresolved discrepancy** | record both; explain or accept as a WSL packaging artefact |
| G10 | SmolVLA episode horizons per suite (`_max_episode_steps`) | SmolVLA | **UNKNOWN** | read from the pinned stack (`02` B4) |
| G11 | `done` in the pinned LIBERO is exactly the BDDL goal check | pi0.5 | **UNVERIFIED** | source read (`02` B4) |
| G12 | OSC clipping and delta frame in robosuite 1.4.1 | pi0.5 | **UNVERIFIED** | source read (`05` P3) |
| G13 | Cause of reused-env history dependence | both | bounded, not named (E03) | not required: fresh env per attempt |
| G14 | Global `np.random.seed(7)` in stock `main.py`, absent from the client | pi0.5 | untested whether anything in the episode path draws from it | test, or declare as a deviation from stock |
| G15 | The WSL kernel has not been restarted since the 2 Nov fault; disk 93% (5 Nov) | host | carried | host health check before any run |
