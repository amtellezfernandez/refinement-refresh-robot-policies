# Attempt accounting and schedule (Protocol B, not authorised)

## T1. Population and attempts per model

| Factor | Levels | Count |
|---|---|---|
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | 4 |
| Tasks per suite | 0–9 | 10 |
| State slots per task (model-native) | 0–4 | 5 |
| Policy seeds | 810001, 810002 | 2 |
| **Identities per model** | 4 × 10 × 5 × 2 | **400** |
| Conditions per identity | R22 (d=2, h=2), R25 (d=5, h=2) | **2** |
| **Attempts per model** | 400 × 2 | **800** |
| Models | SmolVLA, pi0.5 | 2 |
| **Attempts total** | 800 × 2 | **1,600** |

## T2. Breakdown (identical for each model)

| Unit | Identities | R22 attempts | R25 attempts | Attempts |
|---|---|---|---|---|
| per task | 10 | 10 | 10 | 20 |
| per suite | 100 | 100 | 100 | 200 |
| per model | 400 | 400 | 400 | **800** |
| both models | 800 (two disjoint model-native populations) | 800 | 800 | **1,600** |

**Condition order:**
- per task: 5 identities R22-first and 5 R25-first;
- per suite: 50 and 50;
- per model: 200 and 200 (`02` B8).

**Retries are not attempts.** An INFRA_PRE retry (at most one per attempt) is a second record of the same planned attempt. The planned count stays 800 per model. Retries are reported separately.

**Excluded from these counts** (never counted as attempts):
- preflight checks;
- the diagnostic population (2 synthetic counter cases, 3 frozen observations, mechanics fixtures);
- the Protocol E evaluator-validation fixtures (12 per stack, model-free);
- the R3 coverage simulation.

## T3. Schedule (per model; models are run and reported separately)

| Step | Content | Gate |
|---|---|---|
| 1 | Generate the model-native state manifest: 200 slots (40 tasks × 5), in that model's stack | Alba's go-ahead for state generation (not given) |
| 2 | Bind state hashes; ISO collision check against the diagnostic population and the other model's manifest | ISO |
| 3 | Generate the 800-row schedule: keyed identity permutation, balanced condition order, pairs adjacent; hash it | generator written in the implementation draft PR, never run |
| 4 | Protocol A PREFLIGHT_PASS, Protocol E EVALUATOR_QUALIFIED and every qualification item PASSED for that model, on the planned host (D4) | PRE |
| 5 | R3 coverage check run on synthetic data; result recorded (it decides only whether the secondary interval is reported) | before release |
| 6 | Alba's release bound to the protocol sha256 and the schedule sha256 | G0-B |
| 7 | Execute 400 pairs in schedule order, each pair consecutive, fresh env per attempt | RESET per attempt |
| 8 | All 800 attempts finally classified; only then read outcomes and run the frozen analysis | B9 |

## T4. Compute and timing: kept separate by model, framework, host and renderer

**No end-to-end runtime has been measured for any Protocol B condition.** Earlier campaign extrapolations of 33–144 h were extrapolations, and they are not used. The figures below are the only measurements on record, each in its own frame. None is combined into a campaign estimate here.

| Model | Framework | Host | Renderer | Measured quantity | Value | Frame and limits |
|---|---|---|---|---|---|---|
| SmolVLA | PyTorch 2.7.1+cu128 | RTX 5090, WSL2 | n/a (generation only) | chunk generation, median host time (p95), d = 2 / 5 / 10 | 126 (160) / 249 (352) / 425 (551) ms | 1 Nov counted run (verdict PREFLIGHT_FAILED; `review/GPU_PREFLIGHT_RESULT.md` §3). 30 calls per d after warm-up, one fixed input, **host not quiet** |
| pi0.5 | JAX 0.5.3 (compiled) | RTX 5090, WSL2 | n/a | same quantity | 59 (61) / 66 (70) / 77 (89) ms | same run; **default XLA flags**. The executed step count was **not verified** at the time. The deterministic-flag cost is **unmeasured** |
| SmolVLA | MuJoCo 3.8.1 | RTX 5090, WSL2 | OSMesa | simulation per step at 360 px | about 156 ms | same run, same caveats |
| pi0.5 client | MuJoCo 3.2.3 | RTX 5090, WSL2 | OSMesa | simulation per step at 256 px | about 100 ms | same run, same caveats |
| pi0.5 | JAX 0.5.3 (compiled, traced `num_steps`) | RTX 5090, WSL2 | n/a | first call, including compile / later calls | 4.7 s / about 0.06 s | B01; synthetic input; not a profile |
| pi0.5 client | MuJoCo 3.2.3 | RTX 5090, WSL2, CPU | OSMesa / llvmpipe | fresh env build | about 0.8 s | `alba-v2-probe-5`; settle only |
| both | n/a | Alba's 4060 Ti GPU-EGL host | NVIDIA EGL | anything | not applicable | **excluded by D4**; no measurement needed |

**Arithmetic from authored constants (upper bounds, not measurements), pi0.5:**
- Horizons of 220 / 280 / 300 / 520 steps, plus 10 wait steps, give at most 200 × (230 + 290 + 310 + 530) = **272,000 simulator steps per model**.
- At h=2, generations per attempt ≤ ceil(horizon / 2). Per model that is ≤ 200 × (110 + 140 + 150 + 260) = **132,000 chunk generations**.
- Network function evaluations are ≤ 100 × 660 × (2 + 5) = **462,000**.
- Episodes end at first success, so the real totals are lower by an unknown amount, which depends on outcomes.

**SmolVLA:** horizons UNKNOWN (G10), so no bound is computed.

**Owed before release:**
- CNT-T: deterministic-flag timing, 100 + 100 calls per d, with and without the flags.
- Per-rollout wall time on a diagnostic fixture with synthetic actions. This measures simulator plus render cost only, per host and renderer.
- A campaign estimate built from those, labelled as an estimate.
