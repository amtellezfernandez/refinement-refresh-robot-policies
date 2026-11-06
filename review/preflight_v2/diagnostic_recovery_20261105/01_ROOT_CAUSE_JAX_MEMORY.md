# Phase A: why the frozen eager pi0.5 count runs out of memory

**Diagnostic A01** (RTX 5090, exclusive reservation R018, `gpu.lock`, synthetic `make_libero_example()` input, no simulator, no run001). The frozen `server_openpi.py` (`c3eca9a4...`) was imported unchanged. All instrumentation is runtime wrapping in `drivers/pdr_a_memory.py` (`7470c08d...`). Raw output: `runs/A01/`.

## Answer

**Live-buffer retention by a JAX cache, which causes real over-demand.** *(Corrected after independent review. First written as "allocation retention, not physical exhaustion". That was wrong: the retained bytes are live buffers. Projected from P7, without release, d=5 needs about 36 GiB and d=10 about 59 GiB, more than the 31.84 GiB card. The exhaustion is physical. Its cause is avoidable retention, and the intrinsic per-call demand is small.)* In eager mode (`jax.disable_jit()`), every forward pass through a scanned Flax block (SigLIP, Gemma) leaves a per-layer copy of that block's weights alive in a **JAX-internal cache**. `gc.collect()` does not free it; **`jax.clear_caches()` frees all of it**. With one call to `jax.clear_caches()` after each `ToNNX` module call, the frozen counter completes at d=2, 5 and 10 and counts **2, 5 and 10**, and its d=2 output is bitwise identical to the run without the release.

## One variable at a time (A01, one process)

| Step | Change | JAX live GiB | New since load | Reading |
|---|---|---|---|---|
| load | | 6.499 | | model 6.246 GiB of parameters |
| S1 | frozen eager count, d=2 | 22.716 | **+15.75** (2,163 arrays) | reproduces P7 (+15.7 GiB after d=2); count = 2 |
| S2 | `gc.collect()` | 22.716 | +15.75 | gc releases nothing |
| S3 | `jax.clear_caches()` + gc | **6.499** | **+0.000** | **all retained memory released** |
| S4 | release hook ON: `clear_caches` after each `ToNNX` call | 6.751 / 6.499 / 6.499 after d=2/5/10 | +0.25 / 0 / 0 | **counts 2 / 5 / 10**; d=10 completed (P1, P2, P5, P6, P7 never did) |
| S5 | hook OFF again (restore), d=2 | 22.598 | **+15.75** | **retention returns**: the release is causal |

## What is retained

- **99.99995 % of the retained bytes** are arrays whose shape equals a parameter leaf with its leading (scan-layer) axis removed. 96 % also match dtype. These are **per-layer weight slices**, not activations or KV cache. That agrees with P7's finding that the increment does not scale with sequence length.
- The counts fit the architecture: each Gemma slice shape appears **54 = 18 layers x 3 LLM passes** (prefix plus two denoising steps at d=2). Each SigLIP slice shape appears **81 = 27 layers x 3 images**.
- Magnitudes fit too: about 2.46 GiB per image-encoding stage (3 images of SigLIP weights) and about 4.6 GiB per LLM pass (Gemma plus action expert weights, bf16). Those are P7's per-stage increments.
- **Referrers:** the three largest retained arrays (`(2, 2048, 16384)` bf16) are each held by a `list`, which is held by a `dict`.

## Owner

**Established:** the holder is cleared by `jax.clear_caches()` and by nothing else tried. It is therefore a JAX-level trace, lowering or compilation cache, not a Python reference in the driver or wrapper. Parameter identities are unchanged after the call (`S1_param_ids_unchanged: true`), so OpenPI did not replace the model's weights.

**Strongest supported explanation (inferred, not proven):** under `disable_jit`, `nn.scan` slices the stacked parameters layer by layer in Python. `nn.remat` (`jax.checkpoint`, which still traces under `disable_jit`) then captures those slices as constants of a traced jaxpr, which JAX memoises. Each forward pass makes new slice objects, so each pass adds new cache entries.

**Not identified:** the specific cache object. The referrer chain stops at anonymous `list`/`dict` containers.

## Physical memory versus allocation

`nvidia-smi` reported 9,104 MiB after load and 25,402 MiB from S1 to the end, **including after S3 had freed 15.75 GiB of live arrays**. The BFC allocator keeps its pool, so process-level GPU memory reflects the allocator's high-water mark, not live demand. The failures in P1, P2, P5, P6 and P7 were the allocator reaching its cap **because of retained cache entries**. The cap values (23.88 GiB at the default fraction; about 31.4 GiB used of the 31.84 GiB card at fraction 1.0) come from P7 and P5/P6. **A01 did not record `bytes_limit` and did not reproduce an OOM.** It ran d=5 and d=10 only with the release. That retention grows linearly with d without release is projected from P7 and the per-pass counts, not re-measured here. True demand per call, measured with the release hook, stayed within 0.25 GiB of the post-load level between calls. The within-call peak under the hook was **not measured separately**: `peak_bytes_in_use` is cumulative, and S1 had already set it. It is therefore **at most 22.80 GiB** (the cumulative peak).

## Consequences

1. The frozen eager count is **not intrinsically infeasible on this card**. It fails because of a JAX cache artefact of eager execution.
2. The release hook makes it run, but each call is slow (about 52 / 83 / 138 s at d = 2 / 5 / 10), because every module call is re-traced.
3. Eager output is not bitwise equal to compiled output (X01: 0.0086-0.023). Whether that exceeds run-to-run noise is **not established**: compiled outputs from two default-flag processes also differ, by about 0.005-0.007 in the first action (D01). The firm point is narrower: **the served program is the compiled one**, so counts are taken there (`03_COUNTING_METHOD.md`).
4. Any use of the hook in a counted run changes the measurement step relative to the frozen protocol. It needs v3.

## Residual uncertainty

- The exact cache object is unnamed.
- One host, JAX 0.5.3, Flax as pinned by OpenPI `215abfb`. Other JAX versions may cache differently.
- One synthetic input. The retention mechanism is input-independent by construction (weights, not activations), but only this input was observed.
