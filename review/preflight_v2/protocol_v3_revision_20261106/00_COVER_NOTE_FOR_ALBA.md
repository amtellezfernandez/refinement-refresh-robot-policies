# Cover note for Alba: protocol v3, revision 2 (DRAFT, NOT RELEASED, UNSENT)

Alba,

This revision implements your 6 November decisions on R6, R9, R4, N1, CNT and the scientific design. It is documentation only:
- no preflight, parity, confirmation or state generation was run;
- no reserved identity was touched;
- no frozen file was changed (v2 27/27 and diagnostic package 126/126 checksums verified on this branch).

## What changed

- **Two protocols.**
  - **A** is an engineering preflight (`01`): synthetic fixed actions, fresh envs, five diagnostic inputs, no outcome values.
  - **B** is the scientific confirmation (`02`): separately scoped, not authorised.
- **Estimand.** Delta_m = success(d=2, h=2) − success(d=5, h=2) per model. 400 model-native identities × 2 = 800 per model, 1,600 total (`04`).
  - R52, R55, NATIVE, h=5 outcomes and the five-arm 2,000-per-model design are removed (`03`).
- **R9.** A stock OpenPI `Policy.infer` reference in a separate process, with no wrapper. DO1–DO3 at d=10. Bitwise equality is required. h=5 is used only for the chunk-use check. Full action table, with commanded ±0.05 m / ±0.5 rad kept distinct from measured motion (`05`).
- **R4.** `KeyedPolicy` is a candidate. Six approval items are listed; none is complete (`05` P5).
- **R6.** Per-model environment register. GPU EGL and WSL llvmpipe/OSMesa are distinguished, and 15 host gaps are listed (`07`).
- **N1.** Separate state populations, generated in each model's own stack. Pairs only within a model (`02` B3).
- **CNT.** G02 exists only as a prototype (`pdr_g2_gate.py`). The production gate and entry point are absent from the repository and the MacBook; the RTX workspace was not searched. The requirements are listed (`09`).

## Decisions I need from you

1. **Primary interval (PROPOSED):** task-clustered paired interval (40 clusters, t₃₉), with a coverage study before release. Accept it, or name another method.
2. **Missingness thresholds (PROPOSED):** UNINTERPRETABLE if more than 20/400 pairs are incomplete, or if missing attempts differ by condition by more than 10.
3. **Episode-level evaluator parity:** v2's 4 × 10 × 10 native-parity episodes are removed from the preflight because they read success rates. Do you still want an evaluator-level check as a separate protocol?
4. **Hosts:** which host and renderer each model's Protocol B runs on (G1). Your reference stack's lock files (G2).
5. **Per-request count attestation:** adopt it or not (`09`).
6. **Numerical prediction per model:** optional. It is currently NONE_RECORDED.
7. **SmolVLA preflight scope:** Protocol A tests SmolVLA only for count, provenance, isolation, fresh-env determinism, the reset check and a source read of its success signal. Normalization, units and evaluator parity are not tested, as in v2. Is that sufficient to release SmolVLA into Protocol B?
8. **Deterministic XLA flags:** Protocol B must run in exactly the configuration Protocol A certified. If you decline the flags after the timing check, R9's cross-process bitwise criterion is not expected to hold under default flags (about 5e-3 drift). How should parity be judged then?

## Not claimed

- No parity result.
- No renderer equivalence.
- No cross-model physical-state equivalence.
- No defect or root cause in the earlier pi0.5 campaign, only an unverified instrument and missing records (`08`).
- No SmolVLA instrument validity.

Warren
