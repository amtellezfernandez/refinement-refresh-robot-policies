# Cover note for Alba: protocol v3, revision 2, with your 6 November decisions (DRAFT, NOT RELEASED, UNSENT)

Alba,

This revision implements your decisions on R6, R9, R4, N1, CNT and the scientific design, and now also your eight answers of 6 November to the questions in the first version of this note (D1–D8, `12_DECISION_REGISTER_20261106.md`). It is documentation only:
- no preflight, parity, evaluator validation, confirmation, state generation or coverage check was run;
- no reserved identity was touched;
- no frozen file was changed (v2 27/27 and diagnostic package 126/126 checksums verified on this branch).

A separate draft PR contains the implementation (runner, gates, evaluator harness, schedule generator, analysis, qualification records). None of that code has been run on a simulator, a model or a GPU, and every qualification item in it is UNVERIFIED.

## Your eight decisions and where they are

| # | Decision | Where |
|---|---|---|
| D1 | Primary: exact cohort contrast and four paired counts over the 400 identities. Secondary: 95% interval for fresh states on the same 40 tasks, reported only with complete data and a passed R3 check. No t₃₉ interval | `02` B6.1–B6.4 |
| D2 | UNINTERPRETABLE above 20 incomplete pairs or an arm-missingness difference above 10; bounds over 400; direction only if the bounds exclude zero | `02` B6.5 |
| D3 | Model-free evaluator check for both stacks, independent BDDL inspection, π0.5 `done` semantics | `13` (Protocol E) |
| D4 | RTX 5090 WSL OSMesa for both; bindings before qualification; 4060 Ti EGL excluded | `07`; `01` A3a |
| D5 | Per-request executed-step attestation for π0.5, bound to request and tick; failure stops the run | `09` C4; `02` B5 |
| D6 | NONE_RECORDED; two-sided | `02` B1 |
| D7 | SmolVLA qualification items; non-finite under valid inputs = scientific failure; broken contract = instrument incident | `01` A5 S-QUAL; `02` B5 |
| D8 | Deterministic flags; same-host bitwise parity; matched noise; preprocessing comparison; clipping and frame checks; stop if infeasible | `05` P1–P3, P6 |

The superseded requirements, including the t₃₉ interval, McNemar, the optional attestation and the cross-host plan, are listed in `12` §3 and removed from the current text.

## Decisions on the four implementation questions

1. **π0.5 outcomes.** Apply the same rule as SmolVLA. An attested chunk of the expected floating-point shape that contains a non-finite action is a scientific failure only if valid inputs and a non-finite raw model output are verified for that request. A malformed chunk, failed attestation, or unclear attribution stops the run as an instrument incident before applying the action. Keep PROTOCOL_STOP for unsupported refinement settings.
2. **R3.** Freeze the existing baseline, discordance, and Delta grid. Vary task heterogeneity and within-state seed dependence independently over {0, 0.25, 0.5} at each level. Use 2,000 replicates per feasible cell, seed 20261107, and minimum empirical coverage 0.93. Freeze the grid and code before a separately authorised synthetic check.
3. **Secondary interval.** Use Student t with 160 degrees of freedom as the prespecified candidate. Report it only for complete data if R3 passes. The exact cohort contrast remains primary.
4. **Protocol E fixtures.** Prefer version-matched saved actions. Alternatively use a scripted trajectory frozen as an action file, with its script source and hashes recorded, on each pinned native stack. Use diagnostic states disjoint from confirmation. Validate success and termination through the evaluator with an independent BDDL goal check. A goal teleport alone does not qualify the rollout path.

## Not claimed

- No parity, evaluator or count result.
- No renderer or simulator equivalence between the two models.
- No cross-model physical-state equivalence.
- No defect or root cause in the earlier pi0.5 campaign, only an unverified instrument and missing records (`08`).
- No SmolVLA instrument validity.
- No qualification of any kind: the implementation existing is not evidence that it passes.

Warren
