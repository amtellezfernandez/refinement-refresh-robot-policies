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

## Four questions your decisions leave open

1. **π0.5 non-finite actions.** D7 makes a non-finite SmolVLA action under valid inputs a scientific failure. For π0.5 the written rule (v1 `stop_rule_pi05`) stops the phase instead. I have kept that as written, so the two models classify the same event differently. Do you want π0.5 to follow D7 too?
2. **R3 coverage check.** The grid, generative model, replicate count and seed in `02` B6.4 are my drafting choices (PROPOSED). Please confirm or replace them before the check is run. I expect that some low-discordance cells may fail, because tasks with no discordant pair contribute zero variance; the check is meant to detect exactly that.
3. **Secondary interval reference distribution.** Student t at 160 degrees of freedom (PROPOSED, `02` B6.3).
4. **Goal-reached fixtures for Protocol E.** Neither stack has one. Is a version-matched replay of a LIBERO demonstration acceptable as the source (`13` E3)?

## Not claimed

- No parity, evaluator or count result.
- No renderer or simulator equivalence between the two models.
- No cross-model physical-state equivalence.
- No defect or root cause in the earlier pi0.5 campaign, only an unverified instrument and missing records (`08`).
- No SmolVLA instrument validity.
- No qualification of any kind: the implementation existing is not evidence that it passes.

Warren
