# Protocol v3 revision 2 (6 Nov 2026), with Alba's decisions D1–D8: DRAFT / NOT RELEASED / UNSENT

Documentation and design only. No experiment ran. Alba's eight decisions of 6 November are recorded in `12` and implemented across `00`–`09` and `13`. Branch `warren/pi05-protocol-v3-rev2-review-20261106`, based on the published `ba14f26`, for review as a draft PR.

| File | Deliverable |
|---|---|
| `00_COVER_NOTE_FOR_ALBA.md` | 9: cover note |
| `01_PREFLIGHT_V3_DRAFT_REV2.md` | 1: revised v3 draft (Protocol A, engineering preflight) |
| `02_SCIENTIFIC_CONFIRMATION.md` | 2: Protocol B, separate scientific confirmation |
| `03_V2_TO_V3_DESIGN_DIFF.md` | 3: design diff |
| `04_ATTEMPT_ACCOUNTING.md` | 4 and Part 10: accounting and schedule |
| `05_PARITY_AND_ACTION_UNITS.md` | 5, Part 4 (R9) and Part 5 (R4) |
| `06_DIAGNOSTIC_POPULATION.md` | 6, Part 8 |
| `07_HOST_GAPS_REGISTER.md` | 7, Part 3 (R6) |
| `08_EARLIER_CAMPAIGN_REVISED.md` | 8, Part 9 |
| `09_CNT_G02_REVIEW.md` | Part 7 |
| `10_CONTRADICTIONS_FOUND.md` | contradictions between existing documents |
| `11_DOCUMENT_REVIEW.md` | independent document-level review and dispositions (of the first published revision, before D1–D8) |
| `12_DECISION_REGISTER_20261106.md` | Alba's eight decisions of 6 November, where each is implemented, and the superseded requirements removed |
| `13_EVALUATOR_VALIDATION.md` | Protocol E: model-free evaluator validation for both stacks (D3) |

**Disclosed lint findings** (`prereg_lint.py`, after D1–D8):
- `01`: GATE_POINTER × 5: four for tools not written, one (ATT-START) for code that exists only on the implementation draft branch and has never run.
- `02`: GATE_POINTER × 3 for code on the implementation draft branch (never run), and GATE_UNQUALIFIED for RESET, which has never run on the real simulator.
- `13`: GATE_POINTER × 1 for code on the implementation draft branch (never run).
- All three verdict tables (`02`, `13`, and `01`'s verdict list) are total.
