# Protocol v3 revision 2 (6 Nov 2026): DRAFT / NOT RELEASED / UNSENT

Documentation and design only. No experiment ran. Branch `warren/pi05-protocol-v3-rev2-review-20261106`, based on the published `ba14f26`, for review as a draft PR.

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
| `11_DOCUMENT_REVIEW.md` | independent document-level review and dispositions |

**Disclosed lint findings** (`prereg_lint.py`):
- `01`: GATE_POINTER × 4, for tools not written.
- `02`: GATE_POINTER × 2, for tools not written, and GATE_UNQUALIFIED for RESET, which has never run on the real simulator.
- The verdict table is total.
