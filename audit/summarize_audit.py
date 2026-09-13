#!/usr/bin/env python3
"""Validate and summarize the frozen pilot-audit table."""

from __future__ import annotations

import csv
from pathlib import Path


RESOURCE_COLUMNS = (
    "real_acquisition",
    "human_labor",
    "simulation",
    "wm_generation",
    "training_compute",
    "deployment_compute",
    "wall_time",
    "persistent_storage",
)
VALID = {"Q", "P", "U", "NA"}


rows = list(csv.DictReader(Path(__file__).with_name("pilot_audit.csv").open()))
if len(rows) != 16:
    raise SystemExit(f"expected 16 pilot records, found {len(rows)}")
for row in rows:
    invalid = {row[column] for column in RESOURCE_COLUMNS} - VALID
    if invalid:
        raise SystemExit(f"{row['system']}: invalid status {sorted(invalid)}")
    computed = "yes" if all(row[c] in {"Q", "NA"} for c in RESOURCE_COLUMNS) else "no"
    if row["fully_reconstructable"] != computed:
        raise SystemExit(f"{row['system']}: full-checklist flag contradicts resource statuses")

print(f"systems={len(rows)}")
print(f"full_checklist_complete={sum(all(row[c] in {'Q', 'NA'} for c in RESOURCE_COLUMNS) for row in rows)}")
for column in RESOURCE_COLUMNS:
    print(f"{column}_quantitative={sum(row[column] == 'Q' for row in rows)}")
    print(f"{column}_applicable={sum(row[column] != 'NA' for row in rows)}")

# Boundary sensitivity only: these calculations do not revalidate source evidence.
for label, columns in (
    ("without_storage", tuple(c for c in RESOURCE_COLUMNS if c != "persistent_storage")),
    ("without_storage_human_wall", tuple(c for c in RESOURCE_COLUMNS if c not in
                                         {"persistent_storage", "human_labor", "wall_time"})),
    ("training_and_deployment", ("training_compute", "deployment_compute")),
):
    complete = [row["system"] for row in rows if all(row[c] in {"Q", "NA"} for c in columns)]
    print(f"{label}_complete={len(complete)} ({', '.join(complete)})")
