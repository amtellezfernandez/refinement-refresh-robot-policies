"""Summarize all completed broad-sweep refresh conditions."""

from __future__ import annotations

import json
from pathlib import Path
from result_metrics import summarize


def main() -> None:
    root = Path(__file__).parent / "results" / "broad_sweep"
    rows = []
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        rows.append({"condition": path.stem, **summarize(data)})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
