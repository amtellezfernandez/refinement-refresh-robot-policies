"""Summarize raw LeRobot eval ledgers for the real SmolVLA pilot."""

from __future__ import annotations

import json
from pathlib import Path
from result_metrics import summarize


def main() -> None:
    root = Path(__file__).parent / "results"
    rows = []
    paths = [p for p in root.glob("eval_h*.json") if p.stem[6:].isdigit()]
    for path in sorted(paths, key=lambda p: int(p.stem[6:])):
        data = json.loads(path.read_text())
        rows.append(
            {
                "n_action_steps": int(path.stem[6:]),
                **summarize(data),
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
