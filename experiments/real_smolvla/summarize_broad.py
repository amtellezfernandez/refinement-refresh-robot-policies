"""Print the bounded five-task-per-suite sweep summary."""

from __future__ import annotations

import json
from pathlib import Path
from result_metrics import summarize


def main() -> None:
    root = Path(__file__).parent / "results" / "broad_sweep"
    for path in sorted(root.glob("*.json")):
        d = json.loads(path.read_text())
        row = summarize(d)
        print(
            f"{path.stem:12s} success={row['successes']}/{row['episodes']} "
            f"controls={row['control_steps']} neural={row['neural_chunks']} "
            f"ratio={row['control_to_neural']:.2f}x "
            f"rollout_s={row['rollout_seconds']:.2f}"
        )


if __name__ == "__main__":
    main()
