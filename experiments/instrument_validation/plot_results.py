"""Render the executed finite-suite results; never invent or rerun robot data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
LAYERS = ["syntax", "structure", "aggregates", "full_ledger"]
LABELS = ["JSON\nsyntax", "+ Structural\nschema", "+ Aggregate\nchecks", "+ Full ledger\nreconciliation"]


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    categories = summary["by_case"]
    codes = {"not_detected": 0, "detected": 1, "accepted_control": 2, "unknown": 3}
    colors = ["#ebedf0", "#2667a0", "#35936c", "#e5ad3c"]
    matrix, labels = [], []
    for category in categories:
        row, row_labels = [], []
        for layer in LAYERS:
            counts = category["layers"][layer]
            if len(counts) != 1:
                raise ValueError("Mixed per-source outcomes require a proportion plot; do not hide them")
            outcome, count = next(iter(counts.items()))
            if outcome not in codes:
                raise ValueError(f"Unexpected outcome needs explicit plotting: {outcome}")
            row.append(codes[outcome])
            row_labels.append("unknown" if outcome == "unknown" else
                              f"{0 if outcome == 'not_detected' else count}/{category['n']}")
        matrix.append(row)
        labels.append(row_labels)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.2, 8.1))
    cmap = ListedColormap(colors)
    ax.imshow(np.array(matrix), cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4), aspect="auto")
    ax.set_xticks(range(4), LABELS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(categories)), [c["label"] for c in categories])
    ax.tick_params(axis="both", length=0, pad=8)
    for y, row in enumerate(labels):
        for x, label in enumerate(row):
            value = matrix[y][x]
            ax.text(x, y, label, ha="center", va="center", fontsize=9,
                    color="white" if value in (1, 2) else "#343a40")
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(categories), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [Patch(facecolor=c, label=l) for c, l in zip(colors,
              ["Defect not rejected", "Defect rejected", "Control accepted", "Outside observable scope"])]
    fig.legend(handles=legend, ncol=2, loc="lower center", bbox_to_anchor=(.57, .045), frameon=False, fontsize=9)
    fig.suptitle("Which recorded defects can each validation layer reject?", x=.04, y=.98,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(.04, .935, "Executed deterministic mutation suite · 4 archived sources · 68 constructed trials", fontsize=10)
    fig.text(.04, .018,
             "Defect cells: rejected/source count. Control row: accepted/source count. Sources are not independent robot replicates.",
             fontsize=8.4, color="#495057")
    fig.subplots_adjust(left=.405, right=.98, top=.86, bottom=.135)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 180} if suffix == "png" else {}
        fig.savefig(ROOT / f"mutation_detection.{suffix}", **kwargs)
    plt.close(fig)
    print("Rendered mutation_detection.pdf, .svg, and .png from executed summary.json")


if __name__ == "__main__":
    main()
