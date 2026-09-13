#!/usr/bin/env python3
"""Audit policy language-input shapes without reading rollout outcomes.

Run this inside the pinned campaign environment.  It reconstructs task text from
LIBERO metadata and applies the tokenizer settings stored with the checkpoint.
It deliberately does not open episodes.jsonl or any success record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TASKS = {
    "calibration": {"libero_spatial": [0]},
    "discovery": {"libero_spatial": [5, 6, 7, 8]},
    "transfer": {"libero_object": [5, 6, 7, 8], "libero_goal": [5, 6, 7, 8]},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero import benchmark
    from transformers import AutoTokenizer

    preprocessor_path = args.checkpoint / "policy_preprocessor.json"
    preprocessor = json.loads(preprocessor_path.read_text())
    tokenizer_config = next(
        step["config"]
        for step in preprocessor["steps"]
        if step["registry_name"] == "tokenizer_processor"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["tokenizer_name"],
        local_files_only=True,
        padding_side=tokenizer_config["padding_side"],
    )

    benchmark_classes = benchmark.get_benchmark_dict()
    rows = []
    for phase, suites in TASKS.items():
        for suite, task_ids in suites.items():
            task_suite = benchmark_classes[suite]()
            for task_id in task_ids:
                task = task_suite.get_task(task_id)
                # SmolVLANewLineProcessor appends one newline before tokenization.
                policy_text = task.language + "\n"
                encoded = tokenizer(
                    [policy_text],
                    max_length=tokenizer_config["max_length"],
                    padding=tokenizer_config["padding"],
                    truncation=tokenizer_config["truncation"],
                    return_tensors="pt",
                )
                rows.append(
                    {
                        "phase": phase,
                        "suite": suite,
                        "task_id": task_id,
                        "task_name": task.name,
                        "task_language": task.language,
                        "policy_text_sha256": hashlib.sha256(policy_text.encode()).hexdigest(),
                        "input_ids_shape": list(encoded["input_ids"].shape),
                        "attention_mask_shape": list(encoded["attention_mask"].shape),
                        "nonpadding_tokens": int(encoded["attention_mask"].sum()),
                    }
                )

    calibration_lengths = [r["input_ids_shape"][1] for r in rows if r["phase"] == "calibration"]
    evaluation_lengths = [r["input_ids_shape"][1] for r in rows if r["phase"] != "calibration"]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "LIBERO metadata and policy tokenizer only; no rollout outcomes read",
        "checkpoint": str(args.checkpoint.resolve()),
        "policy_preprocessor_sha256": sha256(preprocessor_path),
        "tokenizer_settings": tokenizer_config,
        "calibration_token_lengths": sorted(set(calibration_lengths)),
        "evaluation_token_length_range": [min(evaluation_lengths), max(evaluation_lengths)],
        "all_evaluation_lengths_at_most_calibration": max(evaluation_lengths) <= min(calibration_lengths),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
