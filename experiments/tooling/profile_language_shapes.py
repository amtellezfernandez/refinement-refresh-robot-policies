#!/usr/bin/env python3
"""Profile dense arithmetic across audited language lengths after rollouts finish."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--shape-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from run_paired import load_policy_and_processors, make_noise, seed_environment

    seed_environment(0)
    policy, _, _, _ = load_policy_and_processors(args.checkpoint)
    stored = torch.load(args.inputs, map_location="cuda", weights_only=False)
    original = stored["prepared_observations"]["native"]
    shape_audit = json.loads(args.shape_audit.read_text())
    lengths = sorted({r["input_ids_shape"][1] for r in shape_audit["rows"]})
    token_key = "observation.language.tokens"
    mask_key = "observation.language.attention_mask"
    if original[token_key].shape[1] != max(lengths):
        raise RuntimeError("The stored calibration input is not the longest audited shape")

    identity = {"suite": "libero_spatial", "task_id": 0, "initial_state_id": 0, "policy_seed": 1000}
    noise = make_noise(policy, identity, 0)
    rows = []
    policy.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda") if policy.config.use_amp else torch.inference_mode():
        for length in lengths:
            batch = copy.deepcopy(original)
            batch[token_key] = batch[token_key][:, :length]
            batch[mask_key] = batch[mask_key][:, :length]
            for d in (2, 5, 10):
                policy.config.num_steps = d
                with FlopCounterMode(display=False) as counter:
                    action = policy.predict_action_chunk(copy.deepcopy(batch), noise=noise.clone())
                torch.cuda.synchronize()
                if action.shape[1] != 50 or not torch.isfinite(action).all().item():
                    raise RuntimeError("Invalid action chunk during shape profile")
                rows.append({"language_tokens": length, "d": d,
                             "generation_flops": int(counter.get_total_flops())})

    native = next(r["generation_flops"] for r in rows if r["language_tokens"] == 20 and r["d"] == 10)
    original_half_ceiling = native / 2
    for row in rows:
        for h in (1, 2, 5):
            row[f"h{h}_rate_flops"] = row["generation_flops"] / h
            row[f"h{h}_eligible_half_native"] = row[f"h{h}_rate_flops"] <= original_half_ceiling
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Dense-arithmetic shape transport only; no rollout outcomes read",
        "original_native_generation_flops": native,
        "original_half_native_rate_ceiling_flops": original_half_ceiling,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
