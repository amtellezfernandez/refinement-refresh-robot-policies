#!/usr/bin/env python3
"""Read-only checks of the historical SmolVLA journals against the controls a shared defect could break.

    python3 audit_shared_instrumentation.py confirmation_completed.tar.gz

Prepared for use once Alba states why the pi0.5 campaign was invalid. A pass says the journals are
self-consistent on these counters. It does not certify the historical data, because the counters
were written by the instrument under question.
"""
import collections
import json
import math
import sys
import tarfile

PHASES = ["confirmation", "primary_deadline", "confirmation_continuation_partial", "confirmation_continuation_rest"]


def main(path):
    failures = collections.Counter()
    totals = collections.Counter()
    settings, horizons = set(), collections.defaultdict(set)
    with tarfile.open(path) as tar:
        for phase in PHASES:
            manifest = json.loads(tar.extractfile(phase + "/runtime_manifest.json").read())
            config = manifest["effective_policy_config"]
            settings.add(json.dumps({k: config.get(k) for k in ("chunk_size", "max_action_dim", "normalization_mapping")}, sort_keys=True))
            for line in tar.extractfile(phase + "/episodes.jsonl").read().decode().splitlines():
                row = json.loads(line)
                counts, h, d, steps = row["counts"], row["h"], row["denoising_steps"], row["steps"]
                totals["records"] += 1
                horizons[row["episode_identity"]["suite"]].add(row["task_horizon"])
                failures["generations != ceil(steps / h)"] += counts["generations"] != math.ceil(steps / h)
                failures["denoise_steps != generations * d"] += counts["denoise_steps"] != counts["generations"] * d
                failures["action_expert_calls != denoise_steps"] += counts["action_expert_calls"] != counts["denoise_steps"]
                failures["prefill_calls != generations"] += counts["prefill_calls"] != counts["generations"]
                failures["vision_encoder_calls != 2 * generations"] += counts["vision_encoder_calls"] != 2 * counts["generations"]
                failures["steps > task_horizon"] += steps > row["task_horizon"]
                failures["failure before horizon"] += (not row["success"]) and steps != row["task_horizon"]
    print("records", totals["records"])
    for name in sorted(failures):
        print("{:45s} {}".format(name, failures[name]))
    print("horizons", {k: sorted(v) for k, v in horizons.items()})
    print("config", sorted(settings))
    print("Self-consistency only. Not a validity certificate.")


if __name__ == "__main__":
    main(sys.argv[1])
