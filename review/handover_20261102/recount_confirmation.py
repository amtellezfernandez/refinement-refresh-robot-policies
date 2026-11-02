#!/usr/bin/env python3
"""Recount the SmolVLA confirmation from the raw journals. Standard library only.

    python3 recount_confirmation.py confirmation_completed.tar.gz

Reads the archive without extracting it and does not use the repository's analyzer.
This checks arithmetic and record binding. It does not validate the instrument that
wrote the journals.
"""
import collections
import hashlib
import json
import random
import statistics
import sys
import tarfile

# Documented precedence: the earliest record of a repeated identity is kept.
PHASES = ["confirmation", "primary_deadline", "confirmation_continuation_partial",
          "confirmation_continuation_rest"]


def main(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    print("archive_sha256", digest.hexdigest())
    canon, repeats, raw = {}, [], 0
    with tarfile.open(path) as tar:
        for phase in PHASES:
            data = tar.extractfile(phase + "/episodes.jsonl").read()
            manifest = json.loads(tar.extractfile(phase + "/runtime_manifest.json").read())
            bound = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
            raw += len(rows)
            print(phase, "records", len(rows), "journal_sha256", hashlib.sha256(data).hexdigest())
            print("  all_bound_to_manifest", all(r["runtime_manifest_sha256"] == bound for r in rows),
                  "all_completed", all(r["status"] == "completed" for r in rows),
                  "model.safetensors", manifest["checkpoint_files_sha256"]["model.safetensors"],
                  "revision", manifest["checkpoint_expected_revision"],
                  "chunk", manifest["effective_policy_config"]["chunk_size"],
                  "device", manifest["device_name"])
            for r in rows:
                i = r["episode_identity"]
                key = (i["suite"], i["task_id"], i["initial_state_id"], i["policy_seed"], r["h"], r["denoising_steps"])
                if key in canon:
                    same = (canon[key]["success"] == r["success"],
                            canon[key]["action_stream_sha256"] == r["action_stream_sha256"])
                    repeats.append((phase, key, same))
                else:
                    canon[key] = r
    print("raw", raw, "unique", len(canon), "repeats", repeats)
    settings = collections.defaultdict(list)
    for key, r in canon.items():
        settings[key[4:]].append(r)
    for setting, rows in sorted(settings.items()):
        print("h,d", setting, "attempts", len(rows), "successes", sum(r["success"] for r in rows),
              "mean_steps", round(statistics.mean(r["steps"] for r in rows), 1),
              "terminations", dict(collections.Counter(r["termination_reason"] for r in rows)))
    a = {k[:4]: r for k, r in canon.items() if k[4:] == (2, 2)}
    b = {k[:4]: r for k, r in canon.items() if k[4:] == (2, 5)}
    print("same_identities", set(a) == set(b),
          "wins", sum(a[k]["success"] and not b[k]["success"] for k in a),
          "losses", sum(b[k]["success"] and not a[k]["success"] for k in a),
          "matching_initial_observation",
          sum(a[k]["initial_observation_sha256"] == b[k]["initial_observation_sha256"] for k in a),
          "primary_actions", sum(r["steps"] for r in a.values()) + sum(r["steps"] for r in b.values()),
          "all_actions", sum(r["steps"] for r in canon.values()))
    tasks = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in a:
        tasks[k[:2]][k[2]].append(int(a[k]["success"]) - int(b[k]["success"]))
    states = {t: [statistics.mean(v) for v in s.values()] for t, s in tasks.items()}
    print("paired_difference_points", statistics.mean(statistics.mean(v) for v in states.values()) * 100)
    rng = random.Random(20261102)  # This script's seed; the paper's bootstrap uses NumPy and its own seed.
    draws = sorted(sum(sum(rng.choices(v, k=len(v))) / len(v) for v in states.values()) / len(states) * 100
                   for _ in range(20000))
    print("bootstrap_95_percent", draws[500], draws[19500],
          "(20000 draws of states within fixed tasks, seeds kept within state)")


if __name__ == "__main__":
    main(sys.argv[1])
