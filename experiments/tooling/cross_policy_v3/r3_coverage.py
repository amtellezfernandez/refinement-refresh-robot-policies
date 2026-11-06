#!/usr/bin/env python3
"""R3 coverage check for the secondary fresh-state interval (02 B6.4; decision D1). Python 3.8+, stdlib only.

CPU simulation on synthetic data. It reads no outcome. Running it is NOT authorised by the implementation
task: `main` refuses without Alba's authorisation file bound to the Protocol B document. The grid, generative
model, replicate count and seed are the PROPOSED values of 02 B6.4.

Generative model (02 B6.4): per pair, a joint outcome in {11, 10, 01, 00} with
    pi10 = (q + delta)/2, pi01 = (q - delta)/2, pi11 = b - pi01, pi00 = 1 - pi11 - pi10 - pi01.
Task level: pi_t ~ Dirichlet(kappa * pi), state level: pi_ts ~ Dirichlet(kappa * pi_t), kappa = (1 - rho)/rho
(no heterogeneity when rho = 0). The two seeds of a state are independent draws from pi_ts.
Target per replicate: (1/T) * sum_t (pi_t10 - pi_t01), the fresh-state difference on that replicate's tasks.
Criterion: empirical coverage >= 0.93 in every feasible cell.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_v2  # noqa: E402
from v3common import PROTOCOL_B_DOCUMENT, atomic_write_json, file_sha, sha  # noqa: E402

SCHEMA = "r3_coverage_record_v1"
GRID = {"baseline": [0.20, 0.50, 0.80, 0.95], "discordance": [0.02, 0.10, 0.30],
        "delta": [-0.15, -0.05, 0.0, 0.05, 0.15], "rho": [0.0, 0.25, 0.5]}
LAYOUT = {"tasks": 40, "states_per_task": 5, "seeds_per_state": 2}
REPLICATES = 2000
SEED = 20261107
CRITERION = 0.93
CATEGORIES = ("11", "10", "01", "00")


def grid_sha256():
    return sha({"grid": GRID, "layout": LAYOUT, "seed": SEED})


def base_probabilities(b, q, delta):
    pi10, pi01 = (q + delta) / 2.0, (q - delta) / 2.0
    pi11 = b - pi01
    pi00 = 1.0 - pi11 - pi10 - pi01
    return {"11": pi11, "10": pi10, "01": pi01, "00": pi00}


def feasible(b, q, delta):
    if q < abs(delta) - 1e-12:
        return False
    return all(-1e-12 <= value <= 1.0 + 1e-12 for value in base_probabilities(b, q, delta).values())


def cells():
    out = []
    for b in GRID["baseline"]:
        for q in GRID["discordance"]:
            for delta in GRID["delta"]:
                for rho in GRID["rho"]:
                    out.append({"baseline": b, "discordance": q, "delta": delta, "rho": rho,
                                "feasible": feasible(b, q, delta)})
    return out


def dirichlet(rng, probabilities, kappa):
    """Dirichlet(kappa * p); zero components stay zero. kappa None means no heterogeneity."""
    if kappa is None:
        return dict(probabilities)
    draws = {k: (rng.gammavariate(kappa * p, 1.0) if p > 1e-15 else 0.0) for k, p in probabilities.items()}
    total = sum(draws.values())
    return {k: v / total for k, v in draws.items()}


def draw_category(rng, probabilities):
    u, cumulative = rng.random(), 0.0
    for key in CATEGORIES:
        cumulative += probabilities[key]
        if u < cumulative:
            return key
    return CATEGORIES[-1]


def replicate(rng, pi, rho):
    kappa = None if rho == 0 else (1.0 - rho) / rho
    task_means, variances, targets = [], [], []
    for _ in range(LAYOUT["tasks"]):
        pi_t = dirichlet(rng, pi, kappa)
        targets.append(pi_t["10"] - pi_t["01"])
        state_values = []
        for _ in range(LAYOUT["states_per_task"]):
            pi_ts = dirichlet(rng, pi_t, kappa)
            total = 0.0
            for _ in range(LAYOUT["seeds_per_state"]):
                category = draw_category(rng, pi_ts)
                total += {"10": 1.0, "01": -1.0}.get(category, 0.0)
            state_values.append(total / LAYOUT["seeds_per_state"])
        mean = sum(state_values) / len(state_values)
        task_means.append(mean)
        variances.append(sum((v - mean) ** 2 for v in state_values) / (len(state_values) - 1))
    interval = analysis_v2.stratified_interval(task_means, variances, LAYOUT["states_per_task"])
    target = sum(targets) / len(targets)
    return interval["lower"] <= target <= interval["upper"]


def run_cell(cell, replicates=REPLICATES):
    seed = int(hashlib.sha256(json.dumps([SEED, cell["baseline"], cell["discordance"], cell["delta"], cell["rho"]])
                              .encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    pi = base_probabilities(cell["baseline"], cell["discordance"], cell["delta"])
    pi = {k: max(0.0, v) for k, v in pi.items()}
    covered = sum(1 for _ in range(replicates) if replicate(rng, pi, cell["rho"]))
    return covered / float(replicates)


def run_check(replicates=REPLICATES):
    results, failing = [], []
    for cell in cells():
        if not cell["feasible"]:
            results.append(dict(cell, coverage=None))
            continue
        coverage = run_cell(cell, replicates)
        results.append(dict(cell, coverage=coverage))
        if coverage < CRITERION:
            failing.append(cell)
    return {"schema": SCHEMA, "status": "PASS" if not failing and replicates == REPLICATES else "FAIL",
            "criterion": CRITERION, "replicates": replicates, "seed": SEED, "grid": GRID, "layout": LAYOUT,
            "grid_sha256": grid_sha256(), "analysis_sha256": file_sha(analysis_v2.__file__),
            "r3_code_sha256": file_sha(__file__), "failing_cells": len(failing), "cells": results}


def main():
    import release
    parser = argparse.ArgumentParser(description="R3 coverage check (synthetic; needs Alba's authorisation).")
    parser.add_argument("--authorisation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release.check_authorisation(args.authorisation, PROTOCOL_B_DOCUMENT,
                                {"r3_coverage.py": __file__, "analysis_v2.py": analysis_v2.__file__},
                                scope="R3_COVERAGE_CHECK")
    if args.output.exists():
        raise SystemExit("REFUSED: output exists; the check's record is never replaced")
    record = run_check()
    atomic_write_json(args.output, record)
    print(json.dumps({"status": record["status"], "failing_cells": record["failing_cells"]}))


if __name__ == "__main__":
    main()
