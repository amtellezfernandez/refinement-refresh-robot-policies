#!/usr/bin/env python3
"""Smallest useful pi0.5 evaluator check through the official client. Python 3.8. Uses no GPU itself."""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_guard  # noqa: E402
from client_openpi import connect, make_env, rollout  # noqa: E402
from common import atomic_write_json  # noqa: E402

SMOKE_SEED = 790012
CONDITIONS = (("NATIVE", 5, 10), ("R22", 2, 2), ("R25", 2, 5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    launch_guard.check("preflight", processes=lambda: [])  # The server owns the GPU.
    report = {"model": "pi05_client", "gates": {"G4_evaluator": "NOT_REACHED"}, "evaluator": {}, "started_unix": time.time()}
    try:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        state = suite.get_task_init_states(0)[0]
        env, prompt = make_env(task)
        client = connect(args.port)
        for name, h, d in CONDITIONS:
            row = {"h": h, "d": d, "episode_identity": {"suite": "libero_spatial", "task_id": 0,
                                                        "state_slot": "official0", "policy_seed": SMOKE_SEED}}
            result = rollout(row, env, prompt, client, state)
            generation = result["timing"]["generation"]
            report["evaluator"][name] = {"h": h, "d": d, "smoke_outcome_not_evidence": result["outcome"],
                                         "action_shape": list(result["actions"].shape), "wall_s": result["timing"]["rollout_wall_s"],
                                         "generations": len(generation),
                                         "round_trip_s": [x["client_round_trip_s"] for x in generation],
                                         "host_generation_s": [x["host_generation_s"] for x in generation],
                                         "memory": result["memory"]}
        env.close()
        report["gates"]["G4_evaluator"] = "PASS"
        report["status"] = "completed"
    except BaseException as error:  # A failed run is the record; never replaced.
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["resources"] = {"max_rss_gib": usage.ru_maxrss / 2**20, "cpu_user_s": usage.ru_utime, "cpu_sys_s": usage.ru_stime}
        report["finished_unix"] = time.time()
        atomic_write_json(args.output / "pi05_client_small.json", report)


if __name__ == "__main__":
    main()
