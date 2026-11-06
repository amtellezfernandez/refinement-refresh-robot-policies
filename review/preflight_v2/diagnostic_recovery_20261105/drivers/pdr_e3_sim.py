#!/usr/bin/env python3
"""PDR Phase E, iteration 3: which carried state breaks reused-env repeatability. CPU, NO POLICY. NON-COUNTING.

Reused env after one dirtying rollout; before each test rollout apply ONE added reset on top of set_init_state:
  none | warm (zero mjData.qacc_warmstart) | goal (controller.reset_goal) | warm+goal | warm+goal+ts (timestep=0)
and compare the final state with a fresh-env rollout. usage: pdr_e3_sim.py CODE_DIR OUT_DIR S0_NPY
"""
import json, os, sys
import numpy as np

C, OUTD, S0P = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import client_openpi as co  # noqa: E402
from libero.libero import benchmark  # noqa: E402

task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
S0 = np.load(S0P)
rng = np.random.default_rng(20261105)
ACTS = rng.uniform(-1, 1, size=(20, 7))
ACTS[:, 6] = np.sign(ACTS[:, 6])


def fresh():
    env, _ = co.make_env(task)
    env.reset()
    return env


def roll(e, extra=()):
    e.set_init_state(S0)
    if "warm" in extra:
        e.env.sim.data.qacc_warmstart[:] = 0
    if "goal" in extra:
        e.env.robots[0].controller.reset_goal()
    if "ts" in extra:
        e.env.timestep = 0
    for _ in range(co.WAIT_STEPS):
        e.step(co.DUMMY_ACTION)
    for a in ACTS:
        e.step(a.tolist())
    return np.asarray(e.get_sim_state()).copy()


ef = fresh()
ref = roll(ef)
ef.close()
ef = fresh()
R = {"fresh_vs_fresh": float(np.abs(roll(ef) - ref).max())}
ef.close()
for extra in ((), ("warm",), ("goal",), ("warm", "goal"), ("warm", "goal", "ts")):
    e = fresh()
    roll(e)  # dirty it
    R["+".join(extra) or "none"] = [float(np.abs(roll(e, extra) - ref).max()) for _ in range(2)]
    e.close()
json.dump(R, open(os.path.join(OUTD, "result_e3.json"), "w"), indent=1)
print("STATUS completed", json.dumps(R), flush=True)
os._exit(0)
