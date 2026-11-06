#!/usr/bin/env python3
"""PDR Phase E: simulator validity, CPU only, NO POLICY. NON-COUNTING DIAGNOSTIC.

Diagnostic state: libero_spatial task 0, env.seed(7) as in the frozen client, then env.reset() placement sample
(NOT an official init state, NOT a generated confirmation slot; confirmation schedules are PENDING_STATE_GENERATION).
Actions are synthetic (fixed pseudo-random or unit probes). No episode, no success signal is read.
usage: pdr_e_sim.py CODE_DIR OUT_DIR TAG          (MUJOCO_GL / PYOPENGL_PLATFORM set by caller)
"""
import hashlib, json, os, sys
import numpy as np

C, OUTD, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import client_openpi as co  # frozen client: make_env, policy_input, quat2axisangle  # noqa: E402
from libero.libero import benchmark  # noqa: E402
import mujoco  # noqa: E402

h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()  # noqa: E731
R = {"tag": TAG, "MUJOCO_GL": os.environ.get("MUJOCO_GL"), "mujoco": mujoco.__version__}
task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)


def fresh():
    env, prompt = co.make_env(task)
    env.reset()
    return env, prompt


env, prompt = fresh()
S0 = np.asarray(env.get_sim_state()).copy()
R["S0_sha"], R["S0_len"] = h(S0), int(S0.size)
np.save(os.path.join(OUTD, "S0_%s.npy" % TAG), S0)


def settle(e, s):
    obs = e.set_init_state(s)
    for _ in range(co.WAIT_STEPS):
        obs, _, _, _ = e.step(co.DUMMY_ACTION)
    return obs


# E1 render + policy input from the settled diagnostic state
obs = settle(env, S0)
pin = co.policy_input(obs, prompt)
np.save(os.path.join(OUTD, "agentview_%s.npy" % TAG), obs["agentview_image"])
np.save(os.path.join(OUTD, "wrist_%s.npy" % TAG), obs["robot0_eye_in_hand_image"])
np.save(os.path.join(OUTD, "pin_image_%s.npy" % TAG), pin["observation/image"])
np.save(os.path.join(OUTD, "pin_state_%s.npy" % TAG), pin["observation/state"])
R["E1"] = {"agentview_sha": h(obs["agentview_image"]), "wrist_sha": h(obs["robot0_eye_in_hand_image"]),
           "pin_image_sha": h(pin["observation/image"]), "pin_state": pin["observation/state"].tolist(),
           "settled_sim_sha": h(env.get_sim_state())}

# E2 state restoration: same env x5, fresh env x2
same = [h(np.asarray(settle(env, S0)["agentview_image"])) for _ in range(5)]
sims = []
for _ in range(5):
    settle(env, S0)
    sims.append(np.asarray(env.get_sim_state()).copy())
env2, _ = fresh()
settle(env2, S0)
fresh_sim = np.asarray(env2.get_sim_state()).copy()
R["E2"] = {"same_env_agentview_distinct": len(set(same)),
           "same_env_settled_max_abs": float(max(np.abs(s - sims[0]).max() for s in sims)),
           "fresh_vs_same_settled_max_abs": float(np.abs(fresh_sim - sims[0]).max()),
           "restored_exact_before_settle": bool(np.array_equal(np.asarray((env.set_init_state(S0), env.get_sim_state())[1]), S0))}

# E3 physics determinism under a fixed synthetic action sequence
rng = np.random.default_rng(20261105)
ACTS = rng.uniform(-1, 1, size=(20, 7))
ACTS[:, 6] = np.sign(ACTS[:, 6])


def roll(e):
    settle(e, S0)
    for a in ACTS:
        e.step(a.tolist())
    return np.asarray(e.get_sim_state()).copy()


f = [roll(env) for _ in range(3)] + [roll(env2)]
R["E3"] = {"max_abs_vs_first": [float(np.abs(x - f[0]).max()) for x in f], "final_sha": [h(x) for x in f]}
np.save(os.path.join(OUTD, "E3_final_%s.npy" % TAG), f[0])

# E4 controller units: config + empirical eef displacement per unit action
ctrl = env.env.robots[0].controller
R["E4_controller"] = {k: (np.asarray(getattr(ctrl, k)).tolist() if hasattr(ctrl, k) else None)
                      for k in ("input_max", "input_min", "output_max", "output_min", "control_delta", "kp", "control_freq")}
R["E4_controller"]["type"] = type(ctrl).__name__
R["E4_control_freq_env"] = getattr(env.env, "control_freq", None)
probe = {}
for name, a in {"zero": [0, 0, 0, 0, 0, 0, -1], "+x0.5": [0.5, 0, 0, 0, 0, 0, -1], "-x0.5": [-0.5, 0, 0, 0, 0, 0, -1],
                "+x1.0": [1, 0, 0, 0, 0, 0, -1], "+z0.5": [0, 0, 0.5, 0, 0, 0, -1], "grip+1": [0, 0, 0, 0, 0, 0, 1]}.items():
    o = settle(env, S0)
    p0, g0 = np.asarray(o["robot0_eef_pos"]).copy(), np.asarray(o["robot0_gripper_qpos"]).copy()
    for _ in range(5):
        o, _, _, _ = env.step(a)
    probe[name] = {"eef_delta_5steps": (np.asarray(o["robot0_eef_pos"]) - p0).tolist(),
                   "gripper_qpos_delta": (np.asarray(o["robot0_gripper_qpos"]) - g0).tolist()}
R["E4_probe"] = probe
json.dump(R, open(os.path.join(OUTD, "result_%s.json" % TAG), "w"), indent=1, default=str)
print("STATUS completed", TAG, flush=True)
os._exit(0)  # skip the EGL context destructor (teardown error is a separate, unresolved item from P4)
