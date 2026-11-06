#!/usr/bin/env python3
"""PDR Phase E, iteration 2: history dependence of a reused LIBERO env. CPU, NO POLICY. NON-COUNTING DIAGNOSTIC.

Same diagnostic state as E01 (libero_spatial task 0, seed 7, env.reset() placement; S0 loaded from E01's .npy).
  F  fresh env per trial: settle + fixed synthetic actions, x3 -> determinism
  H  reused env: dirty it, set_init_state(S0), record controller goal / timestep vs fresh  -> what carries over
  X  one variable: reused env + controller.reset_goal() after set_init_state -> matches fresh?  (+ restore: without it)
  U  unit probes, fresh env per probe
  P  render repeat on one fixed state without stepping
usage: pdr_e2_sim.py CODE_DIR OUT_DIR S0_NPY
"""
import hashlib, json, os, sys
import numpy as np

C, OUTD, S0P = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import client_openpi as co  # noqa: E402
from libero.libero import benchmark  # noqa: E402

h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()  # noqa: E731
task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
S0 = np.load(S0P)
rng = np.random.default_rng(20261105)
ACTS = rng.uniform(-1, 1, size=(20, 7))
ACTS[:, 6] = np.sign(ACTS[:, 6])
R = {"S0_sha": h(S0)}


def fresh():
    env, _ = co.make_env(task)
    env.reset()
    return env


def settle(e, s, reset_goal=False):
    obs = e.set_init_state(s)
    if reset_goal:
        e.env.robots[0].controller.reset_goal()
    for _ in range(co.WAIT_STEPS):
        obs, _, _, _ = e.step(co.DUMMY_ACTION)
    return obs


def roll(e, reset_goal=False):
    settle(e, S0, reset_goal)
    for a in ACTS:
        e.step(a.tolist())
    return np.asarray(e.get_sim_state()).copy()


def ctrl_state(e):
    c = e.env.robots[0].controller
    out = {"timestep": int(getattr(e.env, "timestep", -1)), "done": bool(getattr(e.env, "done", False)),
           "horizon": getattr(e.env, "horizon", None), "ignore_done": getattr(e.env, "ignore_done", None)}
    for k in ("goal_pos", "goal_ori", "relative_ori", "ori_ref"):
        v = getattr(c, k, None)
        out[k] = None if v is None else np.asarray(v).round(6).tolist()
    return out


# F: fresh-env determinism
F = []
for _ in range(3):
    e = fresh()
    F.append(roll(e))
    e.close()
R["F_max_abs_vs_first"] = [float(np.abs(x - F[0]).max()) for x in F]

# H: what a reused env carries after set_init_state
e = fresh()
e.set_init_state(S0)
R["H_fresh_after_set"] = ctrl_state(e)
roll(e)
e.set_init_state(S0)
R["H_dirty_after_set"] = ctrl_state(e)
R["H_simstate_restored_exact"] = bool(np.array_equal(np.asarray(e.get_sim_state()), S0))

# X: reset_goal (one variable), then restore (without)
X_with = [roll(e, reset_goal=True) for _ in range(3)]
X_without = [roll(e, reset_goal=False) for _ in range(2)]
R["X_reused_resetgoal_vs_fresh"] = [float(np.abs(x - F[0]).max()) for x in X_with]
R["X_reused_noreset_vs_fresh"] = [float(np.abs(x - F[0]).max()) for x in X_without]
e.close()
# does reset_goal change a FRESH env's trajectory? (must be ~0 if it only removes carry-over)
e = fresh()
R["X_fresh_resetgoal_vs_fresh"] = float(np.abs(roll(e, reset_goal=True) - F[0]).max())
e.close()

# U: unit probes, fresh env each
U = {}
for name, a in {"zero": [0, 0, 0, 0, 0, 0, -1], "+x0.5": [0.5, 0, 0, 0, 0, 0, -1], "-x0.5": [-0.5, 0, 0, 0, 0, 0, -1],
                "+x1.0": [1, 0, 0, 0, 0, 0, -1], "+y0.5": [0, 0.5, 0, 0, 0, 0, -1], "+z0.5": [0, 0, 0.5, 0, 0, 0, -1],
                "+rz0.5": [0, 0, 0, 0, 0, 0.5, -1], "grip+1": [0, 0, 0, 0, 0, 0, 1]}.items():
    e = fresh()
    o = settle(e, S0)
    p0, q0, g0 = [np.asarray(o[k]).copy() for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")]
    for _ in range(5):
        o, _, _, _ = e.step(a)
    U[name] = {"eef_delta_5steps_m": (np.asarray(o["robot0_eef_pos"]) - p0).round(5).tolist(),
               "axisangle_delta": (co.quat2axisangle(np.asarray(o["robot0_eef_quat"])) - co.quat2axisangle(q0)).round(5).tolist(),
               "gripper_qpos_delta": (np.asarray(o["robot0_gripper_qpos"]) - g0).round(5).tolist()}
    e.close()
R["U"] = U

# P: render repeat on one fixed state, no stepping in between
e = fresh()
settle(e, S0)
imgs = []
for _ in range(3):
    e.env.sim.forward()
    ob = e.env._get_observations(force_update=True)
    imgs.append(np.asarray(ob["agentview_image"]).astype(np.int16))
R["P_render_repeat_max_abs"] = [int(np.abs(x - imgs[0]).max()) for x in imgs]
R["P_negative_control_agent_vs_wrist_max_abs"] = int(np.abs(imgs[0] - np.asarray(ob["robot0_eye_in_hand_image"]).astype(np.int16)).max())
json.dump(R, open(os.path.join(OUTD, "result_e2.json"), "w"), indent=1, default=str)
print("STATUS completed", flush=True)
os._exit(0)
