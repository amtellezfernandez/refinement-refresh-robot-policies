"""Model-free LIBERO replay core.

No policy, no LeRobot import, no repository experiment code.  The reset path is
re-implemented from the pinned LeRobot wrapper source
(lerobot@b6ec0060, src/lerobot/envs/libero.py, sha256 97c984f1...d9d71), which the
archive records as the wrapper used for every confirmation attempt:

    env = OffScreenRenderEnv(bddl, camera_heights=H, camera_widths=W,
                             control_freq=20, hard_reset=True); env.reset()   # creation
    per attempt:  seed python/numpy/torch RNGs with environment_seed       # run_paired.py
                  env.seed(environment_seed); env.reset()                  # hard reset
                  env.set_init_state(init_states[initial_state_id])
                  10 x env.step([0,0,0,0,0,0,-1])                           # settle
                  robot.controller.use_delta = True                         # relative mode
    per tick:     env.step(action); success = env.check_success()

Hidden state that the archived 110/79-value `initial_state` does NOT contain, and
which this reset path therefore has to regenerate rather than inject:
  * Panda gripper `current_action` (integrates 0.01 per step from zero at hard reset)
  * OSC `goal_ori` (only updated when a rotation delta is non-zero)
  * robosuite step counter / horizon bookkeeping
"""
from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np

SETTLE_STEPS = 10
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
CONTROL_FREQ = 20
WRAPPER_HORIZON = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300,
                   "libero_10": 520, "libero_90": 400}


def array_sha(a) -> str:
    """Archive hash rule: sha256(json([dtype, shape]) + C-order bytes)."""
    a = np.ascontiguousarray(np.asarray(a))
    header = json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def array_from_record(rec):
    return np.asarray(rec["values"], dtype=np.dtype(rec["dtype"])).reshape(rec["shape"])


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:  # torch only seeds the policy side in the original; physics never reads it
        pass


def task_bddl(suite: str, task_id: int):
    from libero.libero import benchmark, get_libero_path
    b = benchmark.get_benchmark_dict()[suite]()
    task = b.get_task(task_id)
    path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return b, task, path


def official_init_states(suite: str, task_id: int):
    """Same loader semantics as the pinned wrapper (torch.load of the .pruned_init file)."""
    import torch
    from libero.libero import get_libero_path
    b, task, _ = task_bddl(suite, task_id)
    p = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return np.asarray(torch.load(p, weights_only=False)), p


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Replayer:
    """One environment per (suite, task); hard reset per attempt, as in the original."""

    def __init__(self, suite: str, task_id: int, render: bool = True, image_size: int = 360):
        from libero.libero.envs import OffScreenRenderEnv
        self.suite, self.task_id = suite, task_id
        _, self.task, self.bddl = task_bddl(suite, task_id)
        kwargs = dict(bddl_file_name=self.bddl, camera_heights=image_size, camera_widths=image_size,
                      control_freq=CONTROL_FREQ, hard_reset=True)
        if not render:
            # Physics-only variant. Equivalence to the rendering variant is a tested claim
            # (test_harness.py::render_equivalence), not an assumption.
            kwargs.update(has_offscreen_renderer=False, use_camera_obs=False)
        self.render = render
        self.env = OffScreenRenderEnv(**kwargs)
        self.env.reset()
        self.horizon = WRAPPER_HORIZON[suite]

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def reset(self, pre_settle_state: np.ndarray, environment_seed: int):
        env = self.env
        seed_everything(environment_seed)
        env.seed(environment_seed)
        env.reset()
        env.set_init_state(pre_settle_state)
        for _ in range(SETTLE_STEPS):
            env.step(DUMMY_ACTION)
        for robot in env.robots:
            robot.controller.use_delta = True
        return env.sim.get_state().flatten()

    def inject(self, state: np.ndarray):
        """Overwrite physics state only (hidden controller/gripper state is kept)."""
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

    def predicates(self):
        inner = self.env.env
        out = []
        for goal in inner.parsed_problem["goal_state"]:
            try:
                out.append({"predicate": list(goal), "value": bool(inner._eval_predicate(goal))})
            except Exception as exc:  # report, never guess
                out.append({"predicate": list(goal), "value": None, "error": repr(exc)})
        return out

    def step(self, action: np.ndarray):
        _, reward, done, _ = self.env.step(action)
        return bool(self.env.check_success()), float(reward), bool(done)

    def state_sha(self):
        return array_sha(self.env.sim.get_state().flatten())
