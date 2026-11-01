import hashlib,json,os
import numpy as np
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
import hashlib
def k(key,d): return int.from_bytes(hashlib.sha256(json.dumps([d,key],sort_keys=True,separators=(",",":")).encode()).digest()[:4],"little")
out=os.path.expanduser("~/jobs/alba-render-3/frames.npz"); fr={}
for size in (360,256):
    cfg=LiberoEnv(task="libero_spatial",task_ids=[0],init_states=True,hard_reset=True,control_mode="relative",observation_height=size,observation_width=size)
    env=make_env(cfg,n_envs=1,use_async_envs=False)["libero_spatial"][0]; b=env.envs[0].unwrapped; b.init_state_id=0
    o,_=env.reset(seed=[k(["libero_spatial",0,0],"environment")],options={NEW_ROLLOUT_OPTION:True})
    fr["image_%d"%size]=o["pixels"]["image"][0]; fr["image2_%d"%size]=o["pixels"]["image2"][0]; env.close()
# Method probe for fresh-state generation: one task, no renderer.
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import TASK_MAPPING
from libero.libero.envs.bddl_utils import get_problem_info
from robosuite import load_controller_config
t=benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
bd=os.path.join(get_libero_path("bddl_files"),t.problem_folder,t.bddl_file); info=get_problem_info(bd)
e=TASK_MAPPING[info["problem_name"].lower()](bddl_file_name=bd,robots=["Panda"],controller_configs=load_controller_config(default_controller="OSC_POSE"),has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,ignore_done=True,reward_shaping=True,control_freq=20,horizon=1000)
ss=[]
for seed in (1,2,1):
    np.random.seed(seed); e.reset(); ss.append(np.array(e.sim.get_state().flatten()))
print("gen_len",ss[0].size,"seed_repeat_equal",ss[0].tobytes()==ss[2].tobytes(),"seeds_differ",ss[0].tobytes()!=ss[1].tobytes(),"success_at_reset",bool(e._check_success()))
e.sim.set_state_from_flattened(ss[0]); e.sim.forward(); print("roundtrip",np.array(e.sim.get_state().flatten()).tobytes()==ss[0].tobytes())
offs=benchmark.get_benchmark_dict()["libero_spatial"]().get_task_init_states(0); print("official_len",offs.shape,"gen_differs_from_all_official",not any(np.allclose(offs[i][1:],ss[0][1:],atol=1e-3) for i in range(len(offs))),"min_linf_to_official",float(np.min(np.abs(offs[:,1:]-ss[0][1:]).max(axis=1))))
np.savez(out,**fr); print("saved",out)
