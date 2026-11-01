import hashlib,json,os,sys,time
import numpy as np
RECORDED=np.array([0.5000000000000003, -0.0001305661606154118, -0.17083620561307536, -0.024106275427977056, -2.4525674388386953, -0.00612139682158199, 2.2259022374621193, 0.7677453302367606, 0.03872974791904, -0.03872857216450282, -0.06347439952612711, 0.20202570582559568, 0.8984070750976497, 0.707106783005648, -1.38747552865996e-05, 1.6673002295879823e-06, 0.7071067792293566, -0.18872207605406094, 0.3203476467880973, 0.8984070750976497, 0.707106783005648, -1.387475528659693e-05, 1.6673002295876412e-06, 0.7071067792293566, 0.057859587539025564, 0.026432941343195505, 0.9093854774183913, 0.7071067811865475, 3.005070917632221e-17, -6.019088570868838e-18, 0.7071067811865476, -0.19738350623176887, 0.18914074180082066, 0.8993430792867946, 0.7071067844060658, -5.2297701515516656e-08, 1.137152900817182e-07, 0.7071067779670178, 0.0534172930109977, 0.2051800080313651, 0.9025063385290898, 0.7071068099946488, -1.3218930716110878e-05, 5.9052040077887025e-06, 0.7071067522302282, 0.0, 0.0, 0.0, 0.0, -0.005179244750431266, 0.001702944824561564, 0.0056204394448454605, 0.0015694061717649787, -0.0014894729456260735, -0.003873829533063857, -0.011704636154346236, 0.0031914302247762804, -0.0031850345232583453, 8.495918781109824e-15, -1.086782074952675e-14, 2.878957198982508e-11, -2.709422470080497e-12, 3.5491487369012805e-12, -1.0077458179367964e-15, 8.49684230709992e-15, -1.0867527545252061e-14, 2.878957132206554e-11, -2.709467851824859e-12, 3.5491306052190058e-12, -1.0080283417939455e-15, -3.5707805146444886e-21, 6.020517159537876e-21, 1.4494301576121124e-11, -6.021383207778499e-17, -2.2038889810472637e-16, -5.985830065777927e-31, -9.380962186974243e-18, -1.6860740779342563e-16, 2.4748050878462913e-11, 3.2896980869165164e-15, 1.4460807617765945e-13, 3.551640349922719e-19, -4.22717438665134e-15, 1.265833815854958e-14, 2.1638018183143722e-11, -1.8693590790714027e-12, 5.693760786347199e-12, 4.753536502412561e-16, 0.0, 0.0, 0.0, 0.0],dtype=np.float64)
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False,default=str).encode()
def sha(v): return hashlib.sha256(canonical(v)).hexdigest()
def keyed(k,d): return int.from_bytes(hashlib.sha256(canonical([d,k])).digest()[:4],"little")
def arr(v):
    a=np.ascontiguousarray(np.asarray(v)); h=json.dumps([a.dtype.str,list(a.shape)],separators=(",",":")).encode(); return hashlib.sha256(h+a.tobytes()).hexdigest()
def enc(v):
    if isinstance(v,dict): return {k:enc(x) for k,x in sorted(v.items())}
    if isinstance(v,np.ndarray): return arr(v)
    return v
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
cfg=LiberoEnv(task="libero_spatial",task_ids=[0],init_states=True,hard_reset=True,control_mode="relative",observation_height=360,observation_width=360)
env=make_env(cfg,n_envs=1,use_async_envs=False)["libero_spatial"][0]
b=env.envs[0].unwrapped; seed=keyed(["libero_spatial",0,0],"environment")
def reset():
    b.init_state_id=0; o,_=env.reset(seed=[seed],options={NEW_ROLLOUT_OPTION:True}); return o,np.array(b._env.sim.get_state().flatten(),dtype=np.float64)
o1,s1=reset(); o2,s2=reset()
print("repeat_reset state_equal",s1.tobytes()==s2.tobytes(),"obs_equal",sha(enc(o1))==sha(enc(o2)))
d=np.abs(s1-RECORDED); print("vs_recorded len",s1.size,RECORDED.size,"linf",d.max(),"n_diff",int((d>0).sum()),"n_gt_1e-9",int((d>1e-9).sum()),"n_gt_1e-6",int((d>1e-6).sum()),"argmax",int(d.argmax()),"time",s1[0],RECORDED[0])
print("obs_keys",{k:(list(v) if isinstance(v,dict) else v.shape) for k,v in o1.items()})
def leaf(p,v):
    if isinstance(v,dict):
        for k,x in sorted(v.items()): leaf(p+"/"+k,x)
    else: print(" ",p,getattr(v,'shape',None),getattr(v,'dtype',None),arr(v)[:12],"mean",round(float(np.mean(v)),3))
leaf("",o1)
t=time.time()
for _ in range(30): env.step(np.zeros((1,7)))
print("env_step_ms_with_render",round(1000*(time.time()-t)/30,1))
import platform,subprocess; print(subprocess.check_output("lscpu | grep -E 'Model name'",shell=True,text=True).strip())
import mujoco; print("mujoco",mujoco.__version__,"numpy",np.__version__)
