import os, sys, hashlib, traceback
try:
    import mujoco
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></worldbody></mujoco>')
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 64, 64); r.update_scene(d); px = r.render()
    print("RENDER_OK", os.environ["MUJOCO_GL"], mujoco.__version__, px.shape, int((px > 0).sum()), hashlib.sha256(px.tobytes()).hexdigest(), flush=True)
except BaseException:
    print("FIRST_ERROR", flush=True); traceback.print_exc(); sys.stdout.flush(); os._exit(2)
os._exit(0)
