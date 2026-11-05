W=$HOME/alba-cross-policy-20261101; OUT=$HOME/jobs/alba-v3-p4c-render; mkdir -p $OUT; cd $OUT
export LD_LIBRARY_PATH=$W/syslibs/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib
cat > probe_render.py <<'PY'
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
PY
for E in smolvla client openpi; do for GL in egl osmesa; do
  MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL timeout 120 $W/envs/$E/bin/python probe_render.py > r_${E}_${GL}.txt 2>&1; rc=$?
  echo "$E $GL rc=$rc $(grep -E 'RENDER_OK|FIRST_ERROR' r_${E}_${GL}.txt | cut -c1-200)"; grep -m3 -E "Error:|error:" r_${E}_${GL}.txt | cut -c1-200
done; done
sha256sum r_*.txt probe_render.py
