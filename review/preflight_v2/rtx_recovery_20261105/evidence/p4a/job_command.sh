W=$HOME/alba-cross-policy-20261101; OUT=$HOME/jobs/alba-v3-p4-egl; mkdir -p $OUT; cd $OUT
export LD_LIBRARY_PATH=$W/syslibs/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib
echo "== EGL libraries visible"; ls /usr/lib/wsl/lib | grep -i -E "egl|libcuda.so$" ; ldconfig -p | grep -i -E "libEGL(_nvidia)?\.so" | head -5; ls /usr/share/glvnd/egl_vendor.d/ 2>&1
for E in smolvla client openpi; do
  PY=$W/envs/$E/bin/python
  for GL in egl osmesa; do
    echo "== env=$E MUJOCO_GL=$GL"
    MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL timeout 120 $PY -c "
import os, mujoco
print('mujoco', mujoco.__version__)
c = mujoco.GLContext(64, 64); c.make_current()
from OpenGL import GL
print('vendor', GL.glGetString(GL.GL_VENDOR), 'renderer', GL.glGetString(GL.GL_RENDERER), 'version', GL.glGetString(GL.GL_VERSION))
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"box\" size=\".1 .1 .1\" rgba=\"1 0 0 1\"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d); px = r.render()
import hashlib; print('render_ok shape', px.shape, 'nonzero', int((px > 0).sum()), 'sha256', hashlib.sha256(px.tobytes()).hexdigest()[:16])
" 2>&1 | tail -4; echo "rc_${E}_${GL}=${PIPESTATUS[0]}"
  done
done
