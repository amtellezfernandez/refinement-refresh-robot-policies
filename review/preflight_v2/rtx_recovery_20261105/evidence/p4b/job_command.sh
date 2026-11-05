W=$HOME/alba-cross-policy-20261101; OUT=$HOME/jobs/alba-v3-p4b-egl; mkdir -p $OUT; cd $OUT
echo "shell: $(readlink /proc/$$/exe)"
export LD_LIBRARY_PATH=$W/syslibs/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib
ls /usr/share/glvnd/egl_vendor.d/ /usr/lib/wsl/lib > egl_inventory.txt 2>&1
cat > probe_egl.py <<'PY'
import os, sys, traceback
print("MUJOCO_GL", os.environ.get("MUJOCO_GL"), "PYOPENGL_PLATFORM", os.environ.get("PYOPENGL_PLATFORM"), flush=True)
try:
    import mujoco
    print("mujoco", mujoco.__version__, flush=True)
    c = mujoco.GLContext(64, 64)
    print("STEP context_created", flush=True)
    c.make_current()
    print("STEP make_current_ok", flush=True)
    from OpenGL import GL
    print("vendor", GL.glGetString(GL.GL_VENDOR), "renderer", GL.glGetString(GL.GL_RENDERER), flush=True)
except BaseException:
    print("FIRST_ERROR", flush=True); traceback.print_exc(); sys.stdout.flush(); os._exit(2)
os._exit(0)
PY
for E in smolvla client openpi; do
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl timeout 120 $W/envs/$E/bin/python probe_egl.py > egl_$E.txt 2>&1; rc=$?
  echo "== env=$E MUJOCO_GL=egl rc=$rc"; grep -E "mujoco|STEP|vendor|FIRST_ERROR|Error|error" egl_$E.txt | head -12 | cut -c1-240
done
sha256sum egl_*.txt egl_inventory.txt probe_egl.py
