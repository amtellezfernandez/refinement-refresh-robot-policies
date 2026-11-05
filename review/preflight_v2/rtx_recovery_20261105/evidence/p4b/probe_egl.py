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
