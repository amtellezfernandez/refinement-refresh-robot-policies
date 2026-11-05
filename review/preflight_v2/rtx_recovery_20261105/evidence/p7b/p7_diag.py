# NON-COUNTING diagnostic driver. Imports the FROZEN server_openpi (code-078e166) and runs its own
# load(), KeyedPolicy and count_denoise_loop() unchanged. Instrumentation is runtime wrapping of three
# model methods (embed_prefix, embed_suffix, action_out_proj) that block on outputs and record memory.
# Usage: p7_diag.py <tooling_dir> <checkpoint> <out_json> <d list, e.g. 2,5,10>
import gc, json, os, subprocess, sys, time, traceback
from pathlib import Path
C, CKPT, OUTF, DS = Path(sys.argv[1]), sys.argv[2], sys.argv[3], tuple(int(x) for x in sys.argv[4].split(","))
sys.path.insert(0, str(C))
import jax, numpy as np
import server_openpi as so                      # frozen, unmodified
from openpi.models import pi0 as pi0mod
from openpi.policies.libero_policy import make_libero_example
GIB = 2 ** 30
events = []; state = {"d": None, "step": 0}
def smi():
    try:
        u, f = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        return int(u), int(f)
    except Exception as e:
        return None, None
def mark(name, **extra):
    s = jax.devices()[0].memory_stats() or {}
    used, free = smi()
    ev = {"t": round(time.time(), 3), "d": state["d"], "step": state["step"], "event": name,
          "jax_in_use_gib": round(s.get("bytes_in_use", 0) / GIB, 3), "jax_peak_gib": round(s.get("peak_bytes_in_use", 0) / GIB, 3),
          "jax_limit_gib": round(s.get("bytes_limit", 0) / GIB, 3), "jax_num_allocs": s.get("num_allocs"),
          "jax_largest_alloc_gib": round(s.get("largest_alloc_size", 0) / GIB, 3),
          "smi_used_mib": used, "smi_free_mib": free}
    ev.update(extra); events.append(ev); print("EV", json.dumps(ev), flush=True)
    json.dump({"ds": DS, "events": events}, open(OUTF, "w"), indent=0)
def wrap(cls, name, before, after):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        before(self)
        out = orig(self, *a, **k)
        jax.block_until_ready(out)
        after(self)
        return out
    setattr(cls, name, w)
wrap(pi0mod.Pi0, "embed_prefix", lambda s: mark("image_encode_start"), lambda s: mark("image_encode_done"))
def suffix_before(s):
    if state["step"] == 0:
        mark("prefix_llm_done_denoise_begin")
    mark("step_start")
wrap(pi0mod.Pi0, "embed_suffix", suffix_before, lambda s: None)
mark("process_start")
t = time.time(); loaded = so.load(CKPT); mark("model_loaded", load_s=round(time.time() - t, 1))
policy = so.KeyedPolicy(loaded); example = make_libero_example()
target = loaded._model.action_out_proj; L = type(target); orig_call = L.__call__
def proj(self, *a, **k):
    out = orig_call(self, *a, **k)
    if self is target:
        jax.block_until_ready(out); state["step"] += 1; mark("step_done")
    return out
L.__call__ = proj
# Per-d boundaries: wrap KeyedPolicy.infer, which count_denoise_loop calls once per d.
orig_infer = so.KeyedPolicy.infer
def infer(self, obs):
    state["d"] = obs.get("__d__"); state["step"] = 0; mark("call_start")
    out = orig_infer(self, obs)
    jax.block_until_ready(out["actions"]); mark("call_done", steps_completed=state["step"], actions_type=type(out["actions"]).__name__)
    gc.collect(); mark("after_call_gc", steps_completed=state["step"])
    return out  # real output, unchanged; frozen infer returns host numpy, so it holds no device buffer
so.KeyedPolicy.infer = infer
so.ALLOWED_D = DS                               # runtime override in this driver only; frozen file unchanged
status = "completed"; err = None
try:
    counts = so.count_denoise_loop(policy, example, {})
    mark("count_denoise_loop_returned", counts=counts)
except BaseException as e:
    status = "exception"; err = "".join(traceback.format_exception_only(type(e), e)).strip()
    mark("exception", error=err[:600])
json.dump({"ds": DS, "status": status, "error": err, "events": events}, open(OUTF, "w"), indent=0)
print("STATUS", status, flush=True)
sys.exit(0 if status == "completed" else 1)
