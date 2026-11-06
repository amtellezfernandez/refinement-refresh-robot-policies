#!/usr/bin/env python3
"""PDR Phase A: root cause of eager pi0.5 counting retention. NON-COUNTING DIAGNOSTIC.

Synthetic input only (openpi make_libero_example). Imports the FROZEN server_openpi unchanged;
all instrumentation is runtime wrapping inside this driver.

Steps, one process, one variable changed at a time:
  S1 eager d=2 via frozen count_denoise_loop (ALLOWED_D patched in-driver) -> census of new live arrays
  S2 gc.collect -> census                              (gc alone)
  S3 jax.clear_caches + gc -> census                   (H1: holder is a JAX trace/compile cache)
  S4 release hook ON (clear_caches after every ToNNX call) -> frozen count at d=2,5,10
  S5 release hook OFF -> eager d=2 again -> census     (restore: retention must return if H1 causal)
usage: pdr_a_memory.py CODE_DIR CKPT OUT_JSON
"""
import gc, json, subprocess, sys, time
from collections import Counter

C, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import jax, numpy as np  # noqa: E402
import server_openpi as so  # noqa: E402
from openpi.policies.libero_policy import make_libero_example  # noqa: E402
from flax.nnx import bridge as nnx_bridge  # noqa: E402

GiB = 2 ** 30
R = {"events": [], "jax": jax.__version__, "device": str(jax.local_devices()[0])}


def smi():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception as e:  # noqa: BLE001
        return repr(e)


def census():
    arrs = jax.live_arrays()
    d = {id(a): (tuple(a.shape), str(a.dtype), int(a.nbytes)) for a in arrs}
    del arrs
    return d


def mark(label, base=None, extra=None):
    st = jax.local_devices()[0].memory_stats() or {}
    c = census()
    ev = {"label": label, "t": time.time(), "bytes_in_use_gib": st.get("bytes_in_use", 0) / GiB,
          "peak_gib": st.get("peak_bytes_in_use", 0) / GiB, "live_arrays": len(c),
          "live_bytes_gib": sum(v[2] for v in c.values()) / GiB, "smi_used_mib": smi()}
    if base is not None:
        new = {k: v for k, v in c.items() if k not in base}
        ev["new_arrays"] = len(new)
        ev["new_bytes_gib"] = sum(v[2] for v in new.values()) / GiB
        hist = Counter()
        nb = Counter()
        for v in new.values():
            hist[(v[0], v[1])] += 1
            nb[(v[0], v[1])] += v[2]
        ev["new_top"] = [[str(k[0]), k[1], hist[k], nb[k] / GiB] for k, _ in nb.most_common(12)]
        ev["new_match_param"] = match_params(new)
    if extra:
        ev.update(extra)
    R["events"].append(ev)
    print("MARK", label, round(ev["bytes_in_use_gib"], 3), ev.get("new_bytes_gib"), flush=True)
    return c


def match_params(new):
    """Fraction of new bytes whose (shape,dtype) equals a param leaf, or a param leaf minus its leading (layer) axis."""
    full, sliced = set(), set()
    for (shape, dt) in PARAM_SIGS:
        full.add((shape, dt))
        if len(shape) > 1:
            sliced.add((shape[1:], dt))
    tot = sum(v[2] for v in new.values()) or 1
    f = sum(v[2] for v in new.values() if (v[0], v[1]) in full)
    s = sum(v[2] for v in new.values() if (v[0], v[1]) in sliced and (v[0], v[1]) not in full)
    s_any = sum(v[2] for v in new.values() if v[0] in {x[0] for x in sliced})
    return {"full_param_shape_frac": f / tot, "layer_slice_shape_frac": s / tot, "layer_slice_shape_anydtype_frac": s_any / tot}


def referrers(base, k=3):
    """Type chain of referrers for the k largest arrays created since base."""
    arrs = [a for a in jax.live_arrays() if id(a) not in base]
    arrs.sort(key=lambda a: -a.nbytes)
    out = []
    for a in arrs[:k]:
        lvl1 = [r for r in gc.get_referrers(a) if r is not arrs]
        row = {"shape": str(a.shape), "dtype": str(a.dtype), "lvl1": [type(r).__name__ for r in lvl1][:6], "lvl2": []}
        for r in lvl1[:3]:
            row["lvl2"].append([type(x).__name__ + (":" + getattr(x, "__qualname__", "")[:60] if callable(x) else "")
                                for x in gc.get_referrers(r) if x is not lvl1][:6])
        out.append(row)
    del arrs
    return out


so.ALLOWED_D = (2,)
loaded = so.load(CKPT)
model = loaded._model
from flax import nnx  # noqa: E402
PARAM_SIGS = {(tuple(x.shape), str(x.dtype)) for x in jax.tree.leaves(nnx.state(model))}
R["param_bytes_gib"] = sum(x.nbytes for x in jax.tree.leaves(nnx.state(model))) / GiB
param_ids0 = {id(x) for x in jax.tree.leaves(nnx.state(model))}
keyed = so.KeyedPolicy(loaded)
example = make_libero_example()
base = mark("loaded")

# S1: frozen eager count at d=2
counts = so.count_denoise_loop(keyed, example, chunks := {})
c1 = mark("S1_eager_d2_returned", base, {"counts": counts})
R["S1_referrers"] = referrers(base)
R["S1_param_ids_unchanged"] = param_ids0 == {id(x) for x in jax.tree.leaves(nnx.state(model))}
eager_d2 = chunks["2"]

# S2: gc alone
gc.collect()
mark("S2_after_gc", base)

# S3: clear JAX caches (one variable)
jax.clear_caches(); gc.collect()
mark("S3_after_clear_caches", base)

# S4: release hook ON
orig_call = nnx_bridge.ToNNX.__call__
HOOK = {"on": False, "n": 0}


def releasing(self, *a, **kw):
    out = orig_call(self, *a, **kw)
    if HOOK["on"]:
        jax.block_until_ready(out)
        jax.clear_caches()
        HOOK["n"] += 1
    return out


nnx_bridge.ToNNX.__call__ = releasing
HOOK["on"] = True
so.ALLOWED_D = (2, 5, 10)
s4_counts, s4_chunks = {}, {}
for d in (2, 5, 10):
    so.ALLOWED_D = (d,)
    jax.local_devices()[0]  # noqa
    pre = census()
    try:
        cc = so.count_denoise_loop(keyed, example, ch := {})
        s4_counts[str(d)] = cc[str(d)]
        s4_chunks[str(d)] = ch[str(d)]
        mark("S4_release_d%d_returned" % d, pre, {"count": cc[str(d)], "hook_calls": HOOK["n"]})
    except Exception as e:  # noqa: BLE001
        mark("S4_release_d%d_FAILED" % d, pre, {"error": repr(e)[:400]})
        break
R["S4_counts"] = s4_counts
if "2" in s4_chunks:
    R["S4_d2_release_vs_noreleaseeager_max_abs"] = float(np.abs(s4_chunks["2"] - eager_d2).max())
    R["S4_d2_bitwise_equal"] = bool(np.array_equal(s4_chunks["2"], eager_d2))
R["S4_chunk_sha"] = {k: __import__("hashlib").sha256(np.asarray(v).tobytes()).hexdigest() for k, v in s4_chunks.items()}

# S5: restore (hook OFF), eager d=2 again
HOOK["on"] = False
jax.clear_caches(); gc.collect()
pre = mark("S5_pre")
so.ALLOWED_D = (2,)
try:
    so.count_denoise_loop(keyed, example, {})
    mark("S5_restore_d2_returned", pre)
    gc.collect()
    mark("S5_restore_after_gc", pre)
except Exception as e:  # noqa: BLE001
    mark("S5_restore_FAILED", pre, {"error": repr(e)[:400]})
R["status"] = "completed"
json.dump(R, open(OUT, "w"), indent=1, default=str)
print("STATUS completed", flush=True)
