#!/usr/bin/env python3
"""PDR X: eager(+cache release) vs compiled on ONE seeded synthetic input, one process. NON-COUNTING DIAGNOSTIC.

Corrects a void cross-process comparison (A01 vs B01 used different unseeded make_libero_example inputs).
usage: pdr_x_eager_vs_compiled.py CODE_DIR CKPT OUT_JSON
"""
import hashlib, json, sys

C, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import jax, numpy as np  # noqa: E402
import server_openpi as so  # noqa: E402
from openpi.policies.libero_policy import make_libero_example  # noqa: E402
from flax.nnx import bridge as nnx_bridge  # noqa: E402

sha = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()  # noqa: E731
np.random.seed(20261105)
example = make_libero_example()
R = {"example_sha": {k: (sha(v) if not isinstance(v, str) else v) for k, v in example.items()}}
loaded = so.load(CKPT)
keyed = so.KeyedPolicy(loaded)
FK = json.dumps(["fixed-noise"])
comp = {d: np.asarray(keyed.infer(dict(example, __d__=d, __noise_key__=FK))["actions"]) for d in (2, 5, 10)}
comp2 = {d: np.asarray(keyed.infer(dict(example, __d__=d, __noise_key__=FK))["actions"]) for d in (2, 5, 10)}
orig = nnx_bridge.ToNNX.__call__


def releasing(self, *a, **kw):
    out = orig(self, *a, **kw)
    jax.block_until_ready(out)
    jax.clear_caches()
    return out


nnx_bridge.ToNNX.__call__ = releasing
eager, counts = {}, {}
for d in (2, 5, 10):
    so.ALLOWED_D = (d,)
    ch = {}
    counts[str(d)] = so.count_denoise_loop(keyed, example, ch)[str(d)]
    eager[d] = np.asarray(ch[str(d)])
    print("STATUS eager", d, counts[str(d)], flush=True)
nnx_bridge.ToNNX.__call__ = orig
R["eager_counts"] = counts
R["compiled_repeat_bitwise"] = {str(d): bool(np.array_equal(comp[d], comp2[d])) for d in comp}
R["eager_vs_compiled_max_abs"] = {str(d): float(np.abs(eager[d] - comp[d]).max()) for d in comp}
R["eager_vs_compiled_bitwise"] = {str(d): bool(np.array_equal(eager[d], comp[d])) for d in comp}
R["compiled_between_d_max_abs"] = {"2v5": float(np.abs(comp[2] - comp[5]).max()), "5v10": float(np.abs(comp[5] - comp[10]).max())}
R["compiled_sha"] = {str(d): sha(comp[d]) for d in comp}
R["eager_sha"] = {str(d): sha(eager[d]) for d in eager}
R["status"] = "completed"
json.dump(R, open(OUT, "w"), indent=1)
print("STATUS completed", flush=True)
