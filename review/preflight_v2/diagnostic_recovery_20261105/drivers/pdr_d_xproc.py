#!/usr/bin/env python3
"""PDR D01: cross-process reproducibility of compiled pi0.5 outputs (seeded synthetic input). NON-COUNTING.
Prints SHA of raw compiled actions at d=2,5,10. Called twice per XLA_FLAGS arm by the job.
usage: pdr_d_xproc.py CODE_DIR CKPT OUT_JSON
"""
import hashlib, json, os, sys

C, CKPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, C)
import numpy as np  # noqa: E402
import server_openpi as so  # noqa: E402
from openpi.policies.libero_policy import make_libero_example  # noqa: E402

np.random.seed(20261105)
example = make_libero_example()
keyed = so.KeyedPolicy(so.load(CKPT))
R = {"XLA_FLAGS": os.environ.get("XLA_FLAGS", ""), "sha": {}, "first": {}}
for d in (2, 5, 10):
    a = np.asarray(keyed.infer(dict(example, __d__=d, __noise_key__=json.dumps(["fixed-noise"])))["actions"])
    R["sha"][str(d)] = hashlib.sha256(a.tobytes()).hexdigest()
    R["first"][str(d)] = a[0].tolist()
json.dump(R, open(OUT, "w"))
print("STATUS completed", json.dumps(R["sha"]), flush=True)
