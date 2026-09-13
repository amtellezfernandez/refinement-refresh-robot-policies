"""Portable array records shared by runtime instrumentation and offline tests."""
import hashlib
import json

import numpy as np


def array_record(value):
    a = np.ascontiguousarray(value)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite state/action")
    header = json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":"))
    return {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(),
            "sha256": hashlib.sha256(header.encode() + a.tobytes()).hexdigest()}
