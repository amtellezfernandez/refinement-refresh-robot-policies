"""PDR T01: float32 trip count of pi0.py's while_loop condition (time >= -dt/2) for d=1..200. CPU, source arithmetic only."""
import numpy as np, jax
bad = []
for d in range(1, 201):
    dt = np.float32(-1.0) / np.float32(d); t = np.float32(1.0); k = 0
    while t >= -dt / np.float32(2):
        t = np.float32(t + dt); k += 1
    if k != d: bad.append((d, k))
print("numpy float32 mismatches 1..200:", bad)
@jax.jit
def trips(n):
    dt = -1.0 / n
    return jax.lax.while_loop(lambda c: c[0] >= -dt / 2, lambda c: (c[0] + dt, c[1] + 1), (1.0, 0))[1]
print("jax-jit CPU mismatches 1..200:", [(d, int(trips(d))) for d in range(1, 201) if int(trips(d)) != d], "jax", jax.__version__)
