W=$HOME/alba-cross-policy-20261101
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 timeout 120 $W/envs/openpi/bin/python -c "
import jax; d = jax.devices()[0]; s = d.memory_stats() or {}
print('CONFIG_ACCEPTED', d, 'bytes_limit_GiB', round(s.get('bytes_limit', 0) / 2**30, 2))" > $HOME/jobs/alba-v3-p5pre.txt 2>&1; rc=$?
echo "python_rc=$rc"; tail -5 $HOME/jobs/alba-v3-p5pre.txt | cut -c1-300
