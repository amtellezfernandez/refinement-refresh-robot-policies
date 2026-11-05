W=$HOME/alba-cross-policy-20261101; OUT=$HOME/jobs/alba-v3-health; mkdir -p $OUT; cd $OUT
echo "== identity"; hostname; uname -srm; cat /proc/uptime; date -u
echo "== nvidia"; nvidia-smi --query-gpu=name,driver_version,vbios_version,memory.used,memory.total,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active --format=csv; nvidia-smi | sed -n 3p; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "== host"; free -g; df -h / ; ls -l $HOME/local-science-programme-20261008/gpu.lock; ps -eo pid,etime,rss,args --sort=-rss | head -5
echo "== dmesg (may be denied)"; dmesg 2>&1 | grep -i -E "oom|killed process|xid|nvrm|dxg.*err" | tail -10
echo "== ckpt pi05 rehash vs mini reference"
cat > exp_pi05.txt <<'H'
b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84  ./assets/physical-intelligence/libero/norm_stats.json
303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73  ./params/_METADATA
63f4c57ba6ff10f4132a639b9942eabcc26942eb34081be1d93bf7ddab816501  ./params/_sharding
2b29474f08aa50922da11074deb4d7f35e30f0071a088a30f40df88edc1ebdb0  ./params/array_metadatas/process_0
93b39327a1552b06b1b6ce5190d58131297bcca66c3f2a099bbb2882df7759af  ./params/d/98a77a52a8eb845ae4830eb7fe983979
65246951e69bd2b5118e609646bc9e8c439229bccbc1325643833aeb74f77104  ./params/manifest.ocdbt
0d4412552d4a69fba953f824a60d005ee7559822a9c1c16684906c7f0801b848  ./params/ocdbt.process_0/d/0eaaecefaa9720d30a32cc56e65fd345
ad09cf4b2cacc0628c713afbb1ee9decfd36e295fdacae111c1e0c5eeeb6cc00  ./params/ocdbt.process_0/d/155391c1cbf93a1be16266de052e5b48
3b81b2e8afe1456a4584d19dcf600f2bb6f27beff9f061b336f06657a11d5cb5  ./params/ocdbt.process_0/d/2b6985f48e9da86f68627a7608c5bc25
624a529b0b5f07bf535e07eba4f823e47fdde66a3d0b38295a35ac9d60cfcf0f  ./params/ocdbt.process_0/d/475fab3ee8821662585a8cde3eb32e22
1a5f80ef1aeaa82e70cca883c4d95da65bd410c8da19729d4814317533ba11f5  ./params/ocdbt.process_0/d/6c54da5a6f62c20a09c0f3f8c3329e00
c92342da7593219debce6133f148f3c24ece184ff53f9f4bac800c09fcc6dc1a  ./params/ocdbt.process_0/d/896bf93c5cf2e8ddd274a6ea0a2feec0
1e99d5876e6b8db2e12c43c8079f6b98cffac2c26bbd84e59cc505e983d146d2  ./params/ocdbt.process_0/d/bc613ff288a162563e622d01bb60622a
9c717ff13084524b74f330803421b8a261bc3009a477c933e1e1a1477a796fc3  ./params/ocdbt.process_0/d/bda87d9791f23df771cd2d15293780cc
6e62d35e0689dc12779d0b2e65304823d6787fdd4ccd754c9ebd059fdc4b6765  ./params/ocdbt.process_0/d/efbb46173882cb35ed41ffe0c2db8a5e
3bf70fbb0fac151675595b33aeb8203139e8809de17d606574ab606a758b3591  ./params/ocdbt.process_0/manifest.ocdbt
H
(cd $W/ckpt/checkpoints/pi05_libero && sha256sum -c $OUT/exp_pi05.txt) | tee pi05_check.txt | grep -c ': OK'; grep -v ': OK' pi05_check.txt; echo "files_on_disk: $(cd $W/ckpt/checkpoints/pi05_libero && find . -type f | wc -l)"
echo "== ckpt smolvla"; find $W/ckpt/smolvla_libero -name model.safetensors -exec sha256sum {} \;
echo "== torch (smolvla env)"; timeout 300 $W/envs/smolvla/bin/python -c "
import torch,time;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'dev',torch.cuda.get_device_name(0),'cap',torch.cuda.get_device_capability(0))
a=torch.randn(8192,8192,device='cuda');b=torch.randn(8192,8192,device='cuda');torch.cuda.synchronize();t=time.time()
for _ in range(50): c=a@b
torch.cuda.synchronize();dt=time.time()-t;print('matmul_50x8192_s',round(dt,3),'TFLOPS',round(50*2*8192**3/dt/1e12,1))
ref=(a[:256].double()@b[:,:256].double());err=(c[:256,:256].double()-ref).abs().max().item();print('max_abs_err_vs_fp64',err,'finite',bool(torch.isfinite(c).all()))
x=torch.empty(int(20e9)//4,device='cuda');x.fill_(1.0);print('alloc_20GB_sum_ok',x.sum().item()==x.numel());del x
print('peak_alloc_GiB',round(torch.cuda.max_memory_allocated()/2**30,2))"; echo "torch_rc=$?"
echo "== jax (openpi env)"; XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 300 $W/envs/openpi/bin/python -c "
import jax,jax.numpy as jnp;print('jax',jax.__version__,jax.devices());k=jax.random.PRNGKey(0);a=jax.random.normal(k,(4096,4096));print('jax_matmul_finite',bool(jnp.isfinite(a@a).all()))"; echo "jax_rc=$?"
nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
