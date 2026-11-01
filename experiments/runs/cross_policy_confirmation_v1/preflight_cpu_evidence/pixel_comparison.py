import sys,glob,numpy as np,torch
S=sys.argv[1]; f=np.load(glob.glob(S+'/pix/**/frames.npz',recursive=True)[0])
import numpy._core.multiarray as M, numpy.dtypes as D
allow=[M._reconstruct,np.ndarray,np.dtype]+[getattr(D,n) for n in ('UInt8DType','Float64DType','Float32DType','Int64DType','BoolDType')]
with torch.serialization.safe_globals(allow): d=torch.load(sys.argv[2],weights_only=True,map_location='cpu')
print(list(d)); r=d['raw_observations']; print(type(r),list(r)[:8] if hasattr(r,'__iter__') else '')
def walk(p,v):
    if isinstance(v,dict):
        for k,x in v.items(): walk(p+'/'+str(k),x)
    elif hasattr(v,'shape'): print(p,tuple(v.shape),v.dtype)
walk('',r)
def get(v):
    return v['native' if 'native' in v else list(v)[0]]
base=get(r); px=base['pixels'] if 'pixels' in base else base
for key,name in (('image','image_360'),('image2','image2_360')):
    a=px[key]; a=(a.numpy() if hasattr(a,'numpy') else np.asarray(a)).squeeze(); b=f[name]
    diff=np.abs(a.astype(int)-b.astype(int))
    print(key,a.shape,b.shape,'identical',bool((a==b).all()),'frac_pixels_channels_diff %.5f'%float((diff>0).mean()),'mean_abs %.4f'%float(diff.mean()),'max %d'%int(diff.max()),'p99 %.1f'%float(np.percentile(diff,99)),'frac>8 %.5f'%float((diff>8).mean()))
    # negative control: the other camera must differ strongly
o=px['image2'];o=(o.numpy() if hasattr(o,'numpy') else np.asarray(o)).squeeze(); print('control other camera mean_abs %.2f'%float(np.abs(o.astype(int)-f['image_360'].astype(int)).mean()))
