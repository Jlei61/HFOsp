from common import *
from folds import Chart,metric
import numpy as np,argparse,time
p=argparse.ArgumentParser();p.add_argument('family');p.add_argument('--steps',type=int,default=120);p.add_argument('--parent');p.add_argument('--N',type=int,default=2048);a=p.parse_args()
parent=OUT/'arcs'/(a.parent or a.family+'_global')
if not a.parent:
    while not (parent/'completion.json').exists():time.sleep(2)
rows=read(parent/'progress.json');z0=np.load(rows[-1]['source']);N=a.N;scale=.01
from scipy.signal import resample
z=np.r_[(resample(z0['r'],N,axis=0)/.01).ravel(),np.log(float(z0['T'])),float(z0['g'])/scale]
tan=z0['tangent'];tan=np.r_[resample(tan[:-2].reshape(len(z0['r']),6),N,axis=0).ravel(),tan[-2:]];tan/=np.sqrt(metric(tan,tan,N))
dest=OUT/'arcs'/(a.family+'_fast');dest.mkdir(exist_ok=True);out=[];ds=.5;s=System()
for k in range(a.steps):
    chart=Chart(s,z,tan,N,scale)
    for retry in range(8):
        try:
            q,t,e,*_=chart.solve(z+ds*tan,ds);break
        except RuntimeError as exc:
            print('RETRY',a.family,k,ds,str(exc),flush=True);ds*=.5
    else:raise RuntimeError('arc retries exhausted')
    if metric(t,tan,N)<0:t=-t
    r=q[:-2].reshape(N,6)*.01;T=float(np.exp(q[-2]));g=float(q[-1]*scale)
    path=dest/f'point{k:03d}_g{g:.10f}.npz';np.savez_compressed(path,r=r,T=T,g=g,N=N,residual=e,tangent=t,gscale=scale)
    row=dict(index=k,g=g,T_ms=T,N=N,residual=e,tangent_g=float(t[-1]*scale),mean_hz=(r.mean(0)*1000).tolist(),source=str(path))
    out.append(row);(dest/'progress.json').write_text(json.dumps(out,indent=2)+'\n');print('FAST_ACCEPT',row,flush=True)
    z=q;tan=t;ds=min(3.,ds*1.3)
    if (dest/'STOP').exists() or T>2500 or not .8<g<1.6:
        (dest/'completion.json').write_text(json.dumps(dict(reason='requested stop' if (dest/'STOP').exists() else 'finite continuation boundary',last=row)));break
else:
    (dest/'completion.json').write_text(json.dumps(dict(reason='completed requested continuation points; finite endpoint, not established bifurcation',last=out[-1]),indent=2)+'\n')
