"""Independent dense trust-region root check for the intermediate rate gap."""
from bridge_rate_slices import *
from scipy.optimize import root
from scipy.special import expit,logit
import time


def main(a):
    s=ZMRate();base=DEST/'g20';dest=base/a.label;dest.mkdir(exist_ok=False)
    pool=[]
    for name in ['D_gap_lower_guarded','D_gap_rate_slices']:
        for row in read(base/name/'result.json')['rows']:
            if row.get('converged',True):pool.append((row['global_E_hz'],row['D'],base/name/f'point{row["index"]:04d}.npz'))
    low=max((p for p in pool if p[0]<a.target),key=lambda q:q[0]);high=min((p for p in pool if p[0]>a.target),key=lambda q:q[0])
    with np.load(low[2]) as z:rl=z['r'].copy()
    with np.load(high[2]) as z:rh=z['r'].copy()
    u=(a.target-low[0])/(high[0]-low[0]);blend=(1-u)*rl+u*rh;D0=(1-u)*low[1]+u*high[1]
    w=np.zeros(s.P);w[s.E]=s.mean_weights
    def fun(x):
        r=x[:-1]*RS;D=float(expit(x[-1]));return np.r_[s.residual(r,D),10*(w@r-a.target/1000)]
    def jac(x):
        r=x[:-1]*RS;D=float(expit(x[-1]));A=np.zeros((s.P+1,s.P+1))
        A[:-1,:-1]=s.jacobian(r,D).toarray()*RS
        A[:-1,-1]=s.parameter_derivative(r,D)*D*(1-D);A[-1,:-1]=10*w*RS
        return A
    uniform=blend.copy();uniform[s.E]=a.target/1000
    rows=[];started=time.time()
    for seedname,seed in [('blend',blend),('uniform_E',uniform),('low',rl),('high',rh)]:
        x0=np.r_[seed/RS,logit(D0)]
        result=root(fun,x0,jac=jac,method='hybr',options={'xtol':1e-10,'maxfev':700,'factor':.1})
        rr=result.x[:-1]*RS;D=float(expit(result.x[-1]));r=np.clip(rr,0,np.nextafter(1/s.ref,0))
        res=float(abs(s.residual(r,D)).max()*1000);rate=s.global_rate(r)
        ok=res<2e-8 and abs(rate-a.target)<2e-8
        row=dict(index=len(rows),seed=seedname,target_hz=a.target,converged=ok,D=D,
            global_E_hz=rate,equilibrium_residual_hz=res,rate_error_hz=rate-a.target,
            evaluations=result.nfev,message=result.message,wall_s=time.time()-started)
        if ok:np.savez_compressed(dest/f'point{len(rows):04d}.npz',r=r,D=D,Z=s.Z,tangent=tangent(s,r,D))
        rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows));print(row,flush=True)
        if ok:break
    write(dest/'result.json',dict(status='COMPLETE',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=float,default=140)
    p.add_argument('--label',default='D_gap_hybrid_140');main(p.parse_args())
