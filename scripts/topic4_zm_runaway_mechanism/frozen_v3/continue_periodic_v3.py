"""Pseudo-arclength continuation of periodic orbits in D (prescribed Z path) for the v3 model.

Each step: predictor along the tangent (secant of the last two orbits in the (r, log T, D/1e-3) space),
corrector = the phase-fixed periodic BVP with an arclength constraint. Records every converged orbit
(orbits/), D turning points (brackets) for cycle-fold refinement, and the regional readouts.
"""
from periodic_v3 import *
import argparse
def main(a):
    s=load_model(a.device);o=PeriodicV3(s,a.N,a.device);cp=o.cp;dest=PERIODIC_OUT/a.label;dest.mkdir(parents=True,exist_ok=True);(dest/'orbits').mkdir(exist_ok=True)
    def pack(sol):return np.r_[(sol['r']*1000).ravel(),np.log(sol['T']),sol['D']*1000]
    def unpack(y):n=a.N*s.P;return dict(r=y[:n].reshape(a.N,s.P)/1000,T=float(np.exp(y[n])),D=float(y[n+1]/1000))
    z=np.load(a.first);s1=dict(r=resample(z['r'],a.N,axis=0),T=float(z['T']),D=float(z['D']));z=np.load(a.second);s2=dict(r=resample(z['r'],a.N,axis=0),T=float(z['T']),D=float(z['D']))
    y1=pack(s1);y2=pack(s2);weight=np.full_like(y1,a.w_r/np.sqrt(a.N*s.P));weight[-2]=a.w_logT;weight[-1]=a.w_D
    rows=[];ds=a.ds;brackets=[];status='RUNNING';prev_tD=None
    if a.resume:
        info=read(dest/'continuation.json');rows=info['rows'];brackets=info['brackets'];ds=rows[-1]['ds'];prev_tD=rows[-1]['tangent_D']
    for k in range(a.steps):
        tan=(y2-y1);tan/=np.sqrt(np.sum((tan*weight)**2));pred=y2+ds*tan;sp=unpack(pred);ok=False
        for attempt in range(8):
            sol=o.solve(sp['r'],sp['T'],sp['D'],arc=(pred,tan,weight),maxiter=a.maxiter,tol=a.tol)
            if sol['residual']<a.tol*5:ok=True;break
            ds*=.5;pred=y2+ds*tan;sp=unpack(pred);log('  retry with ds',ds)
        if not ok:status='CONTINUATION_STOPPED_NONCONVERGENCE';break
        y1,y2=y2,pack(sol);tD=float(tan[-1]);g=sol['r'][:,s.E]@s.mean_weights*1000
        name=f'{a.label}_{len(rows):04d}';path=dest/'orbits'/f'{name}.npz';np.savez_compressed(path,r=sol['r'],T=sol['T'],D=sol['D'],residual=sol['residual'])
        row=dict(index=len(rows),orbit=str(path),D=sol['D'],T_ms=sol['T'],global_mean_hz=float(g.mean()),global_min_hz=float(g.min()),global_max_hz=float(g.max()),
            regional_mean_hz=np.array([s.regional_rates(x) for x in sol['r']]).mean(0).tolist(),residual=sol['residual'],ds=ds,tangent_D=tD,newton_iterations=len(sol['history']))
        if prev_tD is not None and prev_tD*tD<0:brackets.append([len(rows)-1,len(rows)]);log('CYCLE FOLD BRACKET',brackets[-1])
        prev_tD=tD;rows.append(row);write(dest/'continuation.json',dict(status=status,rows=rows,brackets=brackets,N=a.N,label=a.label))
        log('ORBIT',row['index'],'D %.6f'%sol['D'],'T %.3f'%sol['T'],'mean %.3f Hz'%row['global_mean_hz'],'max %.1f'%row['global_max_hz'],'tD %.3f'%tD,'ds %.3f'%ds,'it',row['newton_iterations'])
        if len(sol['history'])<=4:ds=min(ds*1.3,a.max_ds)
        if not (a.D_min<=sol['D']<=a.D_max):status='RANGE_REACHED';break
    else:status='REQUESTED_STEPS_COMPLETE'
    write(dest/'continuation.json',dict(status=status,rows=rows,brackets=brackets,N=a.N,label=a.label))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',required=True);p.add_argument('--N',type=int,default=128)
    p.add_argument('--steps',type=int,default=40);p.add_argument('--ds',type=float,default=.5);p.add_argument('--max-ds',type=float,default=3.);p.add_argument('--w-logT',type=float,default=100.);p.add_argument('--w-D',type=float,default=1.);p.add_argument('--w-r',type=float,default=1.,help='weight of the rate block (RMS-normalised); 1 = RMS of rates in Hz counts like 1 unit of D/1e-3')
    p.add_argument('--D-min',type=float,default=0.);p.add_argument('--D-max',type=float,default=1.);p.add_argument('--maxiter',type=int,default=14);p.add_argument('--tol',type=float,default=2e-8)
    p.add_argument('--device',type=int,default=0);p.add_argument('--resume',action='store_true');main(p.parse_args())
