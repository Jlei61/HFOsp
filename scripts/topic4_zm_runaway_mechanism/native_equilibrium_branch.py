"""Follow the persistent-state equilibrium family on the actual spatial Z path.

This family need not be an attractor. No stability is inferred from convergence.
"""
from native_path import *
import equilibria_v3 as arc
import argparse


def main(a):
    s=model();attach_native_path(s);arc.RS=.1;arc.DS=.1
    z=np.load(OUT/'equilibria/native_unconstrained/t9870_tail_average.npz')
    r=z['r'];D=float(z['D']);s.set_D(D);assert abs(s.residual(r)).max()<1e-10
    # The seed is exactly an interpolation knot. Enter the requested open segment
    # before differentiating D; a centred derivative at the kink averages two paths.
    D+=a.direction*1e-5;s.set_D(D);r,ok,tr=s.solve(r)
    assert ok,('one-sided start failed',tr[-1])
    t=arc.tangent(s,r,D,direction=a.direction);out=OUT/f'equilibria/native_{a.label}'
    out.mkdir(parents=True,exist_ok=True);rows=[];folds=[];ds=.12;status='RUNNING'
    for k in range(a.steps):
        s.set_D(D);p=out/f'point{k:04d}.npz';np.savez_compressed(p,r=r,D=D,Z=s.Z,tangent=t)
        row=dict(index=k,path=str(p),D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
            residual_hz=float(abs(s.residual(r)).max()*1000),tangent_D=float(t[-1]),ds=ds,stability='NOT_ESTABLISHED')
        if rows and t[-1]*rows[-1]['tangent_D']<0:folds.append([k-1,k])
        rows.append(row);write(out/'result.json',dict(status=status,rows=rows,fold_brackets=folds))
        log(a.label,k,'D',D,'mean',row['global_E_hz'],'ds',ds)
        if not .215<D<.32:status='REQUESTED_RANGE_REACHED';break
        x=np.r_[r/arc.RS,D/arc.DS]
        for attempt in range(18):
            xx,ok,it=arc.correct(s,x+ds*t,t)
            if ok:
                nt=arc.tangent(s,xx[:-1]*arc.RS,float(xx[-1]*arc.DS),t)
                cosine=float(nt@t);corr=np.linalg.norm(xx-x-ds*t)/ds
                if cosine>.90 and corr<.25:break
                ok=False
            ds*=.5
        if not ok:status='NONCONVERGENCE';break
        r=xx[:-1]*arc.RS;D=float(xx[-1]*arc.DS);t=nt
        if it<=4 and cosine>.98 and corr<.1:ds=min(.25,ds*1.2)
        elif it>=10:ds*=.7
    else:status='REQUESTED_SEGMENT_COMPLETE'
    refined=[]
    for i,j in folds:
        L=dict(np.load(out/f'point{i:04d}.npz'));R=dict(np.load(out/f'point{j:04d}.npz'))
        L['D']=float(L['D']);R['D']=float(R['D']);q=arc.refine_fold(s,L,R)
        if q is None:refined.append(dict(bracket=[i,j],status='NOT_REFINED'));continue
        np.savez_compressed(out/f'fold{len(refined):03d}.npz',r=q.pop('r'),D=q['D'],Z=s.Z,v=q.pop('v'),w=q.pop('w'))
        q['bracket']=[i,j];refined.append(q)
    write(out/'result.json',dict(status=status,rows=rows,fold_brackets=folds,folds=refined))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--direction',type=int,choices=[-1,1],required=True)
    p.add_argument('--label',required=True);p.add_argument('--steps',type=int,default=1200);main(p.parse_args())
