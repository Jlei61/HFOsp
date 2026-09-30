"""Equilibrium continuation on the same local rate Z family as the cycles."""
from equilibrium_unconstrained import *
import equilibria_v3 as arc
import argparse


def main(a):
    s=model();attach_rate_entry_path(s);arc.RS=.1;arc.DS=.1
    seed=np.load(a.seed);r=seed['r'];D=float(seed['D']);s.set_D(D)
    assert abs(s.Z-seed['Z']).max()<1e-12 and abs(s.residual(r)).max()<1e-10
    t=arc.tangent(s,r,D,direction=a.direction);step=.1;rows=[];folds=[]
    out=OUT/'equilibria'/a.label;out.mkdir(parents=True,exist_ok=True)
    status='REQUESTED_SEGMENT_COMPLETE'
    for k in range(a.steps):
        s.set_D(D);file=out/f'point{k:04d}.npz'
        np.savez_compressed(file,r=r,D=D,Z=s.Z,tangent=t)
        row=dict(index=k,D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
                 residual_per_ms=float(abs(s.residual(r)).max()),tangent_D=float(t[-1]),
                 path=str(file),stability='NOT_ESTABLISHED')
        if rows and t[-1]*rows[-1]['tangent_D']<0:folds.append([k-1,k])
        rows.append(row);write(out/'result.json',dict(status='RUNNING',rows=rows,fold_brackets=folds))
        if k%10==0:log('RATE EQUILIBRIUM',row)
        if not a.minimum_D<D<a.maximum_D:status='REQUESTED_RANGE_REACHED';break
        x=np.r_[r/arc.RS,D/arc.DS]
        for attempt in range(22):
            try:nxt,ok,it=arc.correct(s,x+step*t,t)
            except ValueError:ok=False
            if ok:
                nt=arc.tangent(s,nxt[:-1]*arc.RS,float(nxt[-1]*arc.DS),t)
                cosine=float(nt@t);correction=float(np.linalg.norm(nxt-x-step*t)/step)
                if cosine>.9 and correction<.3:break
                ok=False
            step*=.5
        if not ok:status='CORRECTOR_UNRESOLVED';break
        r=nxt[:-1]*arc.RS;D=float(nxt[-1]*arc.DS);t=nt
        if it<=4 and cosine>.98 and correction<.1:step=min(.25,step*1.2)
        elif it>=10:step*=.7
    write(out/'result.json',dict(status=status,rows=rows,fold_brackets=folds,
          scope='Actual local rate spatial Z path, dynamic M; roots are not automatically attractors'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed');p.add_argument('--direction',type=int,choices=[-1,1],required=True)
    p.add_argument('--label',required=True);p.add_argument('--steps',type=int,default=800)
    p.add_argument('--minimum-D',type=float,default=.14298039);p.add_argument('--maximum-D',type=float,default=.14769197)
    main(p.parse_args())
