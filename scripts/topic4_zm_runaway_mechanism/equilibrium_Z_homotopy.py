"""Numerical homotopy from a verified equilibrium to the actual rate Z path.

Intermediate spatial fields are solver aids, not a physiological trajectory or
bifurcation diagram. Only a root at the exact target field can enter that diagram.
"""
from equilibrium_unconstrained import *
import equilibria_v3 as arc
import argparse


class FieldHomotopy:
    def __init__(self,base,left,right):
        self.base=base;self.left=left;self.right=right;self.D=0.
    def __getattr__(self,name):return getattr(self.base,name)
    def set_D(self,value):
        self.D=float(value);self.base.set_Z((1-value)*self.left+value*self.right)


def main(a):
    base=model();attach_rate_entry_path(base);base.set_D(a.D);target=base.Z.copy()
    seed=np.load(OUT/'equilibria/native_unconstrained/t9870_tail_average.npz')
    s=FieldHomotopy(base,seed['Z'],target);s.set_D(0.);r=seed['r'];assert abs(s.residual(r)).max()<1e-10
    arc.RS=.1;arc.DS=.1;lam=0.;t=arc.tangent(s,r,lam);step=.1;rows=[]
    out=OUT/f'equilibria/rate_homotopy_D{a.D:.7f}';out.mkdir(parents=True,exist_ok=True)
    status='REQUESTED_SEGMENT_COMPLETE'
    for k in range(a.steps):
        row=dict(index=k,homotopy_lambda=lam,physical_D=base.D,global_E_hz=s.global_rate(r),
                 residual_per_ms=float(abs(s.residual(r)).max()),tangent_lambda=float(t[-1]))
        rows.append(row)
        if k%25==0:
            log('Z HOMOTOPY',row);write(out/'result.json',dict(status='RUNNING',target_D=a.D,rows=rows))
            np.savez_compressed(out/'last.npz',r=r,lambda_=lam,Z=base.Z,tangent=t)
        if lam>.995:
            s.set_D(1.);rr,ok,tr=solve(s,r)
            if ok:
                y=s.equilibrium_state(rr);f,_=s.rhs(y,[A@rr for A in s.matrices()],dynamic_z=False)
                assert abs(f).max()<1e-7
                np.savez_compressed(out/'target.npz',r=rr,D=a.D,Z=target,state=y)
                status='TARGET_ROOT_REACHED';log(status,a.D,s.global_rate(rr));break
            s.set_D(lam)
        x=np.r_[r/arc.RS,lam/arc.DS]
        for attempt in range(22):
            nxt,ok,it=arc.correct(s,x+step*t,t)
            if ok:
                nt=arc.tangent(s,nxt[:-1]*arc.RS,float(nxt[-1]*arc.DS),t)
                cosine=float(nt@t);correction=float(np.linalg.norm(nxt-x-step*t)/step)
                if cosine>.9 and correction<.3:break
                ok=False
            step*=.5
        if not ok:status='CORRECTOR_UNRESOLVED';break
        r=nxt[:-1]*arc.RS;lam=float(nxt[-1]*arc.DS);t=nt
        s.set_D(lam)
        if it<=4 and cosine>.98 and correction<.1:step=min(.25,step*1.2)
        elif it>=10:step*=.7
        if k>0 and lam<1e-8:status='RETURNED_TO_HOMOTOPY_START';break
    write(out/'result.json',dict(status=status,target_D=a.D,rows=rows,
          claim='Intermediate lambda is a numerical root-finding path, not the scientific D coordinate'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--D',type=float,default=.14502)
    p.add_argument('--steps',type=int,default=2000);main(p.parse_args())
