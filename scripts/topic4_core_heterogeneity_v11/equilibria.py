"""Continue the known low-rate equilibrium fold in mean-preserving h."""
from common import *
from branches import fold
from spectral import leading
from eigen import root_at
from scipy.linalg import eig
import argparse


def run():
    rows=[]
    levels=np.unique(np.r_[np.linspace(0,1,41),np.arange(.960,.981,.001)])
    for h in levels:
        s=System(h);r=np.array([.0001,.0002,0,0,0,0])
        f=None
        for g in np.arange(.9,1.301,.002):
            rn,err,ok=s.solve(g,r)
            if not ok or max(eig(s.jac(rn,g))[0].real)>0:
                ev,vv=eig(s.jac(r,g));v=vv[:,np.argmin(abs(ev))].real;v/=np.linalg.norm(v)
                try:f=fold(s,np.r_[r,g,v])
                except AssertionError:
                    # The two nearly independent cores can both be critical.
                    for pop in (1,0):
                        v=np.eye(6)[pop]
                        try:f=fold(s,np.r_[r,g,v]);break
                        except AssertionError:pass
                break
            r=rn
        if f is None:
            rows.append(dict(h=float(h),status='UNRESOLVED'));write('equilibrium_fold.json',rows);continue
        r=np.array(f['r_hz'])/1000
        vals=leading(s,r,f['g'],N=40)
        near=vals[np.argmin(abs(vals))]; other=np.delete(vals,np.argmin(abs(vals)))
        f.update(h=float(h),sigma_A_mV=h*s.original_std_A,status='REFINED',critical_core='AB'[int(np.argmax(np.abs(f['v'][:2])))],
                 leading_nonzero_real_per_s=float(other.real.max()),zero_generator_per_s=[near.real,near.imag])
        rows.append(f)
        write('equilibrium_fold.json',rows)
        print('FOLD',h,f['g'],r[0]*1000,other.real.max(),flush=True)
    return rows


if __name__=='__main__':run()
