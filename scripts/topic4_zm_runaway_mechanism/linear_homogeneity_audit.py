"""Check the small-correction roundoff floor without changing the rate model."""
from exact_periodic import *
import argparse


def main(a):
    s=model();attach_rate_entry_path(s);z=np.load(OUT/'periodic/rate_near_G2049_M8192.npz')
    s.set_D(float(z['D']));ExactGalerkin.harmonic_block=33
    o=ExactGalerkin(s,513,2048,a.device);o.cache_mean_operators=False;cp=o.cp;T=float(z['T'])
    def old(v):
        x=o.inputs(v,T,s.Z)
        for i in [0,3]:x[i]-=o.gp[3]
        for i in [1,4,6]:x[i]-=o.gp[4]
        return x
    v=cp.asarray(np.random.default_rng(724).normal(size=(o.N,s.P)))*.001
    zero=cp.zeros_like(v);old0=old(zero);new0=o.linear_inputs(zero,T,s.Z)
    base=o.linear_inputs(v,T,s.Z);normal=old(v);rows=[]
    for h in [1.,1e-4,1e-8,1e-12]:
        before=old(v*h);after=o.linear_inputs(v*h,T,s.Z);expected=base*h
        rows.append(dict(scale=h,old_relative_error=float(cp.linalg.norm(before-expected)/cp.linalg.norm(expected)),
                         new_relative_error=float(cp.linalg.norm(after-expected)/cp.linalg.norm(expected))))
    result=dict(old_zero_max=float(cp.max(abs(old0))),new_zero_max=float(cp.max(abs(new0))),rows=rows,
                ordinary_direction_relative_error=float(cp.linalg.norm(base-normal)/cp.linalg.norm(base)))
    assert result['new_zero_max']==0. and result['ordinary_direction_relative_error']<1e-12
    assert max(r['new_relative_error'] for r in rows)<1e-12
    result['status']='PASS';write(OUT/'linear_homogeneity_audit.json',result);log('LINEAR HOMOGENEITY',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args())
