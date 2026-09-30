"""Directional finite differences of the full Galerkin Newton operator."""
from exact_periodic import *
import argparse


def main(a):
    s=model();attach_rate_entry_path(s);z=np.load(a.orbit);N=len(z['r'])
    ExactGalerkin.harmonic_block=33;o=ExactGalerkin(s,N,a.M,a.device);o.cache_mean_operators=False
    cp=o.cp;cp.fft.config.get_plan_cache().set_size(0);n=N*s.P
    r=cp.asarray(z['r']);T=float(z['T']);D=float(z['D']);y=cp.r_[(r/RS).ravel(),np.log(T),D*1000]
    phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=N,axis=0)
    c=cp.zeros(n+2);c[-2]=1000.;arc=(y.copy(),c,cp.ones(n+2));F,A,_=o.evaluate(y,r,phase,D,arc=arc,derivative=True)
    q=np.load(a.iterate);v=cp.r_[(cp.asarray(q['r'])/RS).ravel(),np.log(float(q['T'])),float(q['D'])*1000]-y
    v/=cp.max(abs(v));exact=A@v;rows=[]
    for h in [1e-2,1e-3,1e-4]:
        plus=o.evaluate(y+h*v,r,phase,D,arc=arc);minus=o.evaluate(y-h*v,r,phase,D,arc=arc)
        fd=(plus-minus)/(2*h);error=float(cp.linalg.norm(fd-exact)/cp.linalg.norm(exact))
        rows.append(dict(h=h,relative_error=error,max_error=float(cp.max(abs(fd-exact)))))
        log('BVP JACOBIAN',rows[-1]);del plus,minus,fd;cp.get_default_memory_pool().free_all_blocks()
    write(OUT/'bvp_jacobian_check.json',dict(status='PASS' if min(q['relative_error'] for q in rows)<1e-5 else 'FAIL',
        orbit=a.orbit,iterate=a.iterate,N=N,M=a.M,rows=rows,direction='actual recent Newton branch displacement'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('iterate');p.add_argument('--M',type=int,default=16384)
    p.add_argument('--device',type=int,default=1);main(p.parse_args())
