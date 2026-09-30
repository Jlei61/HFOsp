"""Delayed equilibrium spectrum with only frequency-independent gains cached."""
from native_path import *
from root_count_v3 import count,refine_root
from scipy import sparse
import argparse


def cache_characteristic(s,r):
    original=s.characteristic
    mu,ve,vi=s.moments(r);g=s.phi(mu,ve,vi);w,_=s.resp.weights(s,mu,ve,vi)
    al,aE,aI,eE,eI=w;tf,ts,tE,tI,tvE,tvI=s.poles
    def characteristic(rr,lam,dynamic_z=False):
        assert not dynamic_z and np.array_equal(rr,r)
        a,b,qa,qb=s.matrices(lam)
        Hm=al+(1-al)/(1+lam*ts);HE=aE+(1-aE)/(1+lam*tvE);HI=aI+(1-aI)/(1+lam*tvI)
        gm=g['d_mu']*Hm;gE=g['d_ve']*HE+g['d_mu']*eE*lam*tE/(1+lam*tE)
        gI=g['d_vi']*HI+g['d_mu']*eI*lam*tI/(1+lam*tI)
        ha=1/((1+lam*s.rise[0])*(1+lam*s.decay[0]));hg=1/((1+lam*s.rise[1])*(1+lam*s.decay[1]))
        K=sparse.diags(gm*s.tm)@(s.area[0]*ha*a-sparse.diags(s.Z*s.area[1]*hg)@b)
        K+=sparse.diags(gE*s.tm*s.area[0]**2/(1+lam*s.tau[0]/2))@qa
        K+=sparse.diags(gI*s.Z**2*s.tm*s.area[1]**2/(1+lam*s.tau[1]/2))@qb
        return (sparse.eye(s.P,format='csc')+sparse.diags(.5*s.E*gm/(1+lam*1000))-K).tocsc()
    checks=[]
    for lam in [0.,.01+.03j,.2j,2j]:
        diff=characteristic(r,lam)-original(r,lam)
        error=float(max(abs(diff.data),default=0.));assert error<1e-10
        checks.append(dict(lambda_per_ms=lam,error=error))
    s.characteristic=characteristic
    return checks


def main(a):
    s=model();z=np.load(a.point);r=z['r'];s.set_Z(z['Z']);out=OUT/'equilibrium_spectra';out.mkdir(parents=True,exist_ok=True)
    name=Path(a.point).stem;checks=cache_characteristic(s,r);roots=[]
    for lam in [.005+.01j,.01+.04j,.01+.1j,.01+.2j,.01+.3j,.01+.5j,.01+.8j]:
        try:ans=refine_root(s,r,lam)
        except Exception as exc:log('root error',lam,repr(exc));continue
        if ans is None:continue
        v,V,err=ans
        if any(abs(v-complex(*q['lambda_per_ms']))<1e-7 for q in roots):continue
        q=dict(lambda_per_ms=[v.real,v.imag],frequency_hz=abs(v.imag)*1000/(2*np.pi),residual=err)
        energy=s.E*s.sizes*abs(V)**2;q['mode_energy_A_B_surround']=[float(energy[s.geo['group_region']==i].sum()/energy.sum()) for i in range(3)]
        np.savez_compressed(out/f'{name}_mode{len(roots)}.npz',v=V,lambda_per_ms=v,r=r,Z=s.Z,D=s.D)
        roots.append(q);log('ROOT',q)
        write(out/f'{name}.json',dict(status='RUNNING',source=a.point,roots=roots,checks=checks))
    result=count(s,r,N=512)
    write(out/f'{name}.json',dict(status='COMPLETE',source=a.point,D=s.D,global_E_hz=s.global_rate(r),roots=roots,checks=checks,count=result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');main(p.parse_args())
