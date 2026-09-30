"""Refine the observed high-activity Hopf crossing on the same spatial Z path."""
from audit_and_spectrum import refine
from periodic_zm import PERIODIC_OUT
from model_zm import *
from scipy.optimize import brentq


def main():
    s=ZMSpatialRate();folder=OLD/'g20/D_arclength_upper_focused'
    seeds=[np.load(folder/f'point{i:04d}.npz') for i in [102,104]];records=[];cache=[]
    def solve(D):
        z=min(seeds,key=lambda z:abs(float(z['D'])-D));r,ok,_=s.solve_D(D,z['r']);assert ok
        start=min(cache,key=lambda q:abs(q[0]-D))[1:] if cache else (.0003+.148j,None)
        found=refine(s,r,D,*start,tol=1e-12);assert found is not None
        lam,v,err=found;cache.append((D,lam,v));records.append(dict(D=D,lambda_per_ms=lam,residual=err))
        print('HOPF',D,lam,err,flush=True)
        return lam,r,v,err
    lo,hi=sorted(float(z['D']) for z in seeds);D=brentq(lambda x:solve(x)[0].real,lo,hi,xtol=2e-12)
    lam,r,v,err=solve(D);h=1e-6;slope=(solve(D+h)[0]-solve(D-h)[0])/(2*h)
    energy=s.sizes*abs(v)**2;energy/=energy.sum();reg=s.geo['group_region'];field=np.bincount(s.geo['group_cell'][s.E],weights=energy[s.E],minlength=s.grid*s.grid)
    row=dict(D=D,J_EE_core=1.,global_E_hz=s.global_rate(r),omega_per_ms=lam.imag,frequency_hz=lam.imag*1000/(2*np.pi),
             eigenvalue=lam,characteristic_residual=err,lambda_D=slope,transversality=float(slope.real),
             E_energy=[energy[s.E&(reg==k)].sum() for k in range(3)],I_energy=energy[~s.E].sum(),
             type='Hopf crossing; cubic criticality pending',root_trace=records,
             scope='High-rate equilibrium branch, not the loss of the low-frequency self-limited cycle')
    np.savez_compressed(PERIODIC_OUT/'hopf_high.npz',r=r,D=D,w=lam.imag,q=v,lambda_D=slope,field_energy=field)
    write(PERIODIC_OUT/'hopf_high.json',row);print('HIGH HOPF',clean(row),flush=True)


if __name__=='__main__':main()
