"""Independent original-LIF calibration parity and rate-vector-field tangents."""
from common import *
from model import RateSystem
from calibrate_transfer import CODE
import cupy as cp

def main():
    cp.cuda.Device(0).use();p=read(KIN/'coarse_40/prepared.json')['params'];rng=np.random.default_rng(519711)
    results=[]
    for pop in ['E','I']:
        R=32;theta=np.array([14.5,18.,18.]);drive=np.array([1.,-3.,30.]);N=R*3;steps=1200
        ext=rng.poisson(1.317*.1,size=(steps,N)).astype(np.uint32);q=np.zeros(N);cur=q.copy();v=np.full(N,11.);ref=np.zeros(N,np.int32);count=np.zeros(3,np.uint32)
        dq=cp.asarray(q);dc=cp.asarray(cur);dv=cp.asarray(v);dr=cp.asarray(ref);dn=cp.asarray(count);de=cp.asarray(ext);dt=cp.asarray(theta);dd=cp.asarray(drive)
        tm=p[f'tau_m_{pop}'];ar=np.exp(-DT/p['tau_r_AMPA']);ad=np.exp(-DT/p['tau_d_AMPA']);am=np.exp(-DT/tm);jump=tm/p['tau_r_AMPA']*p[f'J_ext_{pop}'];nr=round(p[f'tau_ref_{pop}']/DT)
        kernel=cp.RawKernel(CODE,'calibrate',options=('--fmad=false',))
        for k in range(steps):
            q=ar*q+jump*ext[k];cur=ad*cur+(1-ad)*q;ref=np.maximum(0,ref-1)
            v=np.where(ref==0,am*v+(1-am)*(cur+np.repeat(drive,R)),11.)
            hit=(ref==0)&(v>=np.repeat(theta,R));v[hit]=11.;ref[hit]=nr;count+=hit.reshape(3,R).sum(1).astype(np.uint32)
            kernel((1,),(128,),(de,dq,dc,dv,dr,dn,dt,dd,np.int32(N),np.int32(R),np.int32(k),np.int32(1),
                *[np.float64(x) for x in [ar,ad,am,jump]],np.int32(nr)))
        err=max(float(np.max(abs(a-b.get()))) for a,b in [(q,dq),(cur,dc),(v,dv)])
        assert err<1e-10 and np.array_equal(ref,dr.get()) and np.array_equal(count,dn.get())
        results.append(dict(population=pop,maximum_state_error=err,spike_counts=count))
    s=RateSystem();r=np.full(s.P,.005);x=s.equilibrium_state(r,.8);d=rng.normal(size=x.shape);da=rng.normal(size=s.P);dg=rng.normal(size=s.P)
    a,b=s.coupling(.8);v=s.jvp(x,d,da,dg);eps=1e-5
    finite=(s.rhs(x+eps*d,a@r+eps*da,b@r+eps*dg)-s.rhs(x-eps*d,a@r-eps*da,b@r-eps*dg))/(2*eps)
    err=np.max(abs(v-finite))/max(1.,np.max(abs(v)));assert err<1e-7
    lam=.003+.02j;analytic=s.characteristic(lam,r,.8,True)
    finite=(s.characteristic(lam+1e-6,r,.8)-s.characteristic(lam-1e-6,r,.8))/(2e-6)
    delta=analytic-finite;charerr=np.max(abs(delta.data))/max(1.,np.max(abs(analytic.data)));assert charerr<1e-6
    identity=s.characteristic(0.,r,.8)+s.equilibrium_jacobian(r,.8);assert np.max(abs(identity.data))<1e-10
    write(BASE/'implementation_check.json',dict(status='PASS',original_calibration_parity=results,rate_rhs_jvp_relative_error=err,
        exact_delay_characteristic_derivative_relative_error=charerr,stationary_characteristic_identity_max_error=np.max(abs(identity.data)),
        rate_groups=s.P,dynamic_variables_without_delay_history=8*s.P,
        scope='Executable continuous rate vector field and analytic tangent checked; spatial dynamics correspondence is a separate requirement'))
    print('PASS',s.P,err,charerr,flush=True)

if __name__=='__main__':main()
