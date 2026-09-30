"""Pseudo-arclength continuation of the full two-angle spatial rate torus.

Allows the amplitude projection to turn without misidentifying a failed
fixed-amplitude Newton solve as a physical endpoint. Two phase conditions and
one continuation hyperplane replace the old fixed-amplitude constraint.
"""
from rate_invariant_torus import *


def main(a):
    s=RateField();nt,np_=a.nt,a.np;critical=read(PERIODIC_OUT/f'{a.critical}_N128.json')
    T0=critical['T_ms'];lam=complex(*critical['lambda_per_ms']);nu0=abs(lam.imag-round(lam.imag*T0/(2*np.pi))*2*np.pi/T0)
    seed0,seed1=np.load(a.first),np.load(a.second)
    def encode(z):
        r=resample(resample(z['r'],nt,axis=0),np_,axis=1)
        return np.r_[(r*1000).ravel(),np.log(float(z['T'])),float(z['nu'])/nu0,float(z['J'])*1000]
    x0,x1=encode(seed0),encode(seed1);q=resample(seed1['q'],nt,axis=0)
    weight=np.r_[np.full(nt*np_*s.P,10/np.sqrt(nt*np_*s.P)),50.,1.,1.]
    o=Torus(s,nt,np_,a.device);o.precondition_harmonics=a.precondition_harmonics;o.krylov_restart=a.krylov_restart
    dest=PERIODIC_OUT/'tori';ds=a.ds;rows=[];amp=float(seed1['amplitude_hz'])/1000
    for i in range(a.start_index,a.start_index+a.steps):
        tangent=x1-x0;tangent/=np.linalg.norm(tangent*weight)
        for retry in range(6):
            pred=x1+ds*tangent
            r,T,nu,J,err,history=o.solve(pred[:-3].reshape(nt,np_,s.P)/1000,
                np.exp(pred[-3]),pred[-2]*nu0,pred[-1]/1000,q,amp,nu0,
                arc=(pred,tangent,weight),maxiter=14,tol=a.tol)
            if err<a.tol:break
            ds*=.5
        else:
            write(PERIODIC_OUT/f'{a.label}_continuation.json',dict(status='NUMERICAL_STOP',rows=rows,
                failed_index=i,failed_residual=err,ds=ds,meaning='No bifurcation inferred from solver failure.'))
            return
        projection=np.vdot(q,np.fft.fft(r,axis=1)[:,1,:]/np_)/np.vdot(q,q)
        name=f'{a.label}_{i:04d}_N{nt}x{np_}'
        path=dest/(name+'.npz')
        np.savez_compressed(path,r=r,T=T,nu=nu,J=J,q=q,amplitude_hz=projection.real*1000,residual=err,history=history)
        row=dict(status='CONVERGED',J_EE_core=J,T_ms=T,modulation_period_ms=2*np.pi/nu,
            modulation_frequency_per_ms=nu,amplitude_hz=projection.real*1000,N_theta=nt,N_psi=np_,
            residual=err,source=str(path),ds=ds,stability='NOT_COMPUTED',
            method='Pseudo-arclength two-angle full-spatial BVP; no mode truncation in the equations.')
        write(path.with_suffix('.json'),row);rows.append(row)
        write(PERIODIC_OUT/f'{a.label}_continuation.json',dict(status='RUNNING',rows=rows,pid=os.getpid()))
        print('TORUS ARC SAVED',row,flush=True)
        x0=x1;x1=np.r_[(r*1000).ravel(),np.log(T),nu/nu0,J*1000]
        if len(history)<5:ds=min(ds*1.2,a.ds*2)
        if len(history)>7:ds*=.7
    write(PERIODIC_OUT/f'{a.label}_continuation.json',dict(status='BATCH_COMPLETE',rows=rows,
        meaning='Bounded continuation batch; not a complete torus branch or stability inventory.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second')
    p.add_argument('--nt',type=int,default=64);p.add_argument('--np',type=int,default=32)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--ds',type=float,default=.03);p.add_argument('--label',required=True)
    p.add_argument('--device',type=int,default=0);p.add_argument('--critical',default='TR_A_return')
    p.add_argument('--tol',type=float,default=1e-10);p.add_argument('--precondition-harmonics',type=int,default=16)
    p.add_argument('--krylov-restart',type=int,default=120)
    main(p.parse_args())
