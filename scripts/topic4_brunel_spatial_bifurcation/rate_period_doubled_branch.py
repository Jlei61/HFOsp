"""Switch from an antiperiodic null mode to an independently solved 2T cycle.

A fixed nonzero projection on the antiperiodic mode prevents convergence to
the repeated parent orbit. Branch side is not by itself a stability verdict.
"""
from rate_periodic import *


def main():
    p=argparse.ArgumentParser();p.add_argument('critical_json');p.add_argument('mode')
    p.add_argument('--N',type=int,default=512,help='Parent temporal resolution; new period uses 2N')
    p.add_argument('--amplitudes',type=float,nargs='+',default=[.1,.2,.4,.8])
    p.add_argument('--device',type=int,default=0);p.add_argument('--label',default='PDchild')
    p.add_argument('--derivative-chunk-size',type=int,default=0)
    p.add_argument('--krylov-restart',type=int,default=160)
    p.add_argument('--linear-normalize',action='store_true')
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--stream-harmonics',action='store_true',help='Apply all harmonics in blocks without storing the full GPU bank')
    p.add_argument('--harmonic-chunk-size',type=int,default=64)
    p.add_argument('--tol',type=float,default=1e-9)
    p.add_argument('--from-orbit',help='Continue an already solved child, with its measured mode projection')
    a=p.parse_args();s=RateField();crit=read(Path(a.critical_json));z=np.load(crit['orbit'])
    r0=resample(z['r'],a.N,axis=0);T0=2*float(z['T']);J0=float(z['J'])
    mode=np.load(a.mode)['u'];u=resample(np.r_[mode,-mode],2*a.N,axis=0)
    u/=np.max(abs(u));base=np.r_[r0,r0];c=np.r_[u.ravel()/np.sum(u*u),0.,0.]
    o=Periodic(s,2*a.N,a.device);o.low_memory=True;o.krylov_restart=a.krylov_restart
    o.derivative_chunk_size=a.derivative_chunk_size;o.normalize_linear_rhs=a.linear_normalize
    o.host_krylov=a.host_krylov
    o.stream_harmonics=a.stream_harmonics;o.harmonic_chunk_size=a.harmonic_chunk_size
    o.linear_target_aware=a.tol<1e-9
    r=base;T=T0;J=J0;last_amp=0;rows=[]
    if a.from_orbit:
        seed=np.load(a.from_orbit);r=resample(seed['r'],2*a.N,axis=0);T=float(seed['T']);J=float(seed['J'])
        last_amp=float(c@np.r_[((r-base)*1000).ravel(),0.,0.])
        prior=PERIODIC_OUT/f'{a.label}_branch_N{2*a.N}.json'
        if prior.exists():rows=read(prior)
    for amp in a.amplitudes:
        r=r+(amp-last_amp)*u/1000
        pred=np.r_[(r*1000).ravel(),np.log(T),J*1000]
        target=amp+c@np.r_[(base*1000).ravel(),0.,0.]
        pred+=c*(target-c@pred)/(c@c)
        r=pred[:-2].reshape(2*a.N,s.P)/1000
        r,T,J,err,hist=o.solve(r,T,J,arc=(pred,c,np.ones_like(c)),maxiter=26,tol=a.tol)
        if err>=2e-8 or (a.tol<1e-9 and err>a.tol*1.01):
            # Retain the failed iterate for diagnosis without classifying it
            # as a periodic orbit or adding it to the converged child branch.
            save_orbit(s,r,T,J,err,hist,f'{a.label}_failed_a{amp:.5f}_N{2*a.N}')
        assert err<2e-8,(amp,J,err)
        if a.tol<1e-9:assert err<=a.tol*1.01, ('Fine child did not meet requested tolerance',amp,J,err)
        halfdef=float(np.linalg.norm(r[:a.N]-r[a.N:])/np.linalg.norm(r))
        assert halfdef>1e-7,'Converged to a repeated parent orbit'
        path=save_orbit(s,r,T,J,err,hist,f'{a.label}_a{amp:.5f}_N{2*a.N}')
        row=dict(amplitude_hz=amp,J_EE_core=J,J_shift=J-J0,J_shift_over_amplitude_squared=(J-J0)/amp**2,
                 T_ms=T,parent_T_ms=T0/2,half_period_relative_mismatch=halfdef,orbit=str(path),
                 minimum_group_rate_hz=float(r.min()*1000),
                 first_harmonic_relative_norm=float(np.linalg.norm(np.fft.rfft(r,axis=0)[1])/np.linalg.norm(np.fft.rfft(r,axis=0)[1:])))
        rows.append(row);write(PERIODIC_OUT/f'{a.label}_branch_N{2*a.N}.json',rows)
        print('DOUBLED BRANCH',row,flush=True);last_amp=amp


if __name__=='__main__':main()
