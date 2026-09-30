"""Refine a torus crossing by a periodic BVP and a complex Floquet exponent."""
from rate_floquet_spectral import *
from scipy.optimize import brentq


def main():
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',default='TR_A_B');p.add_argument('--N',type=int,default=64)
    p.add_argument('--core',default='B');p.add_argument('--device',type=int,default=0)
    p.add_argument('--orbit-tol',type=float,default=2e-8);p.add_argument('--eigen-tol',type=float,default=2e-9)
    p.add_argument('--root-tol',type=float,default=1e-9);p.add_argument('--slope-step',type=float)
    p.add_argument('--gain-order',type=int,choices=[2,4],default=2);p.add_argument('--gain-step',type=float,default=1e-5)
    p.add_argument('--seed-mode',help='Previously converged complex Floquet eigenfunction; preserves the tracked mode across precision checks')
    p.add_argument('--cache-orbits',nargs='*',default=[],help='Additional same-branch orbit seeds; root bracket remains the first two positional files')
    a=p.parse_args();s=RateField();cache=[]
    for f in [a.first,a.second]:
        z=np.load(f);cache.append((float(z['J']),resample(z['r'],a.N,axis=0),float(z['T'])))
    for f in a.cache_orbits:
        z=np.load(f);cache.append((float(z['J']),resample(z['r'],a.N,axis=0),float(z['T'])))
    o=Periodic(s,a.N,a.device);hopf=np.load(RATE_OUT/f'hopf_{a.core}.npz');v=np.tile(hopf['vector'],(a.N,1));lam=.0001+1j*float(hopf['omega'])
    seeds=sorted((PERIODIC_OUT/'spectral_floquet').glob(f'{Path(a.first).stem}_{a.core}_N*.npz'))
    if a.seed_mode or seeds:
        seed=np.load(a.seed_mode or seeds[-1]);v=resample(seed['u'],a.N,axis=0);lam=complex(seed['lam'])
    modecache=[]
    def evaluate(J):
        nonlocal v,lam
        _,r,T=min(cache,key=lambda q:abs(q[0]-J));r,T,J,err,history=o.solve(r,T,J,tol=a.orbit_tol);assert err<a.orbit_tol
        path=save_orbit(s,r,T,J,err,history,f'{a.label}_eval_J{J:.13f}_N{a.N}');cache.append((J,r,T))
        if modecache:
            _,lam,v=min(modecache,key=lambda q:abs(q[0]-J))
        f=SpectralFloquet(s,path,a.N,a.device,a.gain_order,a.gain_step);lam,v,hist=f.refine(lam,v,tol=a.eigen_tol);assert hist[-1]<a.eigen_tol
        modecache.append((J,lam,v.copy()))
        np.savez_compressed(PERIODIC_OUT/f'{a.label}_evalmode_J{J:.13f}_N{a.N}.npz',u=v,lam=lam,J=J,T=T)
        print('TORUS ROOT',J,lam,flush=True);return lam.real,path,T,err,hist[-1]
    ja,jb=sorted([cache[0][0],cache[1][0]]);J=brentq(lambda j:evaluate(j)[0],ja,jb,xtol=a.root_tol,rtol=max(1e-14,a.root_tol*.1))
    _,path,T,err,eigerr=evaluate(J);mult=np.exp(lam*T)
    slope=None
    if a.slope_step:
        lm=evaluate(J-a.slope_step)[0];lp=evaluate(J+a.slope_step)[0]
        slope=dict(step=a.slope_step,real_exponents=[lm,lp],derivative_per_ms_per_J=(lp-lm)/(2*a.slope_step))
        _,path,T,err,eigerr=evaluate(J);mult=np.exp(lam*T)
    assert abs(mult-1)>1e-4 and abs(mult+1)>1e-4,'neutral or real multiplier is not a torus bifurcation'
    np.savez_compressed(PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz',u=v,lam=lam,J=J,T=T)
    row=dict(label=a.label,type='torus bifurcation of periodic orbit',J_EE_core=J,T_ms=T,lambda_per_ms=lam,multiplier=mult,
        distance_from_plus_one=abs(mult-1),distance_from_minus_one=abs(mult+1),orbit=str(path),N=a.N,
        orbit_residual_hz=err,eigen_residual=eigerr,criticality='NOT_COMPUTED',
        orbit_tolerance=a.orbit_tol,eigen_tolerance=a.eigen_tol,root_tolerance=a.root_tol,transversal_slope=slope,
        local_gain_order=a.gain_order,local_gain_step=a.gain_step,
        mode_seed=a.seed_mode,
        interpretation='A non-real conjugate pair crosses the unit circle away from +1 and -1; torus criticality requires additional normal-form analysis')
    write(PERIODIC_OUT/f'{a.label}_N{a.N}.json',row);print('TORUS',row,flush=True)


if __name__=='__main__':main()
