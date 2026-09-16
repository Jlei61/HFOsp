#!/usr/bin/env python3
"""Locate the first tracked oscillatory instability; validate the full tangent."""
from analyze_topic4_prescribed_z_phase import *
from scipy.optimize import root,brentq
from scipy.linalg import eigvals


def tangent_qa(s,r,z):
    t=EOnlyTangent(s,r,z);m=s.m;n=s.n
    h=np.broadcast_to(r.reshape(2,1,n),(2,t.D,n)).copy();g=t.B*t.drive(h)/(1-t.ar);c=g.copy()
    base=np.r_[r,g.ravel(),c.ravel(),h.ravel()]
    def mapping(x):
        rr,gg,cc,hh=t.unpack(x);ng=t.ar*gg+t.B*t.drive(hh);nc=t.ad*cc+(1-t.ad)*ng
        ex=s.moments(rr,z)[1];inh=s.moments(rr,z)[2]
        external=np.r_[np.full(n,m.tau_mem_e_ms*s.ga*m.j_ext_e_mv*m.nu_ext_per_ms),np.full(n,m.tau_mem_i_ms*s.ga*m.j_ext_i_mv*m.nu_ext_per_ms)]
        nr=rr+t.alpha*(s.phi(t.signed(nc)+external,ex,inh)-rr)
        nh=np.empty_like(hh);nh[:,0]=rr.reshape(2,n);nh[:,1:]=hh[:,:-1]
        return np.r_[nr,ng.ravel(),nc.ravel(),nh.ravel()]
    eqerr=float(np.max(abs(mapping(base)-base)));assert eqerr<1e-8,eqerr
    rng=np.random.default_rng(814);direction=(abs(base)+1e-12)*rng.normal(size=len(base));expected=t.action(direction);errors=[]
    for eps in (1e-3,5e-4):
        observed=(mapping(base+eps*direction)-mapping(base-eps*direction))/(2*eps)
        er=float(np.linalg.norm(observed[:200]-expected[:200])/max(np.linalg.norm(expected[:200]),1e-20))
        errors.append({'relative_perturbation':eps,'rate_block_relative_error':er})
    assert errors[-1]['rate_block_relative_error']<1e-3,errors
    return {'equilibrium_map_error':eqerr,'directional_checks':errors,'dimension':t.size}


def main():
    s=MixedTimescaleSystem(quadrature=33);_,_,fields=source();a=np.load(OUT/'branches.npz');r=a['low_0']
    seed=read(OUT/'spectrum_low_0.json')['spectrum']['roots'];v=max(seed,key=lambda d:d['real_per_s']);lam=np.array([v['real_per_s'],abs(v['imag_per_s'])]);rows=[];bracket=None
    write(OUT/'full_tangent_qa.json',{'status':'PASS','low':tangent_qa(s,r,field_at(fields,0)),
        'high':tangent_qa(s,a['high_10500'],field_at(fields,10500))})
    for tm in np.arange(0,2201,25):
        z=field_at(fields,float(tm));r,err,ok=s.solve(z,r);assert ok,(tm,err)
        pre=s.blocks(r,z)
        def equation(x):
            ev=eigvals(s.characteristic(np.exp(complex(*x)/1000*s.dt),r,z,pre=pre));v=ev[np.argmin(abs(ev))]
            return [v.real,v.imag]
        ans=root(equation,lam,tol=1e-9);err=float(max(abs(np.array(equation(ans.x)))));assert err<1e-7,(tm,err)
        lam=ans.x;rows.append({'time_ms':float(tm),'real_per_s':float(lam[0]),'imag_per_s':float(lam[1]),'residual':err})
        write(OUT/'first_instability_progress.json',{'status':'RUNNING','rows':rows})
        if len(rows)>1 and rows[-2]['real_per_s']<0<=rows[-1]['real_per_s']:
            bracket=(rows[-2]['time_ms'],float(tm));break
    if bracket is None:
        write(OUT/'first_instability.json',{'status':'NO_TRACKED_CROSSING','rows':rows});return
    # Resolve the earliest sign change inside the coarse bracket; Z itself need
    # not be monotone, so only this explicit path interval is interpreted.
    fine=[];guess=r.copy()
    for tm in np.linspace(*bracket,51):
        z=field_at(fields,tm);rr,err,ok=s.solve(z,guess);assert ok;guess=rr;pre=s.blocks(rr,z)
        def eq(x):
            ev=eigvals(s.characteristic(np.exp(complex(*x)/1000*s.dt),rr,z,pre=pre));v=ev[np.argmin(abs(ev))];return [v.real,v.imag]
        ans=root(eq,lam,tol=1e-9);err=float(max(abs(np.array(eq(ans.x)))));assert err<1e-7;lam=ans.x
        fine.append({'time_ms':float(tm),'real_per_s':float(lam[0])})
        if len(fine)>1 and fine[-2]['real_per_s']<0<=fine[-1]['real_per_s']:break
    cache={}
    def realpart(tm):
        z=field_at(fields,tm);rr,err,ok=s.solve(z,guess);assert ok;pre=s.blocks(rr,z)
        def eq(x):
            ev=eigvals(s.characteristic(np.exp(complex(*x)/1000*s.dt),rr,z,pre=pre));v=ev[np.argmin(abs(ev))];return [v.real,v.imag]
        ans=root(eq,lam,tol=1e-9);err=float(max(abs(np.array(eq(ans.x)))));assert err<1e-7
        cache.update(r=rr,z=z,lam=ans.x,error=err);return float(ans.x[0])
    tm=brentq(realpart,fine[-2]['time_ms'],fine[-1]['time_ms'],xtol=1e-8);realpart(tm)
    r=cache['r'];z=cache['z'];frequency=float(abs(cache['lam'][1])/2/np.pi)
    counts=[]
    for offset in (-1.,1.):
        zz=field_at(fields,tm+offset);rr,err,ok=s.solve(zz,r);assert ok
        counts.append({'time_ms':tm+offset,**count_unstable(s,rr,zz)})
    slope=(realpart(tm+.1)-realpart(tm-.1))/.2;realpart(tm)
    result={'status':'COMPLETE','time_parameter_ms':tm,'frequency_hz':frequency,'E_equilibrium_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),
        'mean_Z':float(np.average(z,weights=s.m.count_e)),'counts':counts,'real_part_slope_per_s_per_path_ms':float(slope),'rows':rows,'fine_trace':fine,
        'residual':cache['error'],'classification':'Oscillatory linear stability crossing; Hopf-type candidate if a single pair changes stability. Nonlinear criticality/periodic-orbit validation not established.',
        'scope':'Frozen constant-background corrected rate only. Early low-state crossing is not proof of the later native/OU-driven sustained transition.'}
    np.savez_compressed(OUT/'first_instability.npz',r=r,z=z,lambda_per_s=cache['lam'],time_parameter_ms=tm)
    write(OUT/'first_instability.json',result)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'instability_failure.json',{'error':repr(exc)});raise
