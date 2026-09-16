#!/usr/bin/env python3
"""Track frozen-fast stability along the actual 10-ms recorded native Z field."""
from topic4_e_only_z_rate import EOnlySystem
from analyze_topic4_e_only_bifurcation import OUT,ROOT,write,read
from audit_topic4_e_only_stability import count_unstable
from topic4_e_only_z_tangent import sample_spectrum
from scipy.optimize import root,brentq
from scipy.linalg import eigvals
import numpy as np


def main():
    s=EOnlySystem();a=np.load(ROOT/'results/topic4_sef_hfo/autonomous_z_manual_restore_v1/trajectory.npz');b=np.load(OUT/'external_input.npz')
    ce10=b['cell_e'];ce20=a['cell_e'];mapping=np.bincount(ce10*400+ce20,minlength=100*400).reshape(100,400)/s.m.count_e[:,None]
    fields=a['z_field_10ms']@mapping.T
    def field(tsec):
        idx=int(tsec*100);alpha=tsec*100-idx
        return (1-alpha)*fields[idx]+alpha*fields[min(idx+1,len(fields)-1)]
    r=np.full(200,.00005);lam=np.array([-21.02254,41.27894]);rows=[];bracket=None
    for t in np.arange(0,8.001,.1):
        z=field(float(t));rr,err,ok=s.solve(z,r)
        if not ok:
            rows.append({'time_s':float(t),'status':'EQUILIBRIUM_UNRESOLVED','residual':err});break
        r=rr
        def fun(x):
            M=s.char_refined(complex(*x)/1000,r,z,.1);ev=eigvals(M);v=ev[np.argmin(abs(ev))];return [v.real,v.imag]
        ans=root(fun,lam,tol=1e-9);err=float(max(abs(np.array(fun(ans.x)))))
        if err>=1e-7:
            spectrum=sample_spectrum(s,r,z,(0,2,5,10),4)
            candidates=[v for v in spectrum['roots'] if v['imag_per_s']>=0 and v['full_map_residual']<1e-8]
            nearest=min(candidates,key=lambda v:abs(complex(v['real_per_s'],v['imag_per_s'])-complex(*lam)))
            lam=np.array([nearest['real_per_s'],nearest['imag_per_s']]);err=float(max(abs(np.array(fun(lam)))))
            assert err<1e-7,(t,lam,err)
        else:lam=ans.x
        row={'time_s':float(t),'mean_Z':float(np.average(z,weights=s.m.count_e)),'real_per_s':float(lam[0]),'frequency_hz':float(abs(lam[1])/2/np.pi),'residual':err}
        if rows and rows[-1]['real_per_s']<0<=row['real_per_s'] and bracket is None:bracket=[rows[-1]['time_s'],float(t)]
        rows.append(row);write(OUT/'observed_z_path_status.json',{'status':'RUNNING','rows':rows});write(OUT/'observed_z_path_trace.json',rows)
        print(row,flush=True)
        if bracket is not None:break
    assert bracket is not None,'No crossing bracket found; inspect tracked path'
    # The recorded Z field is not monotone: a coarse bracket can contain loss and
    # reappearance of a branch. Resolve its first crossing on a 1-ms path grid.
    fine=[];r,_,ok=s.solve(field(bracket[0]),np.full(200,.00005));assert ok
    lam=np.array([rows[-2]['real_per_s'],2*np.pi*rows[-2]['frequency_hz']])
    for tt in np.linspace(bracket[0],bracket[1],101):
        zz=field(float(tt));r,err,ok=s.solve(zz,r);assert ok,('fine equilibrium',tt,err)
        def funfine(x):
            ev=eigvals(s.char_refined(complex(*x)/1000,r,zz,.1));v=ev[np.argmin(abs(ev))];return [v.real,v.imag]
        sol=root(funfine,lam,tol=1e-9);err=float(max(abs(np.array(funfine(sol.x)))));assert err<1e-7;lam=sol.x
        fine.append({'time_s':float(tt),'real_per_s':float(lam[0]),'imag_per_s':float(lam[1])});write(OUT/'observed_z_path_fine_trace.json',fine)
        if len(fine)>1 and fine[-2]['real_per_s']<0<=fine[-1]['real_per_s']:break
    assert fine[-1]['real_per_s']>=0
    cache={}
    def alpha(tt):
        zz=field(tt);rr,err,ok=s.solve(zz,r);assert ok,err
        def equation(x):
            ev=eigvals(s.char_refined(complex(*x)/1000,rr,zz,.1));v=ev[np.argmin(abs(ev))];return [v.real,v.imag]
        sol=root(equation,lam,tol=1e-9);err=float(max(abs(np.array(equation(sol.x)))));assert err<1e-7
        cache['lambda']=sol.x;cache['error']=err;return float(sol.x[0])
    t=float(brentq(alpha,fine[-2]['time_s'],fine[-1]['time_s'],xtol=1e-10));alpha(t);freq=float(cache['lambda'][1]/2/np.pi);err=cache['error']
    z=field(t);rr,_,ok=s.solve(z,r);assert ok;counts=[]
    for offset in [-.001,.001]:
        zz=field(t+offset);r2,_,ok=s.solve(zz,rr);assert ok
        counts.append(dict(time_s=t+offset,**count_unstable(s,r2,zz)))
    result={'status':'COMPLETE','crossing_time_parameter_s':t,'frequency_hz':freq,'mean_Z':float(np.average(z,weights=s.m.count_e)),
            'equilibrium_E_hz':float(np.average(rr[:100],weights=s.m.count_e)*1000),'residual':err,'counts':counts,'rows':rows,
            'scope':'Frozen fast reduced system along native recorded Z field, with constant reference external mean. Time labels select a Z profile; this is not a nonautonomous/native SNN bifurcation proof and OU is not frozen at its instantaneous value.',
            'field_source':'Same neuron-weighted 20x20-to-10x10 projection and 10-ms interpolation as the native-Z replay validation.'}
    np.savez_compressed(OUT/'reduced_bifurcation/observed_path_hopf.npz',r=rr,z=z,time_s=t,frequency_hz=freq)
    write(OUT/'observed_z_path_status.json',result);print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'observed_z_path_status.json',{'status':'FAILED','error':repr(exc)});raise
