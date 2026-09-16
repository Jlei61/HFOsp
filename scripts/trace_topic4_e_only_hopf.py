#!/usr/bin/env python3
"""Track the low-branch complex crossing and its GABA-decay dependence."""
from analyze_topic4_e_only_bifurcation import Family, OUT, read, write
from audit_topic4_e_only_stability import count_unstable
from topic4_e_only_z_tangent import sample_spectrum
from scipy.optimize import root
from scipy.linalg import eigvals
import numpy as np
import time


def hopf(kind,tau,guess,dt=.1,track_delta=.0002):
    f=Family(kind,tau);s=f.s;r0=np.full(200,.00005);cache={}
    def equation(x):
        q,freq=x;r,err,ok=f.solve(q,r0)
        if not ok:raise RuntimeError(('equilibrium',q,err))
        ev=eigvals(s.char_refined(2j*np.pi*freq/1000,r,f.field(q),dt));v=ev[np.argmin(abs(ev))]
        cache['r']=r;return [v.real,v.imag]
    sol=root(equation,guess,tol=1e-9);error=float(max(abs(np.array(equation(sol.x)))));q,freq=map(float,sol.x)
    assert error<1e-7 and freq>.05 and f.minimum<q<1.1,(sol.message,sol.x,error)
    r=cache['r'].copy();tracks=[]
    for dq in [-track_delta,0,track_delta]:
        qq=q+dq;rr,_,ok=f.solve(qq,r);assert ok
        def rootfun(x):
            ev=eigvals(s.char_refined(complex(*x)/1000,rr,f.field(qq),dt));v=ev[np.argmin(abs(ev))]
            return [v.real,v.imag]
        ans=root(rootfun,[0,2*np.pi*freq],tol=1e-9);err=float(max(abs(np.array(rootfun(ans.x)))));assert err<1e-7
        tracks.append({'q':qq,'real_per_s':float(ans.x[0]),'frequency_hz':float(ans.x[1]/2/np.pi),'residual':err})
    row={'kind':kind,'tau_ms':tau,'dt_ms':dt,'q':q,'frequency_hz':freq,'residual':error,
         'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),'track':tracks,
         'transversality_per_s':(tracks[2]['real_per_s']-tracks[0]['real_per_s'])/(2*track_delta),
         'scope':'Complex crossing of the delayed rate equilibrium; native map is discrete, continuous-delay limit checked separately. Criticality/limit-cycle stability not inferred.'}
    if dt==.1 and abs(tau-20.611550480127335)<1e-8:
        row['counts_nearby']=[]
        for dq in [-.0002,.0002]:
            rr,_,ok=f.solve(q+dq,r);assert ok
            row['counts_nearby'].append(dict(q=q+dq,**count_unstable(s,rr,f.field(q+dq))))
    folder=OUT/'reduced_bifurcation';stem=f'hopf_{kind}_tau{tau:g}_dt{dt}'
    write(folder/f'{stem}.json',row);np.savez_compressed(folder/f'{stem}.npz',r=r,q=q,frequency_hz=freq)
    print(kind,tau,dt,q,freq,flush=True);return row


def main():
    # Wait for the earlier mathematical worker rather than exceeding two reduced workers.
    while read(OUT/'stability_status.json')['status']=='RUNNING':time.sleep(10)
    rows=[];base=20.611550480127335
    write(OUT/'hopf_status.json',{'status':'RUNNING','rows':rows})
    for kind,guess in [('uniform',[.842,2.5]),('native_path',[.869,2.5])]:
        first=hopf(kind,base,guess);rows.append(first)
        for tau in [10.,15.,30.,42.]:
            try:rows.append(hopf(kind,tau,[first['q'],first['frequency_hz']]))
            except Exception as exc:
                rows.append({'kind':kind,'tau_ms':tau,'status':'UNRESOLVED','error':repr(exc)})
            write(OUT/'hopf_status.json',{'status':'RUNNING','rows':rows})
        for dt in [.05,.025,None]:
            rows.append(hopf(kind,base,[first['q'],first['frequency_hz']],dt))
    # Saturated C endpoint and intermediate high branch are separate stability questions.
    for q in [.5,.6,.7]:
        f=Family('uniform');s=f.s;r,err,ok=f.solve(q,np.full(200,.45));assert ok
        result={'kind':'uniform','branch':'high','q':q,'tau_ms':s.tau,
                'spectrum':sample_spectrum(s,r,q,(0,4,20,60,120),4),'root_count':count_unstable(s,r,q)}
        write(OUT/'reduced_bifurcation'/f'stability_uniform_high_q{q:g}.json',result)
    write(OUT/'hopf_status.json',{'status':'COMPLETE','rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'hopf_status.json',{'status':'FAILED','error':repr(exc)});raise
