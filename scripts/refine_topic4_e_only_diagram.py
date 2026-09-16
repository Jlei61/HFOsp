#!/usr/bin/env python3
"""Continuation repair for tau guesses; converged contour counts for crowded spectra."""
from trace_topic4_e_only_hopf import hopf
from analyze_topic4_e_only_bifurcation import Family,OUT,read,write
import numpy as np
import time


def main():
    while read(OUT/'fixed_rate_status.json')['status']=='RUNNING':time.sleep(10)
    folder=OUT/'reduced_bifurcation';rows=[];write(OUT/'diagram_refinement_status.json',{'status':'RUNNING'})
    for kind in ['uniform','native_path']:
        base=read(folder/f'hopf_{kind}_tau20.6116_dt0.1.json')
        for target in [10.,42.]:
            tau0=base['tau_ms'];guess=np.array([base['q'],base['frequency_hz']]);derivative=np.array([-.0012,.05])
            times=np.linspace(tau0,target,int(np.ceil(abs(target-tau0)/.2))+1)[1:]
            for tau in times:
                pred=guess+(tau-tau0)*derivative
                try:
                    row=hopf(kind,float(tau),pred,track_delta=1e-5)
                    new=np.array([row['q'],row['frequency_hz']]);derivative=(new-guess)/(tau-tau0);tau0=tau;guess=new
                    rows.append(row)
                except Exception as exc:
                    rows.append({'kind':kind,'tau_ms':float(tau),'target':target,'status':'UNRESOLVED','error':repr(exc)});break
                write(OUT/'diagram_refinement_status.json',{'status':'RUNNING','curve_rows':rows})
    refinements=[]
    for q in [.6,.7,.78,.79]:
        s=Family('uniform').s;r,err,ok=s.solve(q,np.full(200,.45));assert ok
        pre=s.blocks(r,q);checks=read(folder/f'stability_uniform_high_q{q:g}.json')['root_count']['checks'].copy()
        accepted=False
        for length in [3200,6400,12800]:
            theta=np.r_[0,np.geomspace(1e-7,np.pi,length)];angles=[]
            for th in theta:
                z=np.exp(1j*th);sign,_=np.linalg.slogdet(s.characteristic(z,r,q,pre=pre));ref=(z-1)*s.tr/s.dt+1
                angles.append(np.angle(sign*np.exp(-1j*np.angle(ref).sum())))
            phase=np.unwrap(angles);count=-(phase[-1]-phase[0])/np.pi
            checks.append({'grid':length,'count':float(count),'maximum_phase_step':float(np.max(abs(np.diff(phase))))})
            accepted=(abs(checks[-1]['count']-checks[-2]['count'])<1e-6 and checks[-1]['maximum_phase_step']<np.pi/2)
            if accepted:break
        row={'q':q,'status':'PASS' if accepted else 'UNRESOLVED','unstable_roots':round(count) if accepted else None,'checks':checks}
        refinements.append(row);write(folder/f'refined_count_uniform_high_q{q:g}.json',row)
    write(OUT/'diagram_refinement_status.json',{'status':'COMPLETE','curve_rows':rows,'refined_counts':refinements})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'diagram_refinement_status.json',{'status':'FAILED','error':repr(exc)});raise
