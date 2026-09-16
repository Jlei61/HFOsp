#!/usr/bin/env python3
"""Full delayed-map spectrum on newly computed E-target-only branches."""
from analyze_topic4_e_only_bifurcation import Family, OUT, read, write
from topic4_e_only_z_tangent import sample_spectrum
import numpy as np
import time


def count_unstable(s, r, q, grid_lengths=(800,1600)):
    pre=s.blocks(r,q); checks=[]
    for length in grid_lengths:
        theta=np.r_[0,np.geomspace(1e-7,np.pi,length)]; angles=[]
        for th in theta:
            z=np.exp(1j*th); sign,_=np.linalg.slogdet(s.characteristic(z,r,q,pre=pre))
            reference=(z-1)*s.tr/s.dt+1
            angles.append(np.angle(sign*np.exp(-1j*np.angle(reference).sum())))
        phase=np.unwrap(angles); number=-(phase[-1]-phase[0])/np.pi
        checks.append({'grid':length,'count':float(number),'maximum_phase_step':float(np.max(abs(np.diff(phase))))})
    ok=(abs(checks[-1]['count']-round(checks[-1]['count']))<1e-6 and
        abs(checks[-1]['count']-checks[0]['count'])<1e-6 and checks[-1]['maximum_phase_step']<np.pi/2)
    return {'status':'PASS' if ok else 'UNRESOLVED','unstable_roots':round(checks[-1]['count']) if ok else None,
            'checks':checks,'method':'Numerical argument principle on full-map Schur complement, grid doubling; not interval certified.'}


def main():
    folder=OUT/'reduced_bifurcation';rows=[]
    jobs=[('uniform','low',1.),('uniform','low',.84),('uniform','high',.78),('uniform','high',.79),
          ('native_path','low',1.),('native_path','low',.867),('native_path','high',.665),('native_path','high',.6705)]
    write(OUT/'stability_status.json',{'status':'RUNNING','completed':0,'total':len(jobs)})
    for kind,branch,q in jobs:
        start=time.time();f=Family(kind);s=f.s
        a=np.load(folder/f'{kind}_tau{s.tau:g}_branches.npz');ix=np.flatnonzero(a['branch']==branch)
        ix=ix[np.argmin(abs(a['q'][ix]-q))];r,err,ok=f.solve(q,a['r'][ix]);assert ok,(kind,branch,q,err)
        spectrum=sample_spectrum(s,r,f.field(q),frequencies=(0,4,20,60),k=4)
        count=count_unstable(s,r,f.field(q))
        row={'kind':kind,'branch':branch,'q':q,'tau_ms':s.tau,'fixed_point_residual':err,
             'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),'spectrum':spectrum,
             'root_count':count,'seconds':time.time()-start}
        write(folder/f'stability_{kind}_{branch}_q{q:g}.json',row);rows.append(row)
        write(OUT/'stability_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows})
        print(kind,branch,q,count['unstable_roots'],spectrum['roots'][:2],flush=True)
    write(OUT/'stability_status.json',{'status':'COMPLETE','completed':len(rows),'rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'stability_status.json',{'status':'FAILED','error':repr(exc)});raise
