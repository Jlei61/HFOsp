#!/usr/bin/env python3
"""Matched mean-Z control isolates the spatial field from average inhibition."""
from analyze_topic4_e_only_bifurcation import Family,OUT,read,write
from topic4_e_only_z_tangent import sample_spectrum
import numpy as np


def main():
    q=.665;f=Family('uniform');s=f.s;r,err,ok=s.solve(q,np.full(200,.45));assert ok
    spectrum=sample_spectrum(s,r,q,(0,4,20,60),4);pre=s.blocks(r,q);checks=[]
    for length in [1600,3200]:
        angles=[]
        for theta in np.r_[0,np.geomspace(1e-7,np.pi,length)]:
            z=np.exp(1j*theta);sign,_=np.linalg.slogdet(s.characteristic(z,r,q,pre=pre));reference=(z-1)*s.tr/s.dt+1
            angles.append(np.angle(sign*np.exp(-1j*np.angle(reference).sum())))
        phase=np.unwrap(angles);count=-(phase[-1]-phase[0])/np.pi
        checks.append({'grid':length,'count':float(count),'maximum_phase_step':float(np.max(abs(np.diff(phase))))})
    assert abs(checks[-1]['count']-checks[-2]['count'])<1e-6 and checks[-1]['maximum_phase_step']<np.pi/2,checks
    row={'kind':'uniform','branch':'high','q':q,'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),'fixed_point_residual':err,
         'spectrum':spectrum,'root_count':{'status':'PASS','unstable_roots':round(count),'checks':checks}}
    write(OUT/'reduced_bifurcation/stability_uniform_high_q0.665.json',row)
    other=read(OUT/'reduced_bifurcation/stability_native_path_high_q0.665.json')
    write(OUT/'same_mean_z_comparison.json',{'status':'COMPLETE','matched_mean_Z':q,'uniform':row,'spatial_field':other,
          'scope':'Same mean E-target Z and all remaining parameters; each field has its own solved high equilibrium. Tests field distribution, not only a change in mean.'})


if __name__=='__main__':main()
