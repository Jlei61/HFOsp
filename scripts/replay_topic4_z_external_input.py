#!/usr/bin/env python3
"""Replay native OU and Poisson inputs without computing recurrent spikes."""
from validate_topic4_fixed_rate_base import ROOT, setup, write, read, make_external_drive, spatial_cell_index
from params import compute_nu_theta
import numpy as np
import time

OUT = ROOT / 'results/topic4_sef_hfo/z_transition_bifurcation_audit_v1'


def main():
    started=time.time(); seed=9108401
    s,tr,fr,identity=setup(seed);p=s.params;ne=s.net['NE'];ni=s.net['NI'];n=ne+ni
    ce=spatial_cell_index(s.positions_e,n_grid=10,sheet_l_mm=p.L)
    ci=spatial_cell_index(s.positions_i,n_grid=10,sheet_l_mm=p.L)
    count_e=np.bincount(ce,minlength=100);count_i=np.bincount(ci,minlength=100)
    rng=np.random.default_rng(seed);rng.choice(ne,size=80,replace=False);rng.choice(ni,size=20,replace=False)
    drive=make_external_drive(s,tr['spatial_ou'],seed)
    nu=p.nu_ext_ratio*compute_nu_theta(p)[0]
    aa=np.exp(-p.dt/p.tau_n);bb=p.sigma_n*.001*np.sqrt(p.tau_n/2)*np.sqrt(1-aa*aa);xi=0.
    steps=136800;expected=np.empty((steps,2,100),np.float32);sampled=np.empty((steps,2,100),np.uint16)
    scalar=np.empty(steps,np.float64);prefix=[]
    for step in range(steps):
        tm=step*p.dt
        if step in [80000,94000,98000,101800,106800]:
            prefix.append({'time_ms':tm,'xi':float(xi),'rng_state':rng.bit_generator.state,
                           'spatial_rng_state':drive._rng.bit_generator.state})
        xi=aa*xi+bb*rng.standard_normal();global_rate=max(0.,nu+xi)
        nu_vec=np.full(n,global_rate);nu_vec[:ne]=np.maximum(nu_vec[:ne]+drive.step(tm),0.)
        ext=rng.poisson(nu_vec*p.dt,size=n)
        expected[step,0]=np.bincount(ce,weights=nu_vec[:ne],minlength=100)/count_e
        expected[step,1]=global_rate
        sampled[step,0]=np.bincount(ce,weights=ext[:ne],minlength=100).astype(np.uint16)
        sampled[step,1]=np.bincount(ci,weights=ext[ne:],minlength=100).astype(np.uint16)
        scalar[step]=xi
        if step%10000==0:
            write(OUT/'external_input_status.json',{'status':'RUNNING','time_ms':tm,'elapsed_s':time.time()-started})
    np.savez_compressed(OUT/'external_input.npz',expected_rate_per_ms=expected,poisson_count=sampled,
                        xi=scalar,count_e=count_e,count_i=count_i,cell_e=ce,cell_i=ci,dt_ms=p.dt)
    write(OUT/'external_input_prefix.json',prefix)
    write(OUT/'external_input_status.json',{'status':'COMPLETE_PENDING_CHECKPOINT_RNG_QA','duration_ms':steps*p.dt,
          'seconds':time.time()-started,'frozen_identity':identity,
          'scope':'Exact native input innovations and Poisson sampling, independent of recurrent SNN activity; expected cell means saved float32'})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'external_input_status.json',{'status':'FAILED','error':repr(exc)})
        raise
