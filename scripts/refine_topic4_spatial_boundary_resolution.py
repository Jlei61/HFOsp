#!/usr/bin/env python3
"""Bounded 20x20 resolution sensitivity of the unchanged Z/rate equations."""
from topic4_spatial_boundary_common import OUT, OLD, REFERENCE, read, write
from run_topic4_spatial_boundary_rate import Stepper
from topic4_e_only_z_rate import native_gaba_variance_factor
from validate_topic4_fixed_rate_base import setup, make_external_drive, spatial_cell_index
from params import compute_nu_theta
from checkpoint import load
from scipy.special import ndtr
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import time
import os


def prepare_input():
    start=time.time();seed=9108401;s,tr,fr,identity=setup(seed);p=s.params;ne=s.net['NE'];ni=s.net['NI'];n=ne+ni
    ce=spatial_cell_index(s.positions_e,n_grid=20,sheet_l_mm=p.L)
    ci=spatial_cell_index(s.positions_i,n_grid=20,sheet_l_mm=p.L)
    count_e=np.bincount(ce,minlength=400);count_i=np.bincount(ci,minlength=400)
    old=np.load(OLD/'external_input.npz');ce10=old['cell_e'];old_expected=old['expected_rate_per_ms']
    old_xi=old['xi'];count_e10=old['count_e']
    aggregation=np.zeros((100,400));aggregation[ce10,ce]=1
    rng=np.random.default_rng(seed);rng.choice(ne,size=80,replace=False);rng.choice(ni,size=20,replace=False)
    drive=make_external_drive(s,tr['spatial_ou'],seed);nu=p.nu_ext_ratio*compute_nu_theta(p)[0]
    aa=np.exp(-p.dt/p.tau_n);bb=p.sigma_n*.001*np.sqrt(p.tau_n/2)*np.sqrt(1-aa*aa);xi=0.
    expected=np.empty((136800,2,400),np.float32);max_error=0.
    for step in range(len(expected)):
        if step in [80000,94000,98000,101800,106800]:
            ck=load(OLD/'checkpoints'/f't{step//10}ms.npz')
            assert xi == ck['xi'] and rng.bit_generator.state == ck['rng_state']
            assert drive._rng.bit_generator.state == ck['external_drive']['rng_state']
            del ck
        xi=aa*xi+bb*rng.standard_normal();global_rate=max(0.,nu+xi)
        nu_vec=np.full(n,global_rate);nu_vec[:ne]=np.maximum(nu_vec[:ne]+drive.step(step*p.dt),0.)
        rng.poisson(nu_vec*p.dt,size=n)  # Consume the exact native private-input RNG stream.
        sums=np.bincount(ce,weights=nu_vec[:ne],minlength=400)
        expected[step,0]=sums/count_e;expected[step,1]=global_rate
        projected=aggregation@sums/count_e10
        error=float(np.max(abs(projected-old_expected[step,0])));max_error=max(error,max_error)
        assert error<5e-7 and xi==old_xi[step]
        if step%10000==0:write(OUT/'input20_status.json',{'status':'RUNNING','time_ms':step*p.dt,'elapsed_s':time.time()-start,'max_10cell_projection_error':max_error})
    np.savez_compressed(OUT/'external_input20.npz',expected_rate_per_ms=expected,count_e=count_e,count_i=count_i,cell_e=ce,cell_i=ci,dt_ms=p.dt)
    write(OUT/'input20_status.json',{'status':'COMPLETE','elapsed_s':time.time()-start,
        'checkpoint_RNG_QA':'All five checkpoints exact','max_10cell_projection_error':max_error,'frozen_identity':identity})


def run(mode):
    start=time.time();st=Stepper(grid=20);s=st.s;m=st.m;n=st.n;name=f'grid20_{mode}'
    source=np.load(OUT/'external_input20.npz');expected=source['expected_rate_per_ms']
    assert np.array_equal(source['count_e'],m.count_e) and np.array_equal(source['count_i'],m.count_i)
    native=np.load(REFERENCE/'trajectory.npz');z_native=native['z_field_10ms'];proto=read(REFERENCE/'protocol.json')
    tau=proto['tau_z_ms'];threshold=proto['I_th_EI'];factor=native_gaba_variance_factor(s);z=np.ones(n);z_restore=None
    frames=13680;fields=np.empty((frames,2,n),np.float32);zs=np.empty((frames,n),np.float32)
    targets=np.empty_like(zs);currents=np.empty((frames,6,n),np.float32)
    for step in range(frames*10):
        tm=step*s.dt
        if mode=='native_replay':
            index=min(step//100,len(z_native)-1);alpha=(step%100)/100
            z=(1-alpha)*z_native[index]+alpha*z_native[min(index+1,len(z_native)-1)]
        elif tm>=10680:
            if z_restore is None:z_restore=z.copy()
            z=z_restore+min(1.,(tm-10680)/1000)*(1-z_restore)
        raw_variance=factor*(m.v_ei@st.r[n:])
        r=st.step(z,expected[step,0].astype(float),expected[step,1].astype(float))
        target=ndtr((threshold-st.c[1])/np.sqrt(np.maximum(raw_variance,1e-12)))
        if (step+1)%10==0:
            k=step//10;fields[k]=r.reshape(2,n)*1000;zs[k]=z;targets[k]=target;currents[k]=st.c
        if mode=='autonomous_gaussian' and tm<10680:z+=s.dt/tau*(target-z)
        if (step+1)%10000==0:write(OUT/'resolution_progress'/f'{name}.json',{'status':'RUNNING','time_ms':(step+1)*s.dt,'elapsed_s':time.time()-start})
    folder=OUT/'resolution';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{name}.npz',fields_hz=fields,z=zs,z_target=targets,current=currents,
        count_e=m.count_e,count_i=m.count_i,frame_ms=1.)
    e=np.average(fields[:,0],axis=1,weights=m.count_e);e10=e.reshape(-1,10).mean(1)
    hit=np.flatnonzero(np.convolve((e10>=200).astype(int),np.ones(20,dtype=int),'valid')==20)
    row={'status':'COMPLETE','name':name,'seconds':time.time()-start,
        'high_trigger_end_ms':float((hit[0]+20)*10) if len(hit) else None,
        'high_window_E_hz':float(e[10180:10680].mean()),'final_E_hz':float(e[-1000:].mean()),
        'minimum_Z':float(np.average(zs,axis=1,weights=m.count_e).min()),
        'change':'Spatial reduction 10x10 to 20x20; exact native input reprojected, same biological equations, thresholds and time constants.'}
    write(folder/f'{name}.json',row);write(OUT/'resolution_progress'/f'{name}.json',row);return row


def main():
    while True:
        f=OUT/'rate_batch_status.json';status=read(f) if f.exists() else {}
        if status.get('status')=='COMPLETE':break
        if status.get('status')=='FAILED':raise RuntimeError(status)
        os.kill(read(OUT/'rate_batch_process.json')['pid'],0);time.sleep(10)
    prepare_input();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,mode) for mode in ('native_replay','autonomous_gaussian')]):
            rows.append(future.result());write(OUT/'resolution_status.json',{'status':'RUNNING','completed':len(rows),'rows':rows})
    write(OUT/'resolution_status.json',{'status':'COMPLETE','rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'resolution_status.json',{'status':'FAILED','error':repr(exc)});raise
