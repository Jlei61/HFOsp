#!/usr/bin/env python3
"""Bounded validation of a derived transfer correction; no biological tuning."""
from topic4_spatial_boundary_common import OUT,OLD,REFERENCE,read,write
from run_topic4_spatial_boundary_rate import Stepper
from topic4_mixed_timescale_rate import MixedTimescaleSystem
from topic4_e_only_z_rate import native_gaba_variance_factor
from scipy.special import ndtr
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import time


def run(mode):
    started=time.time();s=MixedTimescaleSystem(quadrature=33);st=Stepper(system=s);m=s.m;n=s.n
    source=np.load(OLD/'external_input.npz');expected=source['expected_rate_per_ms'];native=np.load(REFERENCE/'trajectory.npz')
    mapping=np.bincount(source['cell_e']*400+native['cell_e'],minlength=n*400).reshape(n,400)/m.count_e[:,None]
    native_z=native['z_field_10ms']@mapping.T;proto=read(REFERENCE/'protocol.json')
    tau=proto['tau_z_ms'];threshold=proto['I_th_EI'];factor=native_gaba_variance_factor(s)
    z=np.ones(n);z_restore=None;frames=13680
    fields=np.empty((frames,2,n),np.float32);zs=np.empty((frames,n),np.float32);targets=np.empty_like(zs);currents=np.empty((frames,6,n),np.float32)
    for step in range(frames*10):
        tm=step*s.dt
        if step in (80000,88000,94000):st.save(OUT/'mixed_timescale_checkpoints'/f'{mode}_t{step//10}ms.npz',absolute_time_ms=tm,z=z)
        if mode=='native_replay':
            index=min(step//100,len(native_z)-1);alpha=(step%100)/100
            z=(1-alpha)*native_z[index]+alpha*native_z[min(index+1,len(native_z)-1)]
        elif tm>=10680:
            if z_restore is None:z_restore=z.copy()
            z=z_restore+min(1.,(tm-10680)/1000)*(1-z_restore)
        raw_var=factor*(m.v_ei@st.r[n:])
        r=st.step(z,expected[step,0].astype(float),expected[step,1].astype(float))
        target=ndtr((threshold-st.c[1])/np.sqrt(np.maximum(raw_var,1e-12)))
        if (step+1)%10==0:
            k=step//10;fields[k]=r.reshape(2,n)*1000;zs[k]=z;targets[k]=target;currents[k]=st.c
        if mode=='autonomous_gaussian' and tm<10680:z+=s.dt/tau*(target-z)
        if step%5000==0:write(OUT/'mixed_timescale_progress'/f'{mode}.json',{'status':'RUNNING','time_ms':tm,'elapsed_s':time.time()-started,
            'mean_Z':float(np.average(z,weights=m.count_e)),'mean_E_hz':float(np.average(r[:n],weights=m.count_e)*1000)})
    folder=OUT/'mixed_timescale';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{mode}.npz',fields_hz=fields,z=zs,z_target=targets,current=currents,count_e=m.count_e,count_i=m.count_i,frame_ms=1.)
    e=np.average(fields[:,0],axis=1,weights=m.count_e);e10=e.reshape(-1,10).mean(1)
    hit=np.flatnonzero(np.convolve((e10>=200).astype(int),np.ones(20,dtype=int),'valid')==20)
    row={'status':'COMPLETE','mode':mode,'seconds':time.time()-started,
        'high_trigger_end_ms':float((hit[0]+20)*10) if len(hit) else None,
        'high_window_E_hz':float(e[10180:10680].mean()),'final_E_hz':float(e[-1000:].mean()),
        'minimum_mean_Z':float(np.average(zs,axis=1,weights=m.count_e).min()),
        'scope':'Unfitted fast-AMPA / quasi-static-GABA transfer correction, 33-point Gaussian integration. Native graph and biological parameters unchanged; rate relaxation and original Gaussian Z target retained.'}
    write(folder/f'{mode}.json',row);return row


def main():
    assert read(OUT/'mixed_timescale_quadrature_final_qa.json')['status']=='PASS'
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,mode) for mode in ('native_replay','autonomous_gaussian')]):
            rows.append(f.result());write(OUT/'mixed_timescale_status.json',{'status':'RUNNING','completed':len(rows),'rows':rows})
    write(OUT/'mixed_timescale_status.json',{'status':'COMPLETE','rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'mixed_timescale_status.json',{'status':'FAILED','error':repr(exc)});raise
