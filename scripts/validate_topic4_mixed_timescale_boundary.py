#!/usr/bin/env python3
"""Two pre-authorized frozen-field continuations of the unfitted closure."""
from topic4_spatial_boundary_common import OUT, OLD, read, write, checkpoint_path, observables
from run_topic4_spatial_boundary_rate import Stepper
from topic4_mixed_timescale_rate import MixedTimescaleSystem
from validate_topic4_fixed_rate_base import setup, make_external_drive
from checkpoint import load, restore_external_drive
from params import compute_nu_theta
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import time


def prepare_input():
    started=time.time(); s,tr,_,identity=setup(9108401); p=s.params
    ne,ni=s.net['NE'],s.net['NI']; n=ne+ni
    ck=load(checkpoint_path(9400)); rng=np.random.default_rng(); rng.bit_generator.state=ck['rng_state']
    xi=ck['xi']; drive=make_external_drive(s,tr['spatial_ou'],9108401); restore_external_drive(ck,drive)
    old=np.load(OLD/'external_input.npz'); cells=old['cell_e']; counts=old['count_e']
    expected_old=old['expected_rate_per_ms']; nu=p.nu_ext_ratio*compute_nu_theta(p)[0]
    aa=np.exp(-p.dt/p.tau_n); bb=p.sigma_n*.001*np.sqrt(p.tau_n/2)*np.sqrt(1-aa*aa)
    expected=np.empty((60000,2,100),np.float32); max_error=0.; checked=[]
    for k in range(60000):
        xi=aa*xi+bb*rng.standard_normal(); glob=max(0.,nu+xi)
        vec=np.full(n,glob);vec[:ne]=np.maximum(vec[:ne]+drive.step(9400+k*p.dt),0.)
        rng.poisson(vec*p.dt,size=n)
        expected[k,0]=np.bincount(cells,weights=vec[:ne],minlength=100)/counts;expected[k,1]=glob
        if k+94000<len(expected_old):
            err=float(np.max(abs(expected[k]-expected_old[k+94000])));max_error=max(max_error,err)
            assert err<5e-7, (k,err)
        if k+1 in (20000,60000):
            name='z9400_history8000'+('' if k+1==20000 else '_extend4s')
            target=load(OUT/'native_endpoints'/f'{name}.npz')
            assert xi==target['xi'] and rng.bit_generator.state==target['rng_state']
            assert drive._rng.bit_generator.state==target['external_drive']['rng_state']
            assert np.array_equal(drive._cached,target['external_drive']['cached'])
            checked.append(name)
        if (k+1)%10000==0:write(OUT/'mixed_boundary_input_status.json',{'status':'RUNNING','time_ms':(k+1)*p.dt})
    np.savez_compressed(OUT/'mixed_boundary_input.npz',expected_rate_per_ms=expected,cell_e=cells,count_e=counts)
    write(OUT/'mixed_boundary_input_status.json',{'status':'COMPLETE','seconds':time.time()-started,
        'native_endpoint_RNG_and_drive_exact':checked,'maximum_previous_input_difference':max_error,'frozen_identity':identity})


def run(z_time):
    started=time.time();s=MixedTimescaleSystem(quadrature=33);st=Stepper(system=s);m=s.m
    st.restore(OUT/'mixed_timescale_checkpoints/native_replay_t8000ms.npz')
    src=np.load(OUT/'mixed_boundary_input.npz');expected=src['expected_rate_per_ms'];cells=src['cell_e']
    z_neuron=load(checkpoint_path(z_time))['slow']['z'][:len(cells)]
    z=np.bincount(cells,weights=z_neuron,minlength=s.n)/m.count_e
    fields=np.empty((6000,2,s.n),np.float32);currents=np.empty((6000,6,s.n),np.float32)
    name=f'z{z_time}_history8000'
    for k in range(60000):
        r=st.step(z,expected[k,0].astype(float),expected[k,1].astype(float))
        if (k+1)%10==0:fields[k//10]=r.reshape(2,s.n)*1000;currents[k//10]=st.c
        if (k+1)%5000==0:write(OUT/'mixed_boundary_progress'/f'{name}.json',
            {'status':'RUNNING','time_ms':(k+1)*s.dt,'seconds':time.time()-started})
    dest=OUT/'mixed_boundary';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'{name}.npz',fields_hz=fields,current=currents,z=z,count_e=m.count_e,count_i=m.count_i)
    st.save(OUT/'mixed_boundary_endpoints'/f'{name}.npz',z=z,absolute_time_ms=15400.)
    windows=[observables(fields[k:k+1000,0]*m.count_e/1000,m.count_e) for k in range(0,6000,1000)]
    row={'status':'COMPLETE','name':name,'seconds':time.time()-started,'mean_Z':float(np.average(z,weights=m.count_e)),
        'windows':windows,'scope':'Same fixed native Z field and exact future expected input; full corrected-rate history from its native-Z replay at 8 s. Not an exact microscopic state projection.'}
    write(dest/f'{name}.json',row);return row


def main():
    assert read(OUT/'mixed_timescale_status.json')['status']=='COMPLETE'
    write(OUT/'mixed_boundary_protocol.json',{'status':'DEFINED_BEFORE_RUNS','jobs':[8800,9400],
        'duration_ms':6000,'history_ms':8000,'future_clock_ms':9400,'workers':2,
        'authorization':'Final at-most-two frozen-boundary continuations specified in mixed_timescale_protocol.json.',
        'reason':'The corrected native-Z replay restores zero persistent occupancy in the 8–9.4 s window and improves its high trigger to 10.15 s, but autonomous Z worsens to 7.95 s. Fixed-field tests distinguish remaining fast-state mismatch from divergence along the autonomous spatial Z path.',
        'comparison':'Native six-second z8800/history8000 and z9400/history8000, identical expected future input, common 10x10 grid, each one-second quiet-gap and spatial-occupation readouts.',
        'gate':'Require both self-limited and sustained finite-time regimes before attributing the autonomous mismatch primarily to its evolving Z path. A pass does not establish a bifurcation or validate autonomous closure; a failure retains the fast approximation as an unresolved cause. No further automatic search.'})
    prepare_input();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,z) for z in (8800,9400)]):
            rows.append(f.result());write(OUT/'mixed_boundary_status.json',{'status':'RUNNING','completed':len(rows)})
    write(OUT/'mixed_boundary_status.json',{'status':'COMPLETE','rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'mixed_boundary_status.json',{'status':'FAILED','error':repr(exc)});raise
