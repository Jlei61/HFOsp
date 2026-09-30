#!/usr/bin/env python3
"""Exact continuous replay with native 0.5-mm cell currents and spikes."""
from pathlib import Path
import json
import time
import resource
import numpy as np
from run_topic4_historical_manual_z import setup, RefilledZ, MZSlowVarsConfig, make_external_drive, simulate_kick
from validate_topic4_fixed_rate_base import spatial_cell_index
from lfp import LFPRecorder

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'native_transition_v2'


def write(name, value):
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/name;temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(target)


def main():
    started=time.time();duration=11180.;seed=9108401
    write('protocol.json',dict(duration_ms=duration,seed=seed,grid_shape=[40,40],cell_size_mm=.5,
        current_sample_dt_ms=.5,spike_count_dt_ms=1.,Z_sample_dt_ms=5.,
        source=str(BASE/'runs/continuous_refill_release.npz'),
        scope='Exact unperturbed prefix through 11.18 s before external refill; same topology, Vth, native Z and noise. New observers only.',
        planned_windows_s=[[10.25,10.5],[10.5,10.75],[10.75,11.0]],baseline_s=[.5,8.],
        spectral_estimator='250-ms Hann periodogram, linear detrending, Fs=2000 Hz, 4-Hz bins. Equal-duration baseline windows averaged over 0.5-8 s.',
        primary_band_hz=[20,150],secondary_bands_hz=[[20,40],[40,80],[80,150],[150,250]],
        power_signal='Native cell mean |AMPA|+|Z*GABA| current; no virtual electrodes or spatial interpolation.',
        acceptance='Exact sampled spike and E/I count parity; exact virtual-SEEG parity; 40x40 counts sum back to original 20x20 counts. Report missing band enhancement if absent.'))
    write('status.json',dict(status='BUILDING',started_unix=started))
    original=np.load(BASE/'runs/continuous_refill_release.npz')
    s,tr,frozen,identity=setup(seed)
    assert identity==json.loads((BASE/'runs/continuous_refill_release.json').read_text())['frozen_identity']
    ne,ni=s.n_e,s.n_i;dt=s.params.dt;s.params.T=duration
    assert dt==.1
    ce=spatial_cell_index(s.positions_e,n_grid=40,sheet_l_mm=s.params.L)
    ci=spatial_cell_index(s.positions_i,n_grid=40,sheet_l_mm=s.params.L)
    nc=np.bincount(ce,minlength=1600);nic=np.bincount(ci,minlength=1600)
    assert nc.min()>0
    nsteps=round(duration/dt);ns=round(duration/.5);nz=round(duration/5.)
    ampa=np.empty((ns,1600),np.float32);gaba=np.empty_like(ampa)
    ze=np.empty((nz,1600),np.float32)
    ec=np.zeros((round(duration),1600),np.uint16);ic=np.zeros_like(ec)
    rates=np.empty((nsteps,2),float);spikes=np.empty((nsteps,len(original['sample_ids'])),bool)
    ids=original['sample_ids'];lr=np.empty((ns,len(original['contact_xy'])),float)
    recorder=LFPRecorder(s.params,s.net['pos'],s.net['labels'],sites=s.contact_xy)
    slow=RefilledZ(ne+ni,s.params.V_th,MZSlowVarsConfig(use_z=True,use_m=False,tau_z=5000.,I_th_EI=95.19851312666987),NE=ne)
    first_prefix=False
    def observe_current(tm,ie,ii,v):
        k=round(tm/dt)
        if k%5==0:
            j=k//5;applied=ii.copy();applied[:ne]*=slow.z[:ne]
            ampa[j]=np.bincount(ce,weights=abs(ie[:ne]),minlength=1600)/nc
            gaba[j]=np.bincount(ce,weights=abs(applied[:ne]),minlength=1600)/nc
            lr[j]=recorder.sample(ie,applied)
        if k%50==0:
            ze[k//50]=np.bincount(ce,weights=slow.z[:ne],minlength=1600)/nc
    def observe_spikes(tm,spk):
        nonlocal first_prefix
        k=round(tm/dt);e=spk[:ne];i=spk[ne:]
        rates[k]=[e.sum()/ne/dt*1000,i.sum()/ni/dt*1000]
        spikes[k]=spk[ids]
        ec[k//10]+=np.bincount(ce[e],minlength=1600).astype(np.uint16)
        ic[k//10]+=np.bincount(ci[i],minlength=1600).astype(np.uint16)
        if k+1==6000:
            assert np.array_equal(spikes[:k+1],original['sample_spikes'][:k+1])
            assert np.array_equal(rates[:k+1,0],original['rate_e_hz'][:k+1])
            assert np.array_equal(rates[:k+1,1],original['rate_i_hz'][:k+1])
            first_prefix=True
        if (k+1)%5000==0:
            write('status.json',dict(status='RUNNING',time_ms=(k+1)*dt,elapsed_s=time.time()-started,
                  initial_600ms_bitwise_equal=first_prefix))
    s.net['rng']=np.random.default_rng(seed)
    drive=make_external_drive(s,tr['spatial_ou'],seed)
    simulate_kick(s.params,s.net,KICK_BOOST=0.,V_th_per_neuron=s.vtheta,slow=slow,
        external_e_rate_drive=drive,early_stop_runaway=False,current_observer=observe_current,
        spike_observer=observe_spikes,record_dense_spikes=False,fast_scatter=True,verbose=False)
    coarse=ec.reshape(round(duration),20,2,20,2).sum((2,4)).reshape(round(duration),400)
    qa=dict(sampled_spikes_bitwise_equal=bool(np.array_equal(spikes,original['sample_spikes'][:nsteps])),
        E_rates_bitwise_equal=bool(np.array_equal(rates[:,0],original['rate_e_hz'][:nsteps])),
        I_rates_bitwise_equal=bool(np.array_equal(rates[:,1],original['rate_i_hz'][:nsteps])),
        virtual_SEEG_bitwise_equal=bool(np.array_equal(lr,original['lfp_effective'][:ns])),
        native_grid_count_equal=bool(np.array_equal(coarse,original['field_e_count_1ms'][:round(duration)])),
        I_grid_count_conservation=bool(np.array_equal(ic.sum(1),np.rint(rates[:,1]*ni*dt/1000).reshape(-1,10).sum(1))))
    assert all(qa.values()),qa
    np.savez_compressed(OUT/'native_fields.npz',time_s=np.arange(ns)*.0005,
        cell_mean_ampa=ampa,cell_mean_applied_gaba=gaba,Z_time_s=np.arange(nz)*.005,cell_mean_Z=ze,
        E_count_1ms=ec,I_count_1ms=ic,cell_n_E=nc,cell_n_I=nic,
        centers_mm=original['centers_mm'],positions_e=s.positions_e,cell_e=ce,
        contact_xy=original['contact_xy'])
    write('qa.json',dict(checks=qa,frozen_identity=identity,seconds=time.time()-started,
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))
    write('status.json',dict(status='COMPLETE',elapsed_s=time.time()-started,qa_pass=True))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('status.json',dict(status='FAILED',error=repr(exc)))
        raise
