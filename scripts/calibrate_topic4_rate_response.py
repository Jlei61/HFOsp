#!/usr/bin/env python3
"""Isolate the LIF response closure using original thresholds and synaptic kinetics."""
from validate_topic4_fixed_rate_base import OUT, read, write
from src.topic4_patient_zm_meanfield import load_patient_coarse_model, lif_rate_gauss_legendre
import numpy as np
from scipy.ndimage import uniform_filter1d
import time


def main(population='E'):
    start=time.time();p=read(OUT/'reconstruction.json')['params'];dt=p['dt'];nsteps=22000
    suffix='' if population=='E' else '_I'
    if population=='I':
        p={**p,'tau_m_E':p['tau_m_I'],'tau_ref_E':p['tau_ref_I'],'J_ext_E':p['J_ext_I']}
        thresholds=np.full(8000,p['V_th'])
    else:thresholds=np.load(OUT/'coarse_20/geometry.npz')['vtheta']
    n=len(thresholds)
    model=load_patient_coarse_model(OUT/'coarse_20/model.npz');base=model.nu_ext_per_ms
    levels=np.array([.05,.1,.25,.5,1.]);taus=p['tau_m_E']*levels
    protocol={'role':'open_loop_neuron_response_calibration_not_network_validation','population':population,
        'n_e':n,'baseline_external_rate_per_ms':base,'spatial_OU':'off','global_OU':'off',
        'recurrence':'off to isolate single-population closure','thresholds':'all empirical frozen E thresholds',
        'pulses':[[500.,518.,.25],[1000.,1018.,1.],[1500.,1518.,4.]],
        'tau_rate_candidates_ms':taus.tolist(),'fit_pulses':[0,1],'heldout_pulse':2,
        'calibration_metric':'sum of mean squared error per pulse, normalized by SNN pulse mean-square; 0-150 ms after onset, 5ms smoothing',
        'seed':9108201 if population=='E' else 9108202}
    write(OUT/f'isolated_response_protocol{suffix}.json',protocol)
    rng=np.random.default_rng(protocol['seed']);v=np.full(n,p['V_reset']);s=np.zeros(n);cur=np.zeros(n);ref=np.zeros(n,int)
    rates=np.zeros(nsteps);means=np.zeros(nsteps);closures=np.zeros((nsteps,len(taus)))
    gating=0.;current=0.;rr=np.zeros(len(taus));nodes,counts=np.unique(thresholds,return_counts=True)
    for step in range(nsteps):
        t=step*dt;nu=base+sum(dose for lo,hi,dose in protocol['pulses'] if lo<=t<hi)
        s*=np.exp(-dt/p['tau_r_AMPA']);s+=rng.poisson(nu*dt,n)*(p['tau_m_E']/p['tau_r_AMPA']*p['J_ext_E'])
        cur=s+(cur-s)*np.exp(-dt/p['tau_d_AMPA']);ref=np.maximum(ref-1,0);free=ref==0
        vm=cur+(v-cur)*np.exp(-dt/p['tau_m_E']);v=np.where(free,vm,p['V_reset'])
        spk=free&(v>=thresholds);v[spk]=p['V_reset'];ref[spk]=int(round(p['tau_ref_E']/dt));rates[step]=spk.mean()/dt*1000
        gating=gating*np.exp(-dt/p['tau_r_AMPA'])+nu*dt*p['tau_m_E']/p['tau_r_AMPA']*p['J_ext_E']
        current=gating+(current-gating)*np.exp(-dt/p['tau_d_AMPA'])
        phi=np.average(lif_rate_gauss_legendre(current,np.sqrt(p['tau_m_E']*p['J_ext_E']**2*nu),
            tau_mem_ms=p['tau_m_E'],tau_ref_ms=p['tau_ref_E'],v_threshold_mv=nodes,v_reset_mv=p['V_reset']),weights=counts)
        rr+=dt/taus*(phi-rr);closures[step]=rr*1000;means[step]=cur.mean()
    snn=uniform_filter1d(rates,50);cr=uniform_filter1d(closures,50,axis=0)
    errors=[];summaries=[]
    for lo,hi,dose in protocol['pulses']:
        ix=slice(int(lo/dt),int((lo+150)/dt));error=np.mean((cr[ix]-snn[ix,None])**2,axis=0)/max(np.mean(snn[ix]**2),1e-12)
        errors.append(error);summaries.append({'dose':dose,'normalized_mse':error.tolist(),
            'snn_peak_5ms_hz':float(snn[ix].max()),'rate_peak_5ms_hz':cr[ix].max(0).tolist(),
            'snn_peak_lag_ms':float(np.argmax(snn[ix])*dt), 'rate_peak_lag_ms':(np.argmax(cr[ix],axis=0)*dt).tolist()})
    best=int(np.argmin(np.sum(errors[:2],axis=0)))
    np.savez_compressed(OUT/f'isolated_response{suffix}.npz',snn_rate_hz=rates,rate_closures_hz=closures,
                        actual_mean_current=means,tau_rate_ms=taus,dt_ms=dt)
    write(OUT/f'isolated_response_calibration{suffix}.json',{'status':'COMPLETE','population':population,'seconds':time.time()-start,
        'best_tau_rate_ms':float(taus[best]),'fit_error':float(np.sum(errors[:2],axis=0)[best]),
        'heldout_error':float(errors[2][best]),'pulse_summaries':summaries,
        'claim_boundary':'open-loop calibration only, no release of network gate; changing rate relaxation is a closure correction, not changing membrane tau in SNN'})
    print('best tau',taus[best],'heldout error',errors[2][best],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--population',choices=['E','I'],default='E')
    main(parser.parse_args().population)
