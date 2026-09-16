#!/usr/bin/env python3
"""Development-only colored-noise transfer check; no reuse as a blind test."""
from validate_topic4_fixed_rate_base import OUT,read,write
from src.topic4_patient_zm_meanfield import lif_rate_gauss_legendre
import numpy as np
from scipy.ndimage import uniform_filter1d


def main(population='E'):
    suffix='' if population=='E' else '_I'
    protocol=read(OUT/f'isolated_response_protocol{suffix}.json');p=read(OUT/'reconstruction.json')['params']
    if population=='I':
        p={**p,'tau_m_E':p['tau_m_I'],'tau_ref_E':p['tau_ref_I'],'J_ext_E':p['J_ext_I']}
        nodes,counts=np.array([p['V_th']]),np.array([8000])
    else:nodes,counts=np.unique(np.load(OUT/'coarse_20/geometry.npz')['vtheta'],return_counts=True)
    data=np.load(OUT/f'isolated_response{suffix}.npz');dt=float(data['dt_ms']);steps=len(data['snn_rate_hz'])
    taus=data['tau_rate_ms']
    gating=0.;current=0.;rr=np.zeros(len(taus));rates=np.empty((steps,len(taus)))
    for step in range(steps):
        t=step*dt;nu=protocol['baseline_external_rate_per_ms']+sum(d for lo,hi,d in protocol['pulses'] if lo<=t<hi)
        gating=gating*np.exp(-dt/p['tau_r_AMPA'])+nu*dt*p['tau_m_E']/p['tau_r_AMPA']*p['J_ext_E']
        current=gating+(current-gating)*np.exp(-dt/p['tau_d_AMPA'])
        sigma=np.sqrt(p['tau_m_E']*p['J_ext_E']**2*nu)
        shift=2.065/2*sigma*np.sqrt((p['tau_r_AMPA']+p['tau_d_AMPA'])/p['tau_m_E'])
        phi=np.average(lif_rate_gauss_legendre(current-shift,sigma,tau_mem_ms=p['tau_m_E'],
            tau_ref_ms=p['tau_ref_E'],v_threshold_mv=nodes,v_reset_mv=p['V_reset']),weights=counts)
        rr+=dt/taus*(phi-rr);rates[step]=rr*1000
    actual=uniform_filter1d(data['snn_rate_hz'],50);smooth=uniform_filter1d(rates,50,axis=0);rows=[];errors=[]
    for lo,hi,dose in protocol['pulses']:
        ix=slice(int(lo/dt),int((lo+150)/dt));error=np.mean((smooth[ix]-actual[ix,None])**2,axis=0)/np.mean(actual[ix]**2)
        errors.append(error);rows.append({'dose':dose,'normalized_mse':error.tolist(),
            'rate_peak_5ms_hz':smooth[ix].max(0).tolist(),'rate_peak_lag_ms':(np.argmax(smooth[ix],axis=0)*dt).tolist()})
    best=int(np.argmin(np.sum(errors[:2],axis=0)))
    write(OUT/f'colored_response_diagnostic{suffix}.json',{'status':'DEVELOPMENT_ONLY_COMPLETE','population':population,
        'source':'Fourcaud & Brunel 2002; leading colored-noise threshold/reset shift',
        'formula':'delta = alpha/2 * sigma * sqrt((tau_r+tau_d)/tau_mem); Phi(mu-delta,sigma)',
        'limitation':'two-exponential correlation replaced by tau_r+tau_d; finite-time and large-noise behavior require direct validation; pulse4 previously inspected, not new heldout evidence',
        'best_tau_rate_ms':float(taus[best]),'fit_error':float(np.sum(errors[:2],axis=0)[best]),
        'dose4_diagnostic_error':float(errors[2][best]),'pulse_summaries':rows})
    np.savez_compressed(OUT/f'colored_response_diagnostic{suffix}.npz',rate_closures_hz=rates,tau_rate_ms=taus)
    print('colored',taus[best],'fit',np.sum(errors[:2],axis=0)[best],'dose4',errors[2][best],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--population',choices=['E','I'],default='E')
    main(parser.parse_args().population)
