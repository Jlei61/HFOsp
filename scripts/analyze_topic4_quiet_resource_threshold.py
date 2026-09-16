#!/usr/bin/env python3
"""One-sided current bounds on observed quiet periods; NOT new simulations."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np,time
import analyze_topic4_autonomous_recovery as common
from analyze_topic4_long_recovery_contrast import PAIRS,BASE,load_pool

def main():
    rows=[]
    for subdir,name,label in PAIRS:
        root=BASE/subdir;folder=root/'runs'/name
        result=common.read(folder/'result.json');job=result['job']
        a=common.load(folder,['spikes_1ms','regions_1ms']);nr=np.load(root/'geometry.npz')['region_counts']
        n=len(a['spikes_1ms'])//10
        re=a['spikes_1ms'][:,0].reshape(n,10).sum(1)/32000/.01
        rr=a['regions_1ms'].astype(float).reshape(n,10,6).sum(1)/nr/.01
        quiet=(re<5)&(rr[:,:2]<5).all(1)
        p=load_pool(folder);t=p['time_ms'][:-1]/1000
        assert np.allclose(np.diff(p['time_ms']),5)
        ix=np.floor((t+.0025)/.01).astype(int)
        use=quiet[ix]&(t>=result['tracker']['entries'][0]['confirmation_s'])
        low=job['pool_gain']*np.maximum(p['rate_Hz'][:-1]*np.exp(-.005/job['pool_tau_s'])-job['pool_threshold_Hz'],0)
        thresholds=[job['threshold'],200.,500.,1000.,2000.]
        values=[dict(hypothetical_threshold_mV_equiv=threshold,
                     observed_quiet_fraction_guaranteed_above_threshold=float((low[use]>=threshold).mean()))
                for threshold in thresholds]
        row=dict(name=name,label=label,actual_threshold_mV_equiv=job['threshold'],
            observed_s=n*.01,observed_postentry_quiet_s=float(use.sum()*.005),
            minimum_global_current_during_quiet_quantiles_mV_equiv=np.quantile(low[use],[0,.1,.5,.9,1]).tolist(),
            hypothetical_threshold_bound_diagnostic=values)
        rows.append(row);print(label,values,flush=True)
    common.write(BASE/'overnight_review/quiet_resource_threshold_bound.json',dict(updated_at=time.time(),rows=rows,
        definition='Observed all-E and both cores<5Hz in10ms bins after first high confirmation. Within each enclosed5ms interval, nonnegative spike increments give a strict lower bound on the global inhibitory current.',
        interpretation='At the actual threshold, above-threshold global input alone guarantees b=0 despite quiet E firing. Alternative thresholds only diagnose current headroom on this same recorded trajectory; they are NOT rerun parameter conditions and cannot predict entry, termination or Z recovery under a changed threshold. Local GABA can still prevent recovery when the global lower bound is below a hypothetical threshold.',
        next_question='Before treating rho as necessary, test the original Ith/tauZ working point explicitly: allow actual inhibition to recover during quiet periods while preserving entry and the initial finite event repertoire. No such additional simulation is launched by this observer.'))

if __name__=='__main__':main()
