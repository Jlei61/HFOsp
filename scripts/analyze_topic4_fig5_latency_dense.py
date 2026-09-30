#!/usr/bin/env python3
"""Complete-grid latency statistics and the corresponding Z depletion traces."""
import csv
import json
import numpy as np
from run_topic4_fig5_latency_dense import OUT, BASE, write


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    paths=[r['source'] for r in protocol['reused_runs']]
    paths += [str(OUT/'runs'/(j['name']+'.json')) for j in protocol['jobs']]
    tau=np.array(protocol['tau_z_ms']);threshold=np.array(protocol['threshold'])
    observed=np.empty((7,7,3),bool);times=np.empty((7,7,3));zend=np.empty_like(times)
    rowset=[];traces={};identity=None;cellkeys=set()
    for name in paths:
        from pathlib import Path
        path=Path(name);result=json.loads(path.read_text());assert result['status']=='COMPLETE'
        if identity is None:identity=result['frozen_identity']
        assert result['frozen_identity']==identity
        job=result['job'];x=int(np.argmin(abs(tau-job['tau_z_ms'])));y=int(np.argmin(abs(threshold-job['threshold'])))
        s=protocol['seeds'].index(job['seed']);assert (y,x,s) not in cellkeys;cellkeys.add((y,x,s))
        time_ms=result['first_trigger_ms'];event=time_ms is not None and time_ms<=24000.
        event_time=time_ms/1000 if event else None
        capped=event_time if event else 24.
        times[y,x,s]=capped;observed[y,x,s]=event
        a=np.load(path.with_suffix('.npz'));t=a['z_time_ms']/1000;zs=a['z_stats']
        zz=(result['final_mean_Z'] if capped>t[-1] and abs(result['duration_ms']/1000-capped)<1e-8
            else np.interp(capped,t,zs[:,0]));zend[y,x,s]=zz
        rowset.append(dict(tau_z_s=float(tau[x]/1000),threshold=float(threshold[y]),seed=job['seed'],
            observed=event,transition_time_s=event_time,restricted_time_s=capped,Z_at_end=float(zz),source=str(path)))
        if y==3 or x==3:
            mask=t<=capped
            traces[f'y{y}_x{x}_s{s}_time_s']=t[mask]
            traces[f'y{y}_x{x}_s{s}_depletion']=1-zs[mask,0]
            traces[f'y{y}_x{x}_s{s}_duty']=zs[mask,8]
    assert len(cellkeys)==147
    means=times.mean(-1);fractions=observed.mean(-1)
    np.savez_compressed(OUT/'analysis_arrays.npz',tau_s=tau/1000,threshold=threshold,
        restricted_mean_s=means,restricted_sd_s=times.std(-1,ddof=1),transition_fraction=fractions,
        run_restricted_times_s=times,observed=observed,Z_at_end=zend,**traces)
    with (OUT/'transition_times.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rowset[0]));w.writeheader();w.writerows(rowset)
    summary=dict(status='ANALYSIS_COMPLETE',grid_shape=[7,7],runs=147,new_runs=120,reused=27,
        tau_s=(tau/1000).tolist(),threshold=threshold.tolist(),restricted_mean_time_s=means.tolist(),
        transition_fraction=fractions.tolist(),n_censored=int((~observed).sum()),
        adjacent_tau_mean_reversals=int((np.diff(means,axis=1)<0).sum()),
        adjacent_threshold_mean_reversals=int((np.diff(means,axis=0)<0).sum()),
        tau_center_threshold_means_s=means[3].tolist(),threshold_center_tau_means_s=means[:,3].tolist(),
        statistical_unit='Three paired noise seeds, one fixed topology, each simulated grid node; no interpolation or independent-patient inference.',
        parameter_semantics='tau_Z controls both depletion and recovery. I_th is a depletion-current threshold, not inhibitory synapse strength.',
        mechanism_identity='For native E-only Z, with d=1-mean(Z) and u=fraction(I_GABA>=I_th): tau_Z * d_dot = u-d. Z encodes filtered depletion drive, not unweighted accumulated spike count.',
        timing='Detection after 200 ms >=200Hz all-E activity. Unreached runs are capped at 24s and counted as censored.',
        frozen_identity=identity)
    write(OUT/'analysis_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
