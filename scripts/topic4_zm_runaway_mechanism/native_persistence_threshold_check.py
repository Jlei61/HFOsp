"""Check whether the observed loss of quiet is just the 5-Hz readout choice.

Exploratory readout sensitivity on existing, unchanged trajectories. It cannot
distinguish a dynamical crisis from a long, censored intermittent episode.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def readout(fields,counts):
    assert len(fields)%10==0
    r=fields.reshape(-1,10,400).mean(1)@(counts/counts.sum())
    rows=[]
    for threshold in [1.,2.5,5.,10.,20.]:
        below=r<threshold;edges=np.diff(np.r_[0,below.astype(int),0])
        lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
        rows.append(dict(threshold_hz=threshold,fraction_below=float(below.mean()),
            quiet_runs_ge20ms=int((lengths>=2).sum()),
            longest_quiet_ms=int(lengths.max()*10) if len(lengths) else 0))
    return dict(quantile_probabilities=[0,.01,.05,.5,1.],
        global_E_rate_quantiles_hz=np.quantile(r,[0,.01,.05,.5,1.]).tolist(),quiet_sensitivity=rows)


def main():
    data=json.loads((OUT/'native_postcritical_endpoint_audit.json').read_text());rows=[]
    for row in data['rows']:
        path=row['spatial']['source'];z=np.load(path)
        rows.append(dict(D=row['D'],source=path,window_ms=[8000,12000],
            **readout(z['field_E_hz'][-4000:],z['cell_counts'])))
    extension=json.loads((OUT/'native_9420_matched_extension_readout.json').read_text())
    sources=[np.load(p) for p in extension['sources']]
    fields=np.concatenate([z['field_E_hz'] for z in sources])
    long=dict(D=extension['D'],sources=extension['sources'],window_ms=[4000,72000],
        **readout(fields[4000:],sources[0]['cell_counts']))
    q=dict(status='EXPLORATORY_READOUT_SENSITIVITY_COMPLETE',rows=rows,extended9420=long,
        observable='Original all-E neuron-weighted rate in non-overlapping10ms bins; quiet duration remains20ms; only threshold varied',
        baseline='Original5Hz criterion; sensitivity1,2.5,10,20Hz',
        unit='One deterministic trajectory per held spatialZ, same original periodic initial state; long result is one continuous history, not independent blocks',
        limitation='No redefinition of primary category. Finite loss of quiet across thresholds does not identify a bifurcation or prove an infinite-time attractor.')
    (OUT/'native_persistence_threshold_check.json').write_text(json.dumps(q,indent=2)+'\n')
    print('extended minimum Hz',long['global_E_rate_quantiles_hz'][0],flush=True)


if __name__=='__main__':main()
