#!/usr/bin/env python3
"""Read all short density prerequisites; do not certify from a short prefix."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d
from campaign import ROOT,REPO,read,write,sha
sys.path.insert(0,str(REPO/'scripts/topic4_zm_runaway_mechanism/frozen_v3'))
from native_readouts import readouts,window_stats

OUT=ROOT/'density_spatial_baseline'
BASE=REPO/'results/topic4_sef_hfo/fig5_zm_rate_v3_20260918/native_reference'


def summarize(t,field,count,label):
    events,summary,whole,sm=readouts(t,field,count,label)
    complete=[]
    for e in events:
        a=int(np.searchsorted(t,e['start_ms']));b=a+int(e['duration_ms'])
        if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():complete.append(e)
    primary=window_stats(events,500,3000);strict=window_stats(complete,500,3000)
    return dict(summary=summary,events=[{k:v for k,v in e.items() if k!='onset'} for e in events],
        primary_500_3000=primary,complete_quiet_bounded_500_3000=strict,
        quiet_fraction_500_3000=float((sm[(t>=500)&(t<3000)]<5).mean()),
        field_mean_Hz=field[(t>=500)&(t<3000)].mean(0).tolist()),whole,sm


def main():
    assert read(OUT/'runs.json')['status']=='COMPLETE';rows=[]
    count=np.load(REPO/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')['cell_e_counts']
    for seed in [9108401,9108402]:
        with np.load(BASE/f'seed{seed}_readouts.npz') as z:
            d,_,_=summarize(z['t'][:3000],z['rate_cells'][:3000],count,f'native{seed}')
        d.update(label=f'native{seed}',kind='native',source=str(BASE/f'seed{seed}_readouts.npz'));rows.append(d)
    for name in read(OUT/'runs.json')['completed']:
        source=OUT/name/'trajectory.npz'
        with np.load(source) as z:
            assert np.array_equal(z['cell_counts'],count)
            d,_,_=summarize(z['time_ms'],z['field_E_Hz'],count,name)
            d['final_Z_allE']=float(np.average(z['group_Z'][-1,z['population_E']],weights=z['group_sizes'][z['population_E']]))
        d.update(label=name,kind='density_approximation',source=str(source));rows.append(d)
    result=dict(status='SHORT_PREFIX_COMPARISON_ONLY',rows=rows,source_sha256=sha(__file__),
        native_statistical_unit='Two existing SNN realizations, same physical graph. Only seed8401 supplies the matched projected externaldrive.',
        density_statistical_unit='Two numerical resolutions of a grouped Gaussian density approximation; same externaldrive and nested numerical streams. Not new native seeds.',
        positive='Both resolutions retain separated short spatial events. This is a necessary early-substrate check only.',
        remaining='3s does not test lateZdepletion,entry,G/Ktermination,return,contactenvelopeorfullnoise tolerance. It cannot certify a closure or bifurcation.',
        formal_bifurcation_allowed=False,human_review='PENDING')
    write(OUT/'comparison.json',result)
    print([{k:r[k] for k in ['label','primary_500_3000','quiet_fraction_500_3000']} for r in rows],flush=True)


if __name__=='__main__':main()
