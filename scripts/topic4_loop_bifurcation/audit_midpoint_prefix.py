#!/usr/bin/env python3
"""Read a complete ten-second prefix while the native thirty-second cuts run."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_native import original
from coupled_density_exit import ADAPTED
import analyze_target_density_exit as target

OUT=ROOT/'exit_midpoint_prefix_review'


def main():
    OUT.mkdir(exist_ok=True);root=ROOT/'exit_midpoint_probes';geo=dict(np.load(ADAPTED/'geometry.npz'))
    native_geo=dict(np.load(root/'geometry.npz'));counts=np.r_[32000,native_geo['region_counts'][:3]];cells=native_geo['cell_e_counts'];rows=[]
    reference=None
    for history in ['high','recovery']:
        name=f'exit_z0.21_k10.5_fields16p7_{history}';folder=root/'runs'/name;job=read(root/'jobs'/f'{name}.json')
        start=round(job['branch_start_s']*10000);end=start+100000;parts=[]
        for path in sorted((folder/'chunks').glob('*.npz')):
            a,b=map(int,path.stem.split('_'))
            if a>=start and b<=end:
                with np.load(path) as z:parts.append({key:z[key] for key in ['spikes_1ms','regions_1ms','field_5ms','inputs','time_ms']})
        if sum(len(p['time_ms']) for p in parts)!=10000:continue
        raw=np.concatenate([np.c_[p['spikes_1ms'][:,0],p['regions_1ms'][:,:3]] for p in parts]);rate=raw/counts*1000
        field=np.concatenate([p['field_5ms'] for p in parts])/cells/.005
        assert np.array_equal(raw[:,0],raw[:,1:].sum(1))
        inp=np.concatenate([p['inputs'] for p in parts])[:,1:]
        assert inp.shape==(100,3)
        if reference is None:reference=inp
        else:assert np.array_equal(reference,inp)
        oldname=f'exit_z0.21_k9_fields16p7_{history}';oldjob=read(ROOT/'exit_return_probes/jobs'/f'{oldname}.json')
        old=original.load(ROOT/'exit_return_probes/runs'/oldname/'chunks',['inputs'])['inputs']
        old=old[(old[:,0]>=oldjob['branch_start_s']*1000)&(old[:,0]<(oldjob['branch_start_s']+10)*1000),1:]
        assert np.array_equal(inp,old)
        drift=original.load(folder/'conditional_drift_chunks',['time_ms','values']);tm=drift['time_ms']/1000-job['branch_start_s']
        glob=original.load(folder/'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
        keep=(glob['time_ms']>=start*.1)&(glob['time_ms']<end*.1);assert keep.sum()==10000
        R=glob['global_E_rate_Hz'][keep];G=glob['global_raw_conductance_ratio'][keep]
        row=dict(name=name,prefix_complete_s=10.,full_declared_s=30.,native_full_complete=(folder/'result.json').exists(),paired_input_records_bitwise=100,
             native_windows=[])
        for lo,hi in [(0,5),(5,10)]:
            mask=(tm>lo)&(tm<=hi);assert mask.sum()==250
            row['native_windows'].append(dict(interval_s=[lo,hi],rates_Hz=rate[lo*1000:hi*1000].mean(0).tolist(),
                Graw_mean=float(G[lo*1000:hi*1000].mean()),dZ_per_s=drift['values'][mask,:,0].mean(0).tolist()))
        candidate=ROOT/'target_density_field_family'/name/'result.json'
        if candidate.exists():
            assert read(candidate)['status']=='COMPLETE';previous=target.OUT;target.OUT=ROOT/'target_density_field_family'
            try:d=target.load_candidate(name,geo)
            finally:target.OUT=previous
            row['candidate_comparison']=[]
            for lo,hi in [(0,5),(5,10)]:
                delta=d['field'][lo*1000:hi*1000].mean(0)-field[lo*200:hi*200].mean(0)
                row['candidate_comparison'].append(dict(interval_s=[lo,hi],rates_Hz=d['rate'][lo*1000:hi*1000].mean(0).tolist(),
                    Graw_mean=float(d['G'][lo*1000:hi*1000].mean()),dZ_per_s=d['drift'][lo*1000:hi*1000].mean(0).tolist(),
                    field_RMS_Hz=float(np.sqrt(np.average(delta**2,weights=cells)))))
        np.savez_compressed(OUT/f'{name}.npz',rate_1ms_Hz=rate,field_5ms_Hz=field,R_Hz=R,Graw=G)
        rows.append(row);print(row,flush=True)
    write(OUT/'analysis.json',dict(status='COMPLETE_PREFIXES' if len(rows)==2 else 'PARTIAL',rows=rows,
        interpretation='Both complete ten-second prefixes; do not imply complete thirty-second observation, stationary states, or independent native repetitions.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__)))


if __name__=='__main__':main()
