#!/usr/bin/env python3
"""Read-only phase persistence in the already completed matched native control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from campaign import ROOT,read,write,sha

OUT=ROOT/'native_phase_persistence_review'
L=22


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    source=ROOT/'native_K9p35_constant_background_pair_v2';name='high_history_constant_background'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_EXISTING_RASTER_PHASE_AUDIT',created_epoch=time.time(),
        question='Is the reduced phase structure of the phase-map output comparable to native temporalvariation after matching the external protocol?',
        design='Read the80originallyfixedraster samplecells at0.1ms in the completed72-82s highhistory fixedbackgroundcontrol. Compute22step sourcephaseprobabilities in each2s block andthefull10s, aligned to72s phase0. Compare the samecells in the variable-background native72-74s initialguess andphase-map generations1-3. No newneural simulation, periodsearch, or phasefit.',
        limitations='80stratifiedrastercells are not the40000cellnetwork or independentseeds. Nativephase probabilities use onepath; mapoutputs average64 numericalreplicas. Withinpathpersistence can expose finitewindow/phase-drift effects but is not an ensemble equivalence or formalstabilitytest.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    geo=np.load(source/'geometry.npz');cells=geo['sample_ids'];assert len(cells)==80
    chunks=sorted((source/'runs'/name/'chunks').glob('*.npz'));assert len(chunks)==5
    rasters=[];profiles=[];counts=[]
    for k,path in enumerate(chunks):
        start,end=map(int,path.stem.split('_'));assert start==720000+k*20000 and end==start+20000
        x=np.load(path)['raster'];assert x.shape==(20000,80)
        phase=(np.arange(20000)+start-720000)%L;c=np.bincount(phase,minlength=L)
        prof=np.stack([x[phase==q].mean(0) for q in range(L)],axis=1)
        assert np.allclose(prof@c,x.sum(0),rtol=0,atol=1e-10)
        profiles.append(prof);counts.append(c);rasters.append(x)
    profiles=np.array(profiles);counts=np.array(counts);x=np.concatenate(rasters)
    full=np.sum(profiles*counts[:,None,:],axis=0)/counts.sum(0)
    model=ROOT/'high_history_phase_source_closure'
    comparison=np.array([np.load(model/f'generation_{g}/phase_spike_probability.npy')[cells] for g in range(4)])
    raw=np.load(model/'parameters.npz');region=raw['region'][cells];E=cells<32000;rows=[]
    for label,mask in [('sampled_E',E),('sampled_coreA',E&(region==0)),('sampled_coreB',E&(region==1)),('sampled_I',~E)]:
        rows.append(dict(region=label,targets=int(mask.sum()),
            native_fixed_background_2s_phase_variance=profiles[:,mask].var(2).mean(1).tolist(),
            native_fixed_background_10s_phase_variance=float(full[mask].var(1).mean()),
            initial_variable_native_and_three_map_phase_variance=comparison[:,mask].var(2).mean(1).tolist(),
            native_fixed_background_mean_rate_Hz=(x[:,mask].mean(0)*10000).mean().item(),
            native_fixed_phase_profile_change_from_first_block_RMS=np.sqrt(np.mean((profiles[:,mask]-profiles[0,mask])**2,axis=(1,2))).tolist()))
    np.savez_compressed(OUT/'profiles.npz',cells=cells,phase_probabilities_native_fixed_blocks=profiles,
        native_fixed_full10s=full,variable_native_initial_and_map_probabilities=comparison,
        phase_exposures=counts,native_raster_rate_Hz=x.mean(0)*10000)
    result=dict(status='COMPLETE_EXISTING_NATIVE_PHASE_PERSISTENCE_AUDIT',rows=rows,
        scope=read(OUT/'contract.json')['limitations'],producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py');print(result,flush=True)


if __name__=='__main__':main()
