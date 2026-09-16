#!/usr/bin/env python3
"""Observation-only replay of a reviewed global-pool condition.

Keeps every original physics operation and legacy observation unchanged.
Adds 1ms slow/current state and selected 10kHz contact-magnitude components.
The high-rate contact records support proper filtering before downsampling.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,time
from pathlib import Path
import numpy as np
import run_topic4_activity_global_pool as source

ROOT=source.ROOT;SOURCE=source.OUT
def destination(name):return source.PARENT/'native_field_candidates'/name
def prepare(name):
    out=destination(name);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():return source.carrier.base.read(out/'protocol.json')
    p=copy.deepcopy(source.carrier.base.read(SOURCE/'protocol.json'))
    job=source.carrier.base.read(SOURCE/'jobs'/(name+'.json'))
    p['initial_jobs']=[job];p['observation_producer_sha256']=source.carrier.base.sha(__file__)
    p['source_protocol']=str(SOURCE/'protocol.json');p['source_job']=str(SOURCE/'jobs'/(name+'.json'))
    p['source_job_sha256']=source.carrier.base.sha(p['source_job'])
    p['observation_contract']='Same legacy |IE|+|pre-Z deliveredII| proxy retained unchanged. Added |IE|+|Z*deliveredII| proxy plus excitatory and effective global-inhibitory components using the identical E-cell contact weights. These are current-magnitude proxies, not measured SEEG or signed net currents. M is intrinsic and not included in the synaptic proxy.'
    p['dense_state_dt_ms']=1.;p['dense_contact_dt_ms']=.1
    p['native_field_contract']='Direct1mm E-cell averages at native10kHz within the same contact windows, with no electrode projection: effective |IE|+|Z*II| magnitude, |IE| alone, and Z*Jglobal separately. Float32 storage only; physics remains float64. Actual firing-rate fields remain separately available in the native5ms spike-count chunks.'
    p['filtering_contract']='Store at native10kHz within the recorded contact windows. Any1–150Hz power must be computed by filtering this native-rate data first; do not label the unfiltered second moment as band power or simply subsample to500Hz.'
    result=SOURCE/'runs'/name/'result.json'
    if result.exists():
        r=source.carrier.base.read(result);entries=r['tracker']['entries'];rec=r['tracker']['recoveries']
        windows=[[0.,min(job['horizon_s'],entries[0]['confirmation_s']+1.)]] if entries else [[0.,min(job['horizon_s'],15.)]]
        if rec:windows.append([max(0.,rec[0]['start_s']-.5),min(job['horizon_s'],entries[1]['confirmation_s']+1. if len(entries)>1 else rec[0]['confirmation_s']+5.)])
        p['source_result_sha256']=source.carrier.base.sha(result)
    else:windows=[[0.,min(job['horizon_s'],15.)]]
    p['dense_contact_windows_s']=windows
    p['dense_contact_windows_steps']=[[round(lo*10000),round(hi*10000)] for lo,hi in windows]
    source.carrier.base.write(out/'protocol.json',p);source.carrier.base.write(out/'jobs'/(name+'.json'),job)
    return p

def worker(name):
    p=prepare(name);out=destination(name)
    assert source.carrier.base.sha(__file__)==p['observation_producer_sha256']
    job=p['initial_jobs'][0];folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    holder={};records={k:[] for k in ['state_time_ms','state','contact_time_ms','legacy_proxy','effective_proxy','excitation_proxy','effective_global_proxy','native_effective_grid','native_excitation_grid','native_global_grid']}
    original_init=source.GlobalPoolSlow.__init__
    OriginalRecorder=source.carrier.base.old.LFPRecorder
    original_capture=source.checkpoint.capture
    zero=np.zeros(40000,dtype=float)
    def init(obj,*args,**kwargs):
        original_init(obj,*args,**kwargs);holder['slow']=obj;apply0=obj.apply_currents
        def apply(ie,ii,labels=None,rec=None):
            value=apply0(ie,ii,labels,rec);k=obj._step_index;ne=obj.NE
            if k%10==0:
                z=obj.z[:ne];m=obj.m[:ne];jj=obj.delivered[:ne]
                records['state_time_ms'].append(k*.1)
                records['state'].append([z.mean(),ie[:ne].mean(),np.mean(z*jj),obj.cfg.eta_m*m.mean(),obj.pool_rate,obj.pool_current,np.mean(z)*obj.pool_current])
            if any(lo<=k<hi for lo,hi in p['dense_contact_windows_steps']):
                recorder=holder['recorder'];effective=obj.delivered.copy();effective[:ne]*=obj.z[:ne]
                global_i=zero.copy();global_i[:ne]=obj.z[:ne]*obj.pool_current
                raw=OriginalRecorder.sample(recorder,ie,obj.delivered)
                records['contact_time_ms'].append(k*.1);records['legacy_proxy'].append(raw.copy())
                records['effective_proxy'].append(OriginalRecorder.sample(recorder,ie,effective))
                records['excitation_proxy'].append(OriginalRecorder.sample(recorder,ie,zero))
                records['effective_global_proxy'].append(OriginalRecorder.sample(recorder,zero,global_i))
                cells=holder['cells'];counts=holder['cell_counts']
                records['native_effective_grid'].append((np.bincount(cells,weights=np.abs(ie[:ne])+np.abs(effective[:ne]),minlength=400)/counts).astype(np.float32))
                records['native_excitation_grid'].append((np.bincount(cells,weights=np.abs(ie[:ne]),minlength=400)/counts).astype(np.float32))
                records['native_global_grid'].append((np.bincount(cells,weights=global_i[:ne],minlength=400)/counts).astype(np.float32))
                holder['cached_step']=k;holder['cached_legacy']=raw
            return value
        obj.apply_currents=apply
    class Recorder(OriginalRecorder):
        def __init__(self,p,pos,labels,sites=None):
            super().__init__(p,pos,labels,sites=sites);holder['recorder']=self
            holder['cells']=source.carrier.base.old.spatial_cell_index(pos[:self.NE],n_grid=20,sheet_l_mm=p.L)
            holder['cell_counts']=np.bincount(holder['cells'],minlength=400)
            assert np.all(holder['cell_counts']>0) and holder['cell_counts'].sum()==32000
        def sample(self,ie,ii):
            if holder.get('cached_step')==holder['slow']._step_index:return holder['cached_legacy'].copy()
            return OriginalRecorder.sample(self,ie,ii)
    def capture(**kwargs):
        state=original_capture(**kwargs)
        if records['state_time_ms']:
            d=folder/'dense_chunks';d.mkdir(parents=True,exist_ok=True)
            first=round(records['state_time_ms'][0]*10);last=int(kwargs['step'])
            dest=d/f'{first:010d}_{last:010d}.npz';tmp=dest.with_suffix('.tmp.npz')
            payload={k:np.asarray(v) for k,v in records.items()};payload.update(start_step=first,end_step=last)
            np.savez_compressed(tmp,**payload);tmp.replace(dest)
            for v in records.values():v.clear()
        return state
    old_out,old_prepare=source.OUT,source.prepare
    source.OUT=out;source.prepare=lambda:p;source.GlobalPoolSlow.__init__=init
    source.carrier.base.old.LFPRecorder=Recorder;source.checkpoint.capture=capture
    try:source.worker(name)
    finally:
        source.OUT=old_out;source.prepare=old_prepare;source.GlobalPoolSlow.__init__=original_init
        source.carrier.base.old.LFPRecorder=OriginalRecorder;source.checkpoint.capture=original_capture

def verify(name):
    out=destination(name);folder=out/'runs'/name;reference=SOURCE/'runs'/name
    def load(folder):
        parts={}
        for path in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as a:
                for k in a.files:
                    if k not in ['start_step','end_step']:parts.setdefault(k,[]).append(a[k])
        return {k:np.concatenate(v) for k,v in parts.items()}
    a,b=load(folder),load(reference)
    assert a.keys()==b.keys()
    for k in a:assert np.array_equal(a[k],b[k]),k
    x=source.carrier.base.load_pickle(folder/'checkpoint.pkl')['engine'];y=source.carrier.base.load_pickle(reference/'checkpoint.pkl')['engine']
    def equal(a,b,path='checkpoint'):
        assert type(a) is type(b),path
        if isinstance(a,np.ndarray):assert np.array_equal(a,b,equal_nan=True),path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:equal(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for i,(v,w) in enumerate(zip(a,b)):equal(v,w,path+'.'+str(i))
        else:assert a==b,path
    equal(x,y)
    q=[np.load(p) for p in sorted((folder/'dense_chunks').glob('*.npz')) if '.tmp.' not in p.name]
    contacts=[v for v in q if len(v['contact_time_ms'])]
    tt=np.concatenate([v['contact_time_ms'] for v in contacts]);raw=np.concatenate([v['legacy_proxy'] for v in contacts]);eff=np.concatenate([v['effective_proxy'] for v in contacts])
    assert len(tt)>0 and np.all(np.diff(tt)>0)
    ids=np.searchsorted(tt,a['lfp_time_ms']);keep=(ids<len(tt))&(tt[np.minimum(ids,len(tt)-1)]==a['lfp_time_ms'])
    expected=np.zeros(len(ids),bool)
    legacy_steps=np.rint(a['lfp_time_ms']*10).astype(np.int64)
    for lo,hi in prepare(name)['dense_contact_windows_steps']:expected|=(legacy_steps>=lo)&(legacy_steps<hi)
    assert np.array_equal(keep,expected)
    assert np.array_equal(raw[ids[keep]],a['lfp_raw'][keep])
    state_time=np.concatenate([v['state_time_ms'] for v in q]);state=np.concatenate([v['state'] for v in q])
    indices=np.searchsorted(tt,state_time);selected=(indices<len(tt))&(tt[np.minimum(indices,len(tt)-1)]==state_time)
    counts=np.load(out/'geometry.npz')['cell_e_counts']
    grid_errors={}
    for key,expected_values in [('native_effective_grid',state[:,1]+state[:,2]),('native_excitation_grid',state[:,1]),('native_global_grid',state[:,6])]:
        grids=np.concatenate([v[key] for v in contacts]);actual=grids[indices[selected]]@counts/32000
        assert np.allclose(actual,expected_values[selected],rtol=2e-7,atol=1e-5),key
        grid_errors[key]=float(np.max(np.abs(actual-expected_values[selected])))
    source.carrier.base.write(out/'observation_qa.json',dict(status='PASS',all_legacy_observations_bitwise=True,
        entire_checkpoint_bitwise=True,including_Z_M_global_pool_and_all_noise_history=True,
        dense_legacy_proxy_matches_saved_legacy=True,post_Z_proxy_recorded_separately=True,
        contact_samples=int(len(tt)),maximum_raw_vs_effective_proxy_difference=float(np.max(np.abs(raw-eff))),
        native_field_weighted_means_match_actual_currents=True,native_field_mean_absolute_errors=grid_errors,
        note='Observation equality covers the actual saved duration only. Candidate replay must also match its own source trajectory before figure acceptance.'))
    for v in q:v.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','worker','verify']);ap.add_argument('--name',required=True);args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare(args.name)
        elif args.mode=='worker':worker(args.name)
        else:verify(args.name)
    except Exception as e:
        source.carrier.base.write(destination(args.name)/'observation_failure.json',dict(error=repr(e),time=time.time()));raise
