#!/usr/bin/env python3
"""Audit the frozen patient masks using training blocks only; heldout payloads unopened."""
from pathlib import Path
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.topic4_xy_fig5_followup import read,write,sha
from src.topic4_xy_readout_audit import support_summary
from scripts.run_topic4_xy_research import OUT,ART,training_contract


def shape_npy(stream):
    version=np.lib.format.read_magic(stream)
    return np.lib.format._read_array_header(stream,version)[0]


def main():
    config=read(ROOT/'config/topic4_rev10_sa_shaft_aware.json')
    root=Path(config['inputs']['patient_root'])
    files=sorted(root.glob('*_lagPat_withFreqCent.npz'))
    if not files: files=sorted(root.glob('*_lagPat.npz'))
    # Header shapes and recording timestamps only for block indexing. No heldout event values.
    meta=[]
    for p in files:
        with zipfile.ZipFile(p) as z:
            shapes=[]
            for key in ('lagPatRank','lagPatRaw','eventsBool'):
                with z.open(key+'.npy') as stream: shapes.append(shape_npy(stream))
        with np.load(p,allow_pickle=True) as z: start=float(np.asarray(z['start_t']).reshape(-1)[0])
        if any(len(s)!=2 or min(s)==0 for s in shapes):continue
        n=min(s[1] for s in shapes)
        rec=p.name.replace('_lagPat_withFreqCent.npz','').replace('_lagPat.npz','')
        suffix='_packedTimes_withFreqCent.npy' if p.name.endswith('_withFreqCent.npz') else '_packedTimes.npy'
        packed=p.with_name(rec+suffix)
        if packed.exists():
            with open(packed,'rb') as stream: ps=shape_npy(stream)
            n=min(n,ps[0])
        meta.append(dict(path=str(p),start=start,n=n,packed=str(packed)))
    meta.sort(key=lambda r:(r['start'] if np.isfinite(r['start']) else float('inf'),Path(r['path']).name))
    source=ART/'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/shaft_aware_patient_training_target.npz'
    with np.load(source) as z:
        frozen=z['patient_train_onsets'].astype(float)
        indices=z['patient_train_event_indices'];blocks=z['patient_train_block_ids'];names=z['contact_names'].astype(str)
    train_blocks=set(blocks.tolist()); offset=0; all_train=[]; matched=[]; raw_windows=[]; audits=[]
    for block,r in enumerate(meta):
        if block in train_blocks:
            with np.load(r['path'],allow_pickle=True) as z:
                source_names=z['chnNames'].astype(str).tolist()
                order=[source_names.index(n) for n in names]
                mask=np.asarray(z['eventsBool'])[order,:r['n']]>0
                raw=np.asarray(z['lagPatRaw'],float)[order,:r['n']]
            selected=np.flatnonzero(blocks==block)
            local=indices[selected]-offset
            if not np.all((local>=0)&(local<r['n'])):raise RuntimeError('block/event index drift')
            x=raw.T.copy();x[~mask.T]=np.nan
            y=x[local]
            if not np.array_equal(np.isfinite(y),np.isfinite(frozen[selected])):
                raise RuntimeError('frozen patient participation mask differs from raw training block')
            if not np.allclose(y,frozen[selected],rtol=1e-6,atol=1e-7,equal_nan=True):
                raise RuntimeError('frozen patient centroid times differ from training source')
            all_train.append(x);matched.append(y)
            packed=Path(r['packed'])
            if packed.exists():
                a=np.load(packed)[:r['n']]
                if a.ndim==2 and a.shape[1]>=2:raw_windows.extend(((a[:,1]-a[:,0])*1000).tolist())
            audits.append({'block_id':block,'source':r['path'],'sha256':sha(r['path']),
                           'n_raw_training_events':r['n'],'n_frozen_events':len(local),'mask_parity':True,'centroid_parity':True})
        offset+=r['n']
    training,obj=training_contract()
    full=np.concatenate(all_train);chosen=np.concatenate(matched)
    report={'status':'PATIENT_TRAINING_MASK_PARITY_VERIFIED','n_indexed_blocks':len(meta),
            'n_training_blocks_read':len(audits),'heldout_event_payloads_opened':False,
            'metadata_only_for_all_blocks':'NPY array shapes and start_t timestamps only to recover frozen block order',
            'source_contract_sha256':sha(source),
            'raw_training':support_summary(full*1000,training['groups'],training['pairs']),
            'frozen_readable_training':support_summary(chosen*1000,training['groups'],training['pairs']),
            'excluded_by_readable_training_filter':len(full)-len(chosen),
            'packed_window_ms_q05_q50_q95':np.quantile(raw_windows,[.05,.5,.95]).tolist(),
            'patient_time_semantics':'lagPatRaw spectrogram centroids in seconds, masked by eventsBool; converted to ms. Model timing uses half-peak first crossings.',
            'training_blocks':audits}
    write(OUT/'readout_support_audit/patient_training_mask_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('training_blocks','raw_training','frozen_readable_training')},indent=2))
    for k in ('raw_training','frozen_readable_training'):
        print(k,{s:report[k][s] for s in ('n_events','both_shafts_fraction','mean_contacts','median_contacts')})


if __name__=='__main__':main()
