"""Frozen IED readout versus the separately labelled clinical-onset spatial field."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from .engine import infer_asof
from . import data as D
from .train import tensor_hash

SOURCE=Path('/home/honglab/leijiaxin/HFOsp/results')
CACHE=SOURCE/'topic5_ictal_recruitment/t0_feature_cache_bb150_1_150'


def inventory_crosswalk(subject,rows,seizures):
    """Cache index follows clinical-onset-sorted extraction; join patient cases by ID."""
    sid=subject.split('_')[-1]
    ordered=sorted([r for r in rows if r.get('subject')==sid and r.get('clin_onset_epoch')],key=lambda r:float(r['clin_onset_epoch']))
    by_id={str(r['seizure_id']):s for s,r in enumerate(ordered)};result=[]
    for seizure in seizures:
        sid0=str(seizure['seizure_id']);i=by_id.get(sid0)
        if i is None:continue
        inv=ordered[i]
        if not inv.get('eeg_onset_epoch') or abs(float(inv['eeg_onset_epoch'])-float(seizure['onset_epoch']))>1e-3:
            raise ValueError('EEG onset inventory disagrees with interictal exclusion source')
        result.append(dict(source_index=i,seizure_id=sid0,eeg_onset=float(inv['eeg_onset_epoch']),clinical_onset=float(inv['clin_onset_epoch'])))
    return result


@torch.no_grad()
def clinical_spatial_association(model,prep,cfg,adapter_dir,source_root=SOURCE):
    root=Path(source_root);cache=root/'topic5_ictal_recruitment/t0_feature_cache_bb150_1_150'
    inventory=root/'epilepsiae_seizure_inventory.csv'
    if not inventory.exists():inventory=root/'dataset_inventory/epilepsiae_seizure_inventory.csv'
    npz=cache/f'{cfg.subject}.npz';side=cache/f'{cfg.subject}.json';head=Path(adapter_dir)/'adapter_state_history_1min.pt'
    missing=[str(p) for p in (inventory,npz,side,head) if not p.exists()]
    if missing:return dict(status='NOT_AVAILABLE',missing=missing)
    meta=json.loads(side.read_text())
    if meta['t_window']!=[0.,10.] or meta['band_broad_1_150']!=[1.,150.]:raise ValueError('ictal measurement contract changed')
    adapter=torch.load(head,weights_only=False,map_location=prep.device)
    if adapter['status']!='COMPLETE':return dict(status='NOT_ESTIMABLE',reason='no qualified frozen IED identity adapter')
    controls={}
    for kind in ('trait','history','state'):
        path=Path(adapter_dir)/f'adapter_{kind}_1min.pt'
        if path.exists():
            control=torch.load(path,weights_only=False,map_location=prep.device)
            if control['status']=='COMPLETE':controls[kind]=(control,path)
    with open(inventory) as f:cross=inventory_crosswalk(cfg.subject,list(csv.DictReader(f)),prep.split['seizures'])
    # Filter by already-authorized physical support before reading any outcome array.
    cross=[r for r in cross if prep.split['support_start']<=r['eeg_onset'] and r['clinical_onset']+10<=prep.split['support_end']]
    pk=prep.payload['packets'];allowed=D.input_mask(prep.split,'descriptive');rows=[]
    ep=D.event_packets(prep.payload);fit=prep.split['train_packet'][ep];pr=(prep.part[fit].sum(0)+.5)/(fit.sum()+1.);trait=torch.logit(pr).cpu().numpy()
    with np.load(npz,allow_pickle=False) as zz:
        names=zz['channels'].astype(str).tolist();lookup={n:i for i,n in enumerate(names)}
        join=np.array([lookup.get(n,-1) for n in prep.payload['selected_contacts']]);ok=join>=0
        for r in cross:
            key=f"bb150_auc__{r['source_index']}"
            if key not in zz:continue
            q=int(np.searchsorted(pk['end'],r['eeg_onset']-300,side='right')-1)
            if q<0 or prep.split['seizure_mask'][q]:continue
            st=infer_asof(model,prep,[q],'descriptive',cfg.history_hours,producer_hash=tensor_hash(model.state_dict()))
            if not np.isfinite(st.release_time[0]):continue
            tt=pk['end'][q];ix=np.flatnonzero(allowed&(pk['release']<=tt)&(pk['end']<=tt)&(pk['end']>tt-7200)&(np.arange(prep.n_packets)>=st.prefix_start[0]))
            ei=np.concatenate([np.arange(pk['event_lo'][i],pk['event_hi'][i]) for i in ix]) if len(ix) else np.empty(0,int)
            cnt=prep.part[ei].sum(0) if len(ei) else prep.part.new_zeros(prep.part.shape[1])
            hist=torch.logit(((cnt+.5)/(len(ei)+1.)).clamp(1e-4,1-1e-4))
            clock=prep.clock[min(q+1,prep.n_packets-1)]
            features=torch.cat((st.m[0],hist,clock))
            f=adapter['fitted'];x=(features-f['center'])/f['scale'];weights=f['state_dict']
            logits=(x@weights['weight'].T+weights['bias']).cpu().numpy()
            target=np.full(len(join),np.nan);target[ok]=np.asarray(zz[key])[join[ok]];valid=ok&np.isfinite(target)
            scores={}
            predictions=[('IED_fit_trait',trait),('published_recent_identity',hist.cpu().numpy()),('frozen_IED_state_history',logits)]
            for kind,(control,_) in controls.items():
                features={'trait':clock,'history':torch.cat((hist,clock)),'state':torch.cat((st.m[0],clock))}[kind]
                f0=control['fitted'];x0=(features-f0['center'])/f0['scale'];w0=f0['state_dict']
                predictions.append((f'frozen_IED_{kind}',(x0@w0['weight'].T+w0['bias']).cpu().numpy()))
            for name,pred in predictions:
                keep=valid&np.isfinite(pred)
                scores[name]=float(spearmanr(pred[keep],target[keep]).statistic) if keep.sum()>=3 and np.std(pred[keep])>1e-9 and np.std(target[keep])>1e-9 else None
            rows.append(dict(**r,query_time=float(tt),EEG_to_clinical_seconds=r['clinical_onset']-r['eeg_onset'],n_contacts=int(valid.sum()),
                scores=scores,channels=prep.payload['selected_contacts'],target=target,trait=trait,state_history_logits=logits,
                query_metadata=st.metadata(),after_producer_fit=r['eeg_onset']>prep.split['fit_end']))
    return dict(status='DESCRIPTIVE' if rows else 'NOT_ESTIMABLE',rows=rows,
        measurement='1-150 Hz baseline robust-z activation, clinical onset [0,10] seconds, CAR',
        interpretation='Frozen interictal contact-readout spatial association. Not EEG-onset 0-10s forecasting, not an ictal-trained decoder, not propagation.',
        fitted_control_status={k:('COMPLETE' if k in controls else 'NOT_ESTIMABLE') for k in ('trait','history','state')},
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (inventory,npz,side,head,*[v[1] for v in controls.values()])})
