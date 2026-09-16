"""Freeze event-level marks, exact times, exposure, and seizure boundaries."""
import sys, hashlib, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scripts.patient_state_v1.common import ROOT,RUN,write_json
from scripts.explore_e1146_seizure_interictal_association import load_events, outside_seizures
from scripts.analyze_e1146_source_signed_correspondence import FIELD, BASE
from scripts.analyze_e1146_preseizure_template_share import SQL

def merge_ranges(ranges):
    out=[]
    for a,b in sorted(ranges):
        if out and a<=out[-1][1]+1e-6: out[-1][1]=max(b,out[-1][1])
        else: out.append([a,b])
    return out

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    inv,starts,ends,labels,segments,ranges=load_events()
    keep=outside_seizures(starts,ends,inv)
    origin=min(a for a,b in ranges)
    s,e,y,seg=starts[keep],ends[keep],labels[keep],segments[keep]
    assert set(np.unique(y))=={0,1} and np.all(e>s)
    offsets=np.array([r['offset'] for r in inv]);onsets=np.array([r['onset'] for r in inv])
    epoch=np.searchsorted(offsets,s,side='right')
    # Separate inference across ictal periods; ordinary observation gaps are propagated.
    reset=np.r_[True,epoch[1:]!=epoch[:-1]]
    pd.DataFrame(dict(start_epoch=s,end_epoch=e,t_hours=(s-origin)/3600,label_tb=y,
                      coverage_segment=seg,interictal_epoch=epoch,reset_prior=reset)).to_csv(RUN/'events.csv',index=False)
    # Simultaneous marks share a single latent state and a binomial likelihood.
    new=np.r_[True,(np.diff(s)>1e-8)|(np.diff(epoch)!=0)]
    ix=np.flatnonzero(new); counts=np.diff(np.r_[ix,len(s)]);tb=np.add.reduceat(y,ix)
    t=(s[ix]-origin)/3600; epochs=epoch[ix];res=np.r_[True,np.diff(epochs)!=0]
    dt=np.r_[0,np.diff(t)];dt[res]=0
    assert np.all(dt[~res]>0) and counts.sum()==len(s) and tb.sum()==y.sum()
    phase=2*np.pi*(s[ix]/3600)/24
    np.savez_compressed(RUN/'observations.npz',t=t,dt=dt,n=counts,y=tb,reset=res,epoch=epochs,
                        x=np.column_stack([np.ones(len(t)),np.sin(phase),np.cos(phase)]),origin_epoch=origin)
    merged=merge_ranges(ranges);exposure=[]
    for a,b in merged:
        pieces=[[a,b]]
        for seiz in inv:
            nxt=[]
            for lo,hi in pieces:
                if hi<=seiz['onset'] or lo>=seiz['offset']:nxt.append([lo,hi])
                else:
                    if lo<seiz['onset']:nxt.append([lo,seiz['onset']])
                    if hi>seiz['offset']:nxt.append([seiz['offset'],hi])
            pieces=nxt
        exposure.extend(pieces)
    pd.DataFrame(exposure,columns=['start_epoch','end_epoch']).to_csv(RUN/'exposure.csv',index=False)
    write_json(RUN/'seizures.json',inv)
    windows=ROOT/'results/topic5_preseizure_template_association_broadband/cohort_20260909/per_subject/epilepsiae_1146/association/window_observations.csv'
    w=pd.read_csv(windows)
    for r in w.itertuples():
        m=(s>=r.start_epoch)&(e<=r.end_epoch)&(s<r.end_epoch)
        assert int(y[m].sum())==r.n_tb and int(m.sum()-y[m].sum())==r.n_ta
    w.to_csv(RUN/'frozen_seizure_windows.csv',index=False)
    # Cut at interictal-epoch boundaries nearest 40/60/80% elapsed time.
    boundaries=np.flatnonzero(res);bt=t[boundaries]
    cuts=[]
    for f in (.4,.6,.8):
        target=t[0]+f*(t[-1]-t[0]);valid=boundaries[(bt>t[0])&(bt<t[-1])]
        cuts.append(int(valid[np.argmin(abs(t[valid]-target))]))
    cuts=sorted(set(cuts));assert len(cuts)>=2
    folds=[]
    for i,start in enumerate(cuts):
        end=cuts[i+1] if i+1<len(cuts) else len(t)
        folds.append(dict(fold=i,train_end=start,test_start=start,test_end=end,
                          train_hours=[float(t[0]),float(t[start-1])],test_hours=[float(t[start]),float(t[end-1])]))
    write_json(RUN/'splits.json',folds)
    sources=[FIELD,BASE/'event_index.npz',SQL,windows,Path(__file__)]
    summary=dict(status='PREPARED',n_source_events=len(starts),n_interictal_events=len(s),n_ta=int((y==0).sum()),
                 n_tb=int(y.sum()),n_distinct_times=len(t),n_simultaneous_groups=int((counts>1).sum()),
                 max_simultaneous=int(counts.max()),n_epochs=len(np.unique(epochs)),n_coverage_segments=len(merged),
                 span_hours=(ends.max()-origin)/3600,observed_interictal_hours=sum(b-a for a,b in exposure)/3600,
                 event_gap_seconds_quantiles=np.quantile(dt[~res]*3600,[0,.01,.5,.9,.99,1]),
                 n_seizures=len(inv),n_broadband_qualified=12,n_quoted_tb_seizures=2,
                 frozen_window_counts_match=True,created_unix=time.time(),
                 sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                 scope='all available fixed-label interictal events; no claim of all raw monitoring',
                 reset_semantics='shared marginal initial prior across ictal exclusions; no physiological reset claim')
    write_json(RUN/'data_audit.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
