#!/usr/bin/env python
"""S-A: frozen functional state near seizures, against matched within-patient controls.

Seizure labels only select time windows; they never train the producer. The
independent unit is the seizure cluster. A matched case-control contrast is not
an absolute risk.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch

from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311.train import RunConfig,build_run,HORIZON_PACKETS
from src.topic5_group_event_state.v0311 import frozen as FZ

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
PRIMARY=(-7200.,-1800.);SECONDARY=(-1800.,-300.);CLUSTER_GAP=7200.
GRID=1800.


def clusters(seizures,gap=CLUSTER_GAP):
    on=sorted(float(s['onset_epoch']) for s in seizures)
    out=[]
    for t in on:
        if out and t-out[-1][-1]<gap:out[-1].append(t)
        else:out.append([t])
    return out


@torch.no_grad()
def readout_summary(model,prep,cfg,packets,paths=64):
    m,P,kept=FZ.states_at_queries(model,prep,cfg,packets)
    if m is None or len(kept)==0:return None
    clock=prep.clock[torch.as_tensor(kept,device=prep.device)]
    out=model.readout(m,clock)
    v=torch.diagonal(P,dim1=-2,dim2=-1)
    recent=np.array([float(prep.count[max(0,k-30):k].sum()) for k in kept])
    L=cfg.warm_packets+cfg.grad_packets
    # Higher state uncertainty before a seizure could simply mean less data was
    # readable in the prefix; that has to be measured, not assumed away.
    prefix=np.array([float(prep.valid[max(0,k-L+1):k+1].mean()) for k in kept])
    prefix_read=np.array([float((prep.valid[max(0,k-L+1):k+1]
                                 *(1.-prep.seizure_masked[max(0,k-L+1):k+1])).mean()) for k in kept])
    return dict(packet=kept,recent_30min_count=recent,prefix_exposure_fraction=prefix,
                prefix_readable_fraction=prefix_read,log_rate=out['log_rate'].cpu().numpy(),
                composition_entropy=(-torch.softmax(out['composition'],-1)
                                     *torch.log_softmax(out['composition'],-1)).sum(-1).cpu().numpy(),
                morphology_scale=out['band_ratio_logsd'].mean(-1).cpu().numpy(),
                delay_zero_logit=out['iqr'][...,0].cpu().numpy(),
                state_norm=m.norm(dim=-1).cpu().numpy(),
                posterior_trace=v.sum(-1).cpu().numpy())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--card',required=True);ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args()
    card=json.loads(Path(a.card).read_text())
    cfg=RunConfig(**{**card['config'],'device':a.device})
    if cfg.arm!='state':print(json.dumps(dict(status='SKIPPED')));return
    payload=torch.load(f'{cfg.packets_root}/{cfg.subject}.pt',weights_only=False)
    split=D.build_split(payload,cfg.subject,cfg.seed) if cfg.split=='S-E' else D.build_split_id(payload,cfg.subject,cfg.seed)
    px,pt,_=D.packet_tables(payload,split);scaling=D.fit_scaling(payload,split,px,pt)
    dev=torch.device(a.device);prep=Prepared(payload,split,scaling,dev)
    prep.payload=payload
    model,_=build_run(cfg,payload,split,scaling,prep)
    ck=torch.load(Path(a.card).with_suffix('').with_suffix('.ckpt.pt'),weights_only=False,map_location=dev)
    model.load_state_dict(ck['state_dict']);model.eval()
    L=cfg.warm_packets+cfg.grad_packets;pe=prep.packet_end
    def legal(times):
        idx=np.searchsorted(pe,np.asarray(times,float)-1e-6)
        keep=[int(i) for i in idx if 0<=i-L+1 and i<prep.n_packets and not prep.seizure_masked_np[i]
              and prep.valid_np[i] and not prep.seizure_masked_np[max(0,i-L+1):i+1].all()]
        return np.array(sorted(set(keep)),np.int64)
    cl=clusters(split['seizures'])
    cutoff=split.get('se_cutoff')
    rows=[];windows=[]
    for ci,c in enumerate(cl):
        onset=c[0]
        for name,(lo,hi) in (('primary_-2h_-30min',PRIMARY),('secondary_-30min_-5min',SECONDARY)):
            t=np.arange(onset+lo,onset+hi,GRID) if hi-lo>=GRID else np.array([onset+lo])
            p=legal(t)
            windows.append(dict(cluster=ci,onset=onset,window=name,n_requested=len(t),n_legal=int(len(p)),
                                after_producer_cutoff=bool(cutoff is None or onset>cutoff)))
            if len(p):rows.append((ci,name,p))
    used=np.concatenate([r[2] for r in rows]) if rows else np.empty(0,np.int64)
    # Controls: same patient, matched clock stratum, physically outside any pre-ictal window
    all_pre=set()
    for c in cl:
        for lo,hi in (PRIMARY,SECONDARY):
            for t in np.arange(c[0]+lo-GRID,c[0]+hi+GRID,GRID):
                k=int(np.searchsorted(pe,t-1e-6))
                all_pre.add(k)
    grid=np.arange(split['support_start'],split['support_end'],GRID)
    ctrl=[k for k in legal(grid) if k not in all_pre]
    ctrl=np.array(ctrl,np.int64)
    pre=readout_summary(model,prep,cfg,used) if len(used) else None
    con=readout_summary(model,prep,cfg,ctrl) if len(ctrl) else None
    result=dict(subject=cfg.subject,split=cfg.split,inputs=cfg.inputs,family=cfg.family,
                card=str(a.card),n_seizure_clusters=len(cl),windows=windows,
                n_control_queries=int(len(ctrl)),n_preictal_queries=int(len(used)))
    if pre is None or con is None:
        result['status']='NOT_ESTIMABLE'
        result['reason']='no legal pre-ictal or control query after prefix and exclusion rules'
    else:
        stratum_c=D.clock_stratum(pe[con['packet']])
        edges=np.quantile(con['recent_30min_count'],[0.2,0.4,0.6,0.8])
        band_c=np.digitize(con['recent_30min_count'],edges)
        per_cluster=[]
        offset=0
        for ci,name,p in rows:
            sel=slice(offset,offset+len(p));offset+=len(p)
            entry=dict(cluster=ci,window=name,n=len(p),
                       preictal_recent_30min_count_median=float(np.median(pre['recent_30min_count'][sel])))
            st=np.unique(D.clock_stratum(pe[pre['packet'][sel]]))
            bd=np.unique(np.digitize(pre['recent_30min_count'][sel],edges))
            clock_m=np.isin(stratum_c,st)
            rate_m=clock_m&np.isin(band_c,bd)
            entry['n_clock_matched_controls']=int(clock_m.sum())
            entry['n_clock_and_rate_matched_controls']=int(rate_m.sum())
            for k in ('log_rate','composition_entropy','morphology_scale','delay_zero_logit',
                      'state_norm','posterior_trace','recent_30min_count',
                      'prefix_exposure_fraction','prefix_readable_fraction'):
                row={}
                for label,m in (('clock_matched',clock_m),('clock_and_recent_rate_matched',rate_m)):
                    if m.sum()<3:row[label]=None;continue
                    row[label]=dict(preictal_median=float(np.median(pre[k][sel])),
                                    control_median=float(np.median(con[k][m])),
                                    difference=float(np.median(pre[k][sel])-np.median(con[k][m])),
                                    n_controls=int(m.sum()))
                entry[k]=row
            per_cluster.append(entry)
        result['status']='COMPLETE';result['per_cluster']=per_cluster
        result['control_matching']=('controls are matched on clock stratum, and separately on clock '
                                   'stratum plus the observed recent 30-minute count, so a state '
                                   'difference can be separated from a plain rate difference')
        result['interpretation']=('within-patient matched case-control association of a frozen functional '
                                  'state; the independent unit is the seizure cluster and no absolute risk '
                                  'is implied. Clusters before the producer cutoff are retrospective.')
    p=ROOT/'seizure'/(Path(a.card).name.replace('.card.json','.seizure.json'))
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=1,default=str))
    print(json.dumps({k:result[k] for k in ('subject','status','n_seizure_clusters',
                                            'n_preictal_queries','n_control_queries')},default=str))

if __name__=='__main__':main()
