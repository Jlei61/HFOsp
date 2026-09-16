#!/usr/bin/env python
"""Predictive calibration of a frozen checkpoint on the held-out forward segment.

Sharpness without calibration is not prediction: this reports the randomised PIT
of the observed count under the full predictive mixture, plus a reliability
curve of predicted against observed counts.
"""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch

from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311.train import (RunConfig,build_run,HORIZON_PACKETS,
                                                      PACKET_HOURS,_sample_paths,apply_ablations)
from src.topic5_group_event_state.v0311.numerics import propagate_samples
from src.topic5_group_event_state.v0311 import frozen as FZ
from src.topic5_group_event_state.v0311.model import count_log_prob

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')


@torch.no_grad()
def calibrate(model,prep,cfg,packets,paths=64,kmax=4000):
    m,P,kept=FZ.states_at_queries(model,prep,cfg,packets)
    if m is None:return {}
    g=torch.Generator(device=prep.device).manual_seed(int(cfg.seed)+11)
    z=_sample_paths(m,P,paths,g)
    base=torch.as_tensor(kept,device=prep.device)
    cache={};out={}
    rng=np.random.default_rng(int(cfg.seed))
    for step in range(1,max(HORIZON_PACKETS)+1):
        zp=z
        z=propagate_samples(model.dynamics,z,PACKET_HOURS,generator=g,cache=cache)
        if step not in HORIZON_PACKETS:continue
        t=base+step
        keep=(t<prep.n_packets)
        t=t[keep]
        ok=(prep.valid[t]*(1.-prep.seizure_masked[t]))>0
        t=t[ok]
        if len(t)==0:continue
        zs=zp[:,keep][:,ok];ze=z[:,keep][:,ok]
        clock=prep.clock[t].unsqueeze(0).expand(zs.shape[0],len(t),2)
        rs=model.readout(zs,clock)['log_rate'];re=model.readout(ze,clock)['log_rate']
        expo=prep.exposure_hours[t].unsqueeze(0)
        obs=prep.count[t]
        ks=torch.arange(0.,kmax,device=prep.device).reshape(-1,1,1)
        lp=count_log_prob(rs.unsqueeze(0).expand(kmax,-1,-1),re.unsqueeze(0).expand(kmax,-1,-1),
                          expo.unsqueeze(0),ks,model.readout.log_nb_dispersion)
        mix=torch.logsumexp(lp,dim=1)-math.log(lp.shape[1])
        pmf=torch.exp(mix)
        cdf=torch.cumsum(pmf,0)
        idx=obs.long().clamp(0,kmax-1)
        below=torch.where(idx>0,cdf[idx-1,torch.arange(len(t),device=prep.device)],torch.zeros_like(obs))
        at=pmf[idx,torch.arange(len(t),device=prep.device)]
        u=torch.as_tensor(rng.uniform(size=len(t)),device=prep.device,dtype=below.dtype)
        pit=(below+u*at).cpu().numpy()
        mean=(pmf*ks.reshape(-1,1)).sum(0).cpu().numpy()
        o=obs.cpu().numpy()
        edges=np.quantile(mean,np.linspace(0,1,6))
        rel=[]
        for a,b in zip(edges[:-1],edges[1:]):
            sel=(mean>=a)&(mean<=b)
            if sel.sum()>=5:rel.append(dict(predicted_mean=float(mean[sel].mean()),
                                            observed_mean=float(o[sel].mean()),n=int(sel.sum())))
        h=np.histogram(pit,bins=10,range=(0,1))[0]
        out[step]=dict(n=len(t),pit_histogram=h.tolist(),
                       pit_uniformity_chi2=float(((h-h.mean())**2/max(h.mean(),1e-9)).sum()),
                       pit_mean=float(pit.mean()),reliability=rel,
                       total_pmf_mass=float(cdf[-1].mean()))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--card',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--paths',type=int,default=64)
    a=ap.parse_args()
    card=json.loads(Path(a.card).read_text())
    cfg=RunConfig(**{**card['config'],'device':a.device})
    if cfg.arm!='state':print(json.dumps(dict(status='SKIPPED')));return
    payload=torch.load(f'{cfg.packets_root}/{cfg.subject}.pt',weights_only=False)
    split=D.build_split(payload,cfg.subject,cfg.seed) if cfg.split=='S-E' else D.build_split_id(payload,cfg.subject,cfg.seed)
    px,pt,_=D.packet_tables(payload,split);scaling=D.fit_scaling(payload,split,px,pt)
    dev=torch.device(a.device);prep=Prepared(payload,split,scaling,dev)
    apply_ablations(prep,cfg)
    model,_=build_run(cfg,payload,split,scaling,prep)
    ck=torch.load(Path(a.card).with_suffix('').with_suffix('.ckpt.pt'),weights_only=False,map_location=dev)
    model.load_state_dict(ck['state_dict']);model.eval()
    L=cfg.warm_packets+cfg.grad_packets;pe=prep.packet_end
    idx=np.searchsorted(pe,np.asarray(split['forward_starts'],float)-1e-6)
    packets=np.array([int(i) for i in idx if i-L+1>=0 and i+max(HORIZON_PACKETS)<prep.n_packets
                      and not prep.seizure_masked_np[i]],np.int64)
    res=calibrate(model,prep,cfg,packets,paths=a.paths)
    out=dict(card=str(a.card),subject=cfg.subject,split=cfg.split,inputs=cfg.inputs,family=cfg.family,
             horizons=res,note=('randomised PIT of the observed count under the full predictive mixture; '
                                'a flat histogram means calibrated, a U shape means over-confident'))
    p=ROOT/'calibration'/(Path(a.card).name.replace('.card.json','.calibration.json'))
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,indent=1,default=str))
    print(json.dumps({str(k):dict(n=v['n'],pit_mean=round(v['pit_mean'],3),
                                  chi2=round(v['pit_uniformity_chi2'],1),
                                  mass=round(v['total_pmf_mass'],4)) for k,v in res.items()}))

if __name__=='__main__':main()
