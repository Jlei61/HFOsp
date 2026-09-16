#!/usr/bin/env python3
"""Export frozen anchor states and trained-task readouts without new fitting."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition,FutureReadout
from src.topic5_group_event_state.v035.contracts import atomic_json


def export(source,output,device='cpu'):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    card=json.loads(source.read_text());cfg=card['config']
    if sha(card['checkpoint'])!=card['checkpoint_sha256'] or sha(card['data_path'])!=card['data_sha256']:raise ValueError('Frozen upstream changed')
    state=torch.load(card['checkpoint'],map_location='cpu',weights_only=False);data=torch.load(card['data_path'],map_location='cpu',weights_only=False)
    models={'state':EventTransition(data['input_dim'],cfg['family'],width=cfg['width'],seed=cfg['seed']),
            'initialized':EventTransition(data['input_dim'],cfg['family'],width=cfg['width'],seed=cfg['seed']),
            'fixed_history':EventTransition(data['input_dim'],'F')}
    models['state'].load_state_dict(state['observer'])
    readout=FutureReadout(models['state'].width,data['n_recruitment'],0);readout.load_state_dict(state['residual']);readout.to(device).requires_grad_(False)
    for model in models.values():model.to(device).requires_grad_(False)
    samples=data['samples'];key=str(float(cfg['history_hours']));out={k:[] for k in models};out['functional']=[]
    with torch.no_grad():
        for start in range(0,len(samples),256):
            batch=samples[start:start+256];length=max(len(s['histories'][key][0]) for s in batch)
            x=np.zeros((len(batch),length,data['input_dim']),np.float32);dt=np.zeros((len(batch),length),np.float32)
            for i,s in enumerate(batch):
                a,b=s['histories'][key];x[i,:len(a)]=a;dt[i,:len(b)]=b
            x=torch.from_numpy(x).to(device);dt=torch.from_numpy(dt).to(device)
            for name,model in models.items():
                value=model.scan(x,dt,checkpoint_chunk=0);out[name].append(value.cpu().numpy())
                if name=='state':
                    forecast=model.advance(value,2.) if cfg['forecast']=='rollout' else value
                    mu,logits=readout(forecast,forecast.new_empty((len(forecast),0)),2.)
                    functional=mu[:,None] if cfg['view']=='count' else logits if cfg['view']=='recruitment' else torch.cat((mu[:,None],logits),dim=-1)
                    out['functional'].append(functional.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in out.items()}
    arrays['background']=np.array([s['context'] for s in samples],np.float32)
    arrays['anchor_time']=np.array([s['anchor'] for s in samples]);arrays['phase']=np.array([s['phase'] for s in samples])
    if any(not np.isfinite(v).all() for k,v in arrays.items() if k!='phase'):raise FloatingPointError('Nonfinite frozen export')
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    if sha(card['checkpoint'])!=card['checkpoint_sha256']:raise ValueError('Export modified frozen observer')
    meta=dict(status='COMPLETE',subject=card['subject'],family=card['family'],history_hours=cfg['history_hours'],view=cfg['view'],seed=cfg['seed'],
        source_card=str(source),source_card_sha256=sha(source),checkpoint_sha256=card['checkpoint_sha256'],human_data_sha256=card['data_sha256'],
        export=str(output),export_sha256=sha(output),shapes={k:list(v.shape) for k,v in arrays.items()},
        functional_definition='Selected trained-view residual readout at 2-hour lead, with frozen autonomous rollout only when registered; excludes background prediction. Raw latent transfer is a separate estimand.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/topic5_group_event_state/v039/transition.py']},
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False)
    atomic_json(output.with_suffix('.json'),meta);print(json.dumps(dict(status='COMPLETE',subject=card['subject'],family=card['family'],view=cfg['view'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();export(a.source,a.output,a.device)
