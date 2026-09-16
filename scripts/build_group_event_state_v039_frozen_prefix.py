#!/usr/bin/env python3
"""Freeze a calibration-selected decoder and expose exactly two input groups."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder
from src.topic5_wiring_economy_rnn import build_event_tensors
from src.topic5_group_event_state.v035.contracts import atomic_json


def build(subject,root,output,device):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cards=[(p,json.loads(p.read_text())) for p in sorted((root/'decoder_rebuilt'/subject/'cards').glob('seed*.json'))]
    if len(cards)!=3:raise ValueError('Freeze decoder only after all three registered calibration seeds finish')
    def score(row):
        m=row[1]['metrics']['validation'];return m['next_bce']+m['stop_bce'],row[1]['seed']
    source,card=min(cards,key=score)
    unit=Path(card['unit_dir']);cache=Path(card['cache_dir'])
    if sha(unit/'weights.pt')!=card['checkpoint_sha256']:raise ValueError('Changed decoder checkpoint')
    bundle=load_frozen_decoder(unit,cache,device=torch.device(device));model=bundle.model
    target=root/'transfer_data'/f'{subject}.npz';meta=json.loads(target.with_suffix('.json').read_text())
    if sha(target)!=meta['data_sha256']:raise ValueError('Changed expression pairs')
    with np.load(target) as z:
        ranks=z['ranks'];names=z['contact_names'].astype(str);times=z['event_time'];anchors=z['anchor_position']
    if tuple(names)!=bundle.contact_names:raise ValueError('Contact identity/order mismatch')
    tensors=build_event_tensors(ranks);logits=[];stops=[];hidden=[]
    with torch.no_grad():
        for first in range(0,len(ranks),512):
            x=tensors['x'][first:first+512,:2].to(device);recruited=tensors['recruited'][first:first+512,:2].to(device)
            h=x.new_zeros((len(x),model.n_nodes*model.state_dim))
            for step in range(2):h=model._step(h,x[:,step])
            pred=model._readout(h);stop=model._stop(h,torch.full((len(x),),1/max(1,model.n_contacts-1),device=device),recruited[:,1].mean(-1))
            logits.append(pred.cpu().numpy());stops.append(stop.cpu().numpy());hidden.append(h.cpu().numpy())
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,logits=np.concatenate(logits),stops=np.concatenate(stops),hidden=np.concatenate(hidden),event_time=times,anchor_position=anchors)
    record=dict(status='COMPLETE',subject=subject,data_path=str(output),data_sha256=sha(output),selected_decoder_card=str(source),selected_decoder_card_sha256=sha(source),
        decoder_checkpoint_sha256=card['checkpoint_sha256'],selected_seed=card['seed'],decoder_selection='minimum calibration validation next BCE + STOP BCE across all 3 registered seeds',
        candidates=[dict(seed=c['seed'],validation_score=score((p,c))[0],source=str(p)) for p,c in cards],
        transfer_data_path=str(target),transfer_data_sha256=meta['data_sha256'],observed_groups=2,future_event_groups_consumed=0,
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,source_sha256=sha(__file__))
    atomic_json(output.with_suffix('.json'),record);print(json.dumps(dict(status='COMPLETE',subject=subject,selected_decoder_seed=card['seed'],events=len(ranks))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();build(a.subject,a.root,a.output,a.device)
