#!/usr/bin/env python3
"""Compare saved float32 scores with the independent float64 event replay."""
import argparse,hashlib,json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.topic5_group_event_state.v035.contracts import atomic_json


def audit(root,output):
    if output.exists():raise FileExistsError(output)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[];groups=defaultdict(list)
    paths=sorted((root/'human_gradient_audit').glob('*/card.json'))+sorted((root/'view_gradient_audit').glob('*/card.json'))
    if len(paths)!=72:raise ValueError('Wait for all 72 real-event gradient replays')
    for p in paths:
        g=json.loads(p.read_text());source=Path(g['source_card']);c=json.loads(source.read_text())
        if sha(source)!=g['source_card_sha256'] or sha(c['scores'])!=c['scores_sha256']:raise ValueError('Frozen source or score changed')
        with np.load(c['scores']) as z:
            for v in g['records']:
                index=np.flatnonzero(z['anchor_time']==v['anchor'])
                if len(index)!=1:raise ValueError('Replay score anchor is not unique')
                stored=float(z['2h_state_loss'][index[0]]);delta=abs(stored-v['cached_loss'])
                groups[g['subject']].append(delta)
                rows.append(dict(subject=g['subject'],family=g['family'],view=c['config']['view'],history_hours=g['history_hours'],seed=g['seed'],
                    source=str(source),source_sha256=sha(source),gradient_audit=str(p),gradient_audit_sha256=sha(p),anchor=v['anchor'],
                    stored_float32_loss=stored,replayed_float64_loss=v['cached_loss'],absolute_difference=delta))
    atomic_json(output,dict(status='COMPLETE',rows=rows,by_subject={s:dict(n_replayed_windows=len(v),median_absolute_difference=float(np.median(v)),maximum_absolute_difference=max(v)) for s,v in groups.items()},
        interpretation='Observed numerical differences at the registered replay windows, not a whole-dataset error bound. Same-precision independent input/loss replay and finite differences are separate tests. No post-hoc pass threshold is fitted to these differences; tiny effects are not promoted as independent scientific support.',
        source_sha256=sha(__file__),development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
    print(json.dumps(dict(status='COMPLETE',replayed_windows=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)
