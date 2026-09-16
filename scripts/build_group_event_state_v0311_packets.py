#!/usr/bin/env python
"""Build rich minute packets for one v0.3.11 subject."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.topic5_group_event_state.v0311.packets import build_subject

SOURCE='/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired'
ROOT='/data/hfosp_group_event_state_rich_event_identification_v0311'

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--subject',required=True)
    ap.add_argument('--source-root',default=SOURCE)
    ap.add_argument('--out-dir',default=ROOT+'/packets')
    a=ap.parse_args()
    t0=time.time()
    p=build_subject(a.subject,a.source_root,a.out_dir)
    summary=dict(subject=a.subject,seconds=round(time.time()-t0,1),
                 n_events=int(len(p['event_time'])),n_packets=int(len(p['packets']['start'])),
                 n_blocks=len(p['blocks']),n_contacts=p['n_contacts'],shafts=p['shafts'],
                 observed_hours=float((p['observed_support'][:,1]-p['observed_support'][:,0]).sum()/3600),
                 contact_feature_names=p['contact_feature_names'],
                 event_feature_names=p['event_feature_names'],
                 release_delay_seconds=dict(
                     median=float(sorted(b['release']-b['start'] for b in p['blocks'])[len(p['blocks'])//2]),
                     max=float(max(b['release']-b['start'] for b in p['blocks']))))
    Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    (Path(a.out_dir)/f'{a.subject}.build.json').write_text(json.dumps(summary,indent=1,default=str))
    print(json.dumps(summary,default=str)[:900])
