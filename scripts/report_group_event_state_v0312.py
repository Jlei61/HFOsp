#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.topic5_group_event_state.v0312.report import summarize
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=summarize(a.plan,a.out)
 print(json.dumps({k:r[k] for k in ('status','n_complete','n_total','scientific_status')}))
