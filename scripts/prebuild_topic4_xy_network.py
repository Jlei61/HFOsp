#!/usr/bin/env python3
"""Build one corrected topology, once, before dispatching paired XY candidates."""
from pathlib import Path
import argparse
import dataclasses
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'src/snn_engine'):sys.path.insert(0,str(p))
from params import Params
from src.topic4_core_field_runner import get_network,connectivity_config,cache_key,atomic_write_json
from src.topic4_xy_search import sha
from scripts.rebuild_topic4_autapse_corrected_reference import summarize


def prebuild(seed):
    audit_path=ROOT/'results/topic4_sef_hfo/substrate_autapse_correction/graph_rebuild_audit.json'
    audit=json.loads(audit_path.read_text());cfg=audit['new_cache']['config']
    kwargs={k:v for k,v in cfg.items() if k in {f.name for f in dataclasses.fields(Params)}}
    kwargs['seed']=int(seed);p=Params(**kwargs)
    cache=ROOT/'results/topic4_sef_hfo/substrate_autapse_correction/network_cache'
    net,ne,ni,hit=get_network(p,cfg['theta_EE_deg'],cfg['AR'],str(cache))
    stats=summarize(net)
    if any(r['self_edges'] or not r['exact_expected_degree'] for r in stats.values()):
        raise RuntimeError('corrected topology failed structural validation')
    config=connectivity_config(p,cfg['theta_EE_deg'],cfg['AR'])
    path=cache/(cache_key(config)+'.pkl')
    record={'path':str(path),'sha256':sha(path),'topology_seed':int(seed),
            'status':'CORRECTED_GRAPH_VALIDATED','pathways':stats,'config':config,'cache_hit':hit}
    out=ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research/network_records'
    out.mkdir(parents=True,exist_ok=True)
    atomic_write_json(record,str(out/f'{seed}.json'))
    print(json.dumps({'seed':seed,'status':record['status'],'cache_hit':hit}),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    prebuild(parser.parse_args().seed)
