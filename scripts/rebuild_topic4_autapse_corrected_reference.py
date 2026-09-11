#!/usr/bin/env python3
"""Rebuild one full-size graph with the corrected sampler, preserving old assets."""
from pathlib import Path
import dataclasses
import hashlib
import json
import pickle
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'src/snn_engine'):
    sys.path.insert(0,str(p))
from params import Params
from src.topic4_core_field_runner import get_network,connectivity_config,cache_key


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def summarize(net):
    ne,ni=int(net['NE']),int(net['NI'])
    result={}
    for name,bins,start,stop,expected in (
        ('E_to_E',net['ampa_by_delay'],0,ne,800),
        ('E_to_I',net['ampa_by_delay'],ne,ne+ni,800),
        ('I_to_E',net['gaba_by_delay'],0,ne,200),
        ('I_to_I',net['gaba_by_delay'],ne,ne+ni,200)):
        degree=np.zeros(stop-start,int);weight=np.zeros(stop-start);diag=0
        for mat in bins:
            coo=mat.tocoo(copy=False);mask=(coo.row>=start)&(coo.row<stop)
            row=coo.row[mask]-start;col=coo.col[mask];data=coo.data[mask]
            degree+=np.bincount(row,minlength=len(degree))
            weight+=np.bincount(row,weights=data,minlength=len(degree))
            if name in ('E_to_E','I_to_I'):diag+=int(np.sum(row==col))
        result[name]={'targets':len(degree),'edge_count':int(degree.sum()),
                      'in_degree_min':int(degree.min()),'in_degree_max':int(degree.max()),
                      'expected_in_degree':expected,'exact_expected_degree':bool(np.all(degree==expected)),
                      'self_edges':diag,'mean_incoming_weight':float(weight.mean())}
    return result


def main():
    start=time.time()
    source=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/connectivity_design_audit/connectivity_design_audit.json')
    audit=json.loads(source.read_text())
    old_record=audit['per_topology']['2511']['network_cache']
    old_path=Path(old_record['frozen_cache_path'])
    if sha(old_path)!=old_record['cache_sha256']:raise RuntimeError('old graph hash mismatch')
    with open(old_path,'rb') as f:old=pickle.load(f)
    old_config=old['config'];old_graph_summary=summarize(old['net'])
    kwargs={k:v for k,v in old_config.items() if k in {f.name for f in dataclasses.fields(Params)}}
    p=Params(**kwargs)
    out=ROOT/'results/topic4_sef_hfo/substrate_autapse_correction'
    cache=out/'network_cache';out.mkdir(parents=True,exist_ok=True)
    print('Rebuilding 40,000-cell topology 2511; old cache is read-only.',flush=True)
    net,ne,ni,hit=get_network(p,old_config['theta_EE_deg'],old_config['AR'],str(cache))
    new_summary=summarize(net)
    if not np.array_equal(old['net']['pos'],net['pos']):raise RuntimeError('positions changed')
    if any(x['self_edges'] for x in new_summary.values()):raise RuntimeError('autapse remains')
    if not all(x['exact_expected_degree'] for x in new_summary.values()):raise RuntimeError('in-degree changed')
    # Frozen connectivity pickles contain no post-build generator state. The
    # identical draw-count contract is checked at the sampler boundary in tests.
    random_stream_equal=None
    unchanged={}
    for name,key,rows in [('E_to_I','ampa_by_delay',slice(ne,None)),('I_to_E','gaba_by_delay',slice(0,ne))]:
        left=old['net'][key];right=net[key]
        unchanged[name]=len(left)==len(right) and all((a[rows]-b[rows]).nnz==0 for a,b in zip(left,right))
        if not unchanged[name]:raise RuntimeError(f'{name} changed unexpectedly')
    config=connectivity_config(p,old_config['theta_EE_deg'],old_config['AR'])
    path=cache/(cache_key(config)+'.pkl')
    result={'status':'FULL_SIZE_CORRECTED_GRAPH_VERIFIED','old_cache':old_record,
            'new_cache':{'path':str(path),'sha256':sha(path),'cache_hit':hit,'config':config},
            'old':old_graph_summary,'corrected':new_summary,'positions_identical':True,
            'post_build_rng_state_identical':random_stream_equal,
            'rng_contract':'post-build state absent from frozen cache; sampler draw-count and next-draw parity tested separately',
            'unchanged_cross_population_pathways':unchanged,
            'elapsed_seconds':time.time()-start,
            'source_hashes':{str(source):sha(source),**{str(ROOT/s):sha(ROOT/s) for s in ['src/snn_engine/connectivity.py','src/snn_engine/connectivity_rot.py','src/topic4_core_field_runner.py',str(Path(__file__).relative_to(ROOT))]}},
            'claim_boundary':'Structure repair on one full-size topology. Dynamics and revised core geometry are not validated by this check.'}
    (out/'graph_rebuild_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','old_cache','new_cache')},indent=2),flush=True)


if __name__=='__main__':main()
