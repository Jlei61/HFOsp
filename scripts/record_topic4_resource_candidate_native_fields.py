#!/usr/bin/env python3
"""Observation adapter preserving the revised-Z physical class in dense replay."""
import argparse,time
from pathlib import Path
import numpy as np
import record_topic4_global_candidate_native_fields as observer
import run_topic4_continuous_resource_recovery as physical

# The recorder wraps the constructor; it must wrap the actual physical class,
# not silently replace revised-Z dynamics with its GlobalPoolSlow parent.
physical.GlobalPoolSlow=physical.ResourceRecoverySlow
observer.source=physical
observer.SOURCE=physical.OUT
observer.destination=lambda name:physical.PARENT/'native_field_candidates_resource'/name
prepare0=observer.prepare

def prepare(name):
    p=prepare0(name);sha=physical.carrier.base.sha(__file__)
    if 'source_adapter_sha256' in p:assert p['source_adapter_sha256']==sha
    else:
        p.update(source_adapter=str(Path(__file__).resolve()),source_adapter_sha256=sha,
                 physical_class='ResourceRecoverySlow',revised_Z_equation_preserved=True)
        physical.carrier.base.write(observer.destination(name)/'protocol.json',p)
    return p

observer.prepare=prepare

def verify(name):
    observer.verify(name)
    got=observer.destination(name)/'runs'/name;ref=observer.SOURCE/'runs'/name
    def load(folder,kind):
        parts={}
        for path in sorted((folder/kind).glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as a:
                for k in a.files:parts.setdefault(k,[]).append(a[k])
        return {k:np.concatenate(v) for k,v in parts.items()}
    for kind in ['pool_chunks','resource_chunks']:
        a,b=load(got,kind),load(ref,kind);assert a and a.keys()==b.keys(),kind
        for k in a:assert np.array_equal(a[k],b[k]),(kind,k)
    p=observer.destination(name)/'observation_qa.json';q=physical.carrier.base.read(p)
    q.update(actual_revised_Z_class_preserved=True,all_added_resource_flux_and_pool_observations_bitwise=True,
             source_recovery_ratio=physical.carrier.base.read(observer.SOURCE/'jobs'/(name+'.json'))['recovery_ratio'])
    physical.carrier.base.write(p,q)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','worker','verify']);ap.add_argument('--name',required=True);a=ap.parse_args()
    try:
        if a.mode=='prepare':prepare(a.name)
        elif a.mode=='worker':observer.worker(a.name)
        else:verify(a.name)
    except Exception as e:
        physical.carrier.base.write(observer.destination(a.name)/'observation_failure.json',dict(error=repr(e),time=time.time()));raise
