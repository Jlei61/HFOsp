#!/usr/bin/env python3
"""Explicit spatial-field/G/future-noise probes of the same native engine."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import copy
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from campaign import ROOT,NATIVE,read,write,sha
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu


def transform(z,k,zbar,kbar):
    logits=np.log(z)-np.log1p(-z);lo,hi=-40.,40.
    for _ in range(80):
        mid=(lo+hi)/2
        if np.mean(1/(1+np.exp(-(logits+mid))))<zbar:lo=mid
        else:hi=mid
    zz=1/(1+np.exp(-(logits+(lo+hi)/2)));kk=k.copy();kk*=kbar/kk.mean()
    assert abs(zz.mean()-zbar)<1e-13 and abs(kk.mean()-kbar)<1e-13
    return zz,kk


def prepare(root,spec):
    assert not (root/'protocol.json').exists()
    root.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(read(NATIVE/'protocol.json'))
    p.update(stage=spec['stage'],created_epoch=time.time(),deadline_epoch=time.time()+7*86400,
             question=spec['question'],preparer_sha256=sha(__file__),
             probe_spec=spec,max_workers=spec.get('max_workers',1))
    write(root/'protocol.json',p);shutil.copy2(NATIVE/'geometry.npz',root/'geometry.npz')
    native.OUT=root;native.prepare=lambda:p;oldfields=native.fields;names=[]
    for row in spec['rows']:
        name=row['name'];z=float(row['Z']);k=float(row['K'])
        ze=native.read_pickle(Path(row['Z_template']))['engine']['slow']['z'][:32000]
        ke=native.read_pickle(Path(row['K_template']))['engine']['termination_mechanism']['sahp_g']
        zz,kk=transform(ze,ke,z,k)
        native.fields=lambda zbar,kbar:(zz.copy(),kk.copy())
        job=native.make_job(name,row['history_checkpoint'],float(row.get('duration_s',30.)),True,z,k,common_input=True)
        folder=root/'runs'/name;fields=folder/'held_fields.npz'
        np.savez_compressed(fields,Z=zz,K=kk)
        saved=native.read_pickle(folder/'checkpoint.pkl')
        job.update(source_history=row['history_label'],local_cut=row.get('cut','entry'),
             Z_template=row['Z_template'],K_template=row['K_template'],
             held_fields_file=str(fields),held_fields_sha256=sha(fields),
             probe_reason=row['reason'],external_noise_source=row.get('noise_checkpoint',str(native.SOURCE/'runs'/native.NAME/'states/t50s.pkl')))
        if 'G_raw_override' in row:
            before=saved['engine']['global_feedback_response']['global_state']*30.
            saved['engine']['global_feedback_response']['global_state']=float(row['G_raw_override'])/30.
            job.update(G_raw_override=float(row['G_raw_override']),initial_G_raw_before_override=before)
        if 'noise_checkpoint' in row:
            noise=native.read_pickle(Path(row['noise_checkpoint']))['engine']
            for key in ['rng_state','xi','external_drive']:saved['engine'][key]=copy.deepcopy(noise[key])
            offset=saved['engine']['step']-noise['step']
            for key in ['next_step','last_step']:saved['engine']['external_drive'][key]+=offset
        saved['job']=job;native.base.save_pickle(folder/'checkpoint.pkl',saved)
        assert np.array_equal(saved['engine']['slow']['z'][:32000],zz)
        assert np.array_equal(saved['engine']['termination_mechanism']['sahp_g'],kk)
        write(root/'jobs'/f'{name}.json',job);names.append(name)
    native.fields=oldfields
    write(root/'queue.json',dict(names=names,total=len(names),bounded=True,
        job_sha256={n:sha(root/'jobs'/f'{n}.json') for n in names}))
    print(root,names,flush=True)


def worker(root,name,device):
    p=read(root/'protocol.json');j=read(root/'jobs'/f'{name}.json')
    assert sha(__file__)==p['preparer_sha256']
    assert sha(root/'jobs'/f'{name}.json')==read(root/'queue.json')['job_sha256'][name]
    assert sha(j['held_fields_file'])==j['held_fields_sha256']
    for path,h in p['source_hashes'].items():assert sha(path)==h,path
    with np.load(j['held_fields_file']) as f:zz=f['Z'].copy();kk=f['K'].copy()
    def fixed_fields(zbar,kbar):
        assert zbar==j['target_Z'] and kbar==j['target_K']
        return zz.copy(),kk.copy()
    native.fields=fixed_fields
    gpu.worker(root,name,device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--spec',type=Path)
    p.add_argument('--name');p.add_argument('--device',type=int,choices=[0,1],default=1)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root,read(a.spec))
    else:worker(a.root,a.name,a.device)
