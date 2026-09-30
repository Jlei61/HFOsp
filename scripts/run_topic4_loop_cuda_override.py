#!/usr/bin/env python3
"""Explicit execution-only CUDA device override; immutable scientific jobs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import run_topic4_loop_zk_conditional as native
import src.topic4_cuda_ordered_scatter as cuda_backend
from run_topic4_recovery_window import assert_same_state

PRIMARY=native.OUT
QA=PRIMARY/'qa/cuda_device0_route'


def configure(root):
    native.OUT=root
    native.prepare=lambda:native.base.read(root/'protocol.json')


def prepare_qa():
    QA.mkdir(exist_ok=True,parents=True)
    p=copy.deepcopy(native.base.read(PRIMARY/'protocol.json'))
    p.update(stage='EXECUTION_DEVICE_QA_ONLY',initial_jobs=[],deadline_epoch=time.time()+86400)
    native.write(QA/'protocol.json',p)
    shutil.copy2(PRIMARY/'geometry.npz',QA/'geometry.npz')
    configure(QA)
    for history,checkpoint in [('high','entry1_checkpoint.pkl'),('interictal','t50s.pkl')]:
        native.make_job(f'resume_{history}_gpu0',checkpoint,.2)
    print(str(QA),flush=True)


def worker(root,name,device):
    configure(root)
    job=native.base.read(root/'jobs'/f'{name}.json')
    runtime=dict(actual_backend='original_cuda_ordered_scatter',actual_device=device,
        planned_job_backend=job['backend'],planned_job_device=job['device'],
        execution_override_only=True,physics_and_job_unchanged=True,
        wrapper_sha256=native.base.sha(__file__),backend_sha256=native.base.sha(cuda_backend.__file__))
    folder=root/'runs'/name
    native.write(folder/'runtime_backend.json',runtime)
    fixed0=native.fixed.worker
    def gpu_worker(target):
        native.carrier.wrap_simulator=lambda fn,device_index:cuda_backend.wrap_simulator(fn,device_index=device)
        return fixed0(target)
    native.fixed.worker=gpu_worker
    native.worker(name)
    result=native.base.read(folder/'result.json');result['runtime_backend']=runtime
    native.write(folder/'result.json',result);native.write(folder/'progress.json',result)


def verify_qa():
    configure(QA);rows=[]
    for history in ['high','interictal']:
        name=f'resume_{history}_gpu0';native.qa_compare(name)
        a=native.read_pickle(QA/'runs'/name/'checkpoint.pkl')['engine']
        b=native.read_pickle(PRIMARY/'runs'/f'resume_{history}_cpu'/'checkpoint.pkl')['engine']
        assert_same_state(a,b)
        rows.append(dict(history=history,whole_engine_bitwise=True,native_observations_bitwise=True))
    native.write(QA/'gate.json',dict(status='PASS',device=0,checks=rows,
        wrapper_sha256=native.base.sha(__file__),backend_sha256=native.base.sha(cuda_backend.__file__),
        scope='Same saved high and interictal states,0.2s each; native observations plus complete engine state match serial CPU and original source. Execution metadata explicitly records device override.'))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare-qa','worker','verify-qa'])
    p.add_argument('--root',type=Path,default=PRIMARY);p.add_argument('--name');p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.command=='prepare-qa':prepare_qa()
    elif a.command=='verify-qa':verify_qa()
    else:worker(a.root,a.name,a.device)
