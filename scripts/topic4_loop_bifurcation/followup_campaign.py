#!/usr/bin/env python3
"""Preserved-prefix extensions of selected conditional native trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import shutil
import time
from pathlib import Path
import numpy as np
from campaign import ROOT,NATIVE,read,write,sha
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu
from run_topic4_recovery_window import assert_same_state

EXTENSIONS=[f'{cut}_{history}' for cut,histories in [
    ('exit_z0.21_k6',['high','recovery']),('exit_z0.21_k9',['high','recovery']),
    ('entry_z0.74_k0.0002',['high','interictal'])] for history in histories]


def prepare_extensions(root):
    assert read(NATIVE/'status.json')['stage']=='COMPLETE'
    assert not (root/'protocol.json').exists()
    root.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(read(NATIVE/'protocol.json'))
    p.update(stage='HISTORY_PERSISTENCE_120S',created_epoch=time.time(),
        deadline_epoch=time.time()+7*86400,max_workers=6,
        question='Do30s history differences at exitK6/9 and entryZ.74 persist to120s, or relax into the same conditional state?',
        selected_jobs=EXTENSIONS,extension_additional_duration_s=90.,total_observed_s=120.,
        fixed='Same clamped fields, original physics, G/M dynamics, common future input continued from saved complete30s state. No reset.',
        preparer_sha256=sha(__file__),
        selection_evidence=str(NATIVE/'extended_analysis_summary.json'),
        continuation_limits='Six extensions selected by documented history differences. Finite120s persistence still does not certify bistability.')
    shutil.copy2(NATIVE/'geometry.npz',root/'geometry.npz')
    write(root/'protocol.json',p)
    source_prefix={}
    for name in EXTENSIONS:
        source=NATIVE/'runs'/name;dest=root/'runs'/name;dest.mkdir(parents=True)
        original=read(NATIVE/'jobs'/f'{name}.json')
        job=copy.deepcopy(original);job.update(horizon_s=original['horizon_s']+90.,
            stage='conditional_extension',extension_source=str(source),total_conditional_duration_s=120.)
        saved=native.read_pickle(source/'checkpoint.pkl')
        assert saved['engine']['step']==round(original['horizon_s']*10000)
        before=copy.deepcopy(saved['engine']);saved['job']=job
        native.base.save_pickle(dest/'checkpoint.pkl',saved)
        assert_same_state(before,native.read_pickle(dest/'checkpoint.pkl')['engine'])
        source_prefix[name]={}
        for folder in source.iterdir():
            if folder.is_dir() and (folder.name.endswith('chunks') or folder.name=='chunks'):
                (dest/folder.name).mkdir()
                for file in folder.glob('*.npz'):
                    assert '.tmp' not in file.name
                    os.link(file,dest/folder.name/file.name)
                    source_prefix[name][str(file.relative_to(source))]=sha(file)
        write(root/'jobs'/f'{name}.json',job)
    write(root/'preserved_prefix.json',dict(status='ENGINE_IDENTICAL_AT_CONTINUATION',
          source=str(NATIVE),read_only_prefix_hardlinks=True,hashes=source_prefix))
    write(root/'queue.json',dict(names=EXTENSIONS,total=len(EXTENSIONS),bounded=True,
        job_sha256={n:sha(root/'jobs'/f'{n}.json') for n in EXTENSIONS}))
    print(root,flush=True)


def worker(root,name,device):
    p=read(root/'protocol.json')
    assert sha(__file__)==p['preparer_sha256']
    assert sha(root/'jobs'/f'{name}.json')==read(root/'queue.json')['job_sha256'][name]
    for path,digest in p['source_hashes'].items():assert sha(path)==digest,path
    assert sha(native.__file__)==p['runner_sha256']
    gpu.worker(root,name,device)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare-extensions','worker'])
    parser.add_argument('--root',type=Path,default=ROOT/'native_extensions')
    parser.add_argument('--name');parser.add_argument('--device',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.command=='prepare-extensions':prepare_extensions(args.root)
    else:worker(args.root,args.name,args.device)
