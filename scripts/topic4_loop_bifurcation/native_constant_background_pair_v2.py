#!/usr/bin/env python3
"""Repair pre-run job/checkpoint metadata synchronization; native state unchanged."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,shutil,subprocess,time
from pathlib import Path
from campaign import ROOT,PYTHON,read,write,sha
import native_constant_background_pair as implementation
from run_topic4_recovery_window import assert_same_state

FAILED=ROOT/'native_K9p35_constant_background_pair'
OUT=ROOT/'native_K9p35_constant_background_pair_v2'


def configure():
    implementation.OUT=OUT


def prepare():
    assert read(FAILED/'supervisor.json')['status']=='FAILED'
    rows=[]
    for name,parent in zip(implementation.NAMES,implementation.PARENTS):
        a=implementation.base.native.read_pickle(FAILED/'runs'/name/'checkpoint.pkl')
        b=implementation.base.native.read_pickle(parent/'checkpoint.pkl')
        assert_same_state(a['engine'],b['engine'])
        assert set(read(FAILED/'jobs'/f'{name}.json'))-set(a['job'])=={'external_expected_rate_override'}
        rows.append(dict(name=name,failed_engine_exact_initial=True,step=a['engine']['step']))
    configure();implementation.prepare()
    for name in implementation.NAMES:
        path=OUT/'runs'/name/'checkpoint.pkl';saved=implementation.base.native.read_pickle(path)
        before=copy.deepcopy(saved['engine'])
        saved['job']=read(OUT/'jobs'/f'{name}.json')
        implementation.base.native.base.save_pickle(path,saved)
        reloaded=implementation.base.native.read_pickle(path)
        assert_same_state(before,reloaded['engine']);assert reloaded['job']==read(OUT/'jobs'/f'{name}.json')
    c=read(OUT/'contract.json');c['metadata_wrapper_sha256']=sha(__file__)
    c['previous_failed_pre_run']=str(FAILED)
    write(OUT/'contract.json',c)
    write(OUT/'metadata_sync_qa.json',dict(status='PASS',previous_failures=rows,
        new_job_and_checkpoint_job_exact=True,new_full_initial_engines_unchanged=True,
        correction='Synchronize the already declared external-rate metadata in the saved job and JSON before execution. No physical equation, parameter, random stream or initial engine changed.',wrapper_sha256=sha(__file__)))
    shutil.copy2(__file__,OUT/'metadata_wrapper_producer.py')


def worker(index):
    configure();assert read(OUT/'contract.json')['metadata_wrapper_sha256']==sha(__file__)
    implementation.worker(index)


def supervise():
    configure();assert not (OUT/'supervisor.json').exists()
    assert read(ROOT/'individual_source_spectral_independent_value/result.json')['status']=='COMPLETE_INDEPENDENT_FINITE_WINDOW_VALUE'
    write(OUT/'supervisor.json',dict(status='RUNNING',pid=os.getpid(),created_epoch=time.time(),wrapper_sha256=sha(__file__)))
    jobs=[]
    for i in range(2):
        log=(OUT/f'worker{i}.log').open('w')
        p=subprocess.Popen([PYTHON,__file__,'worker','--index',str(i)],stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
    codes=[]
    for p,log in jobs:codes.append(p.wait());log.close()
    if any(codes):
        write(OUT/'supervisor.json',dict(status='FAILED',codes=codes));raise RuntimeError(codes)
    implementation.finish();write(OUT/'supervisor.json',dict(status='COMPLETE',updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','supervise']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    prepare() if a.command=='prepare' else worker(a.index) if a.command=='worker' else supervise()
