#!/usr/bin/env python3
"""User-authorized shorter second-onset observation; unchanged native physics."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys
import run_topic4_m_parameter_modes as core

POLICY=core.OUT/'observation_tail_amendment_20260913.json'


def install_short_tail(seconds):
    original=core.tracker_step
    def tracker(tr,rate,sec,rescue=True):
        original(tr,rate,sec,rescue=rescue)
        if len(tr['entries'])>=2:
            tr['stop_s']=min(tr['stop_s'],tr['entries'][1]['confirmation_s']+seconds)
    core.tracker_step=tracker
    return original


def main():
    args=sys.argv[1:]
    assert len(args)==4 and args[1:3]==['worker','--job']
    assert Path(args[0]).resolve()==Path(core.__file__).resolve()
    job_path=Path(args[3]);job=core.read(job_path);folder=core.OUT/'runs'/job['name']
    policy=core.read(POLICY);assert policy['user_authorized'] and policy['post_second_confirmation_s']==2
    record=policy['adopted_jobs'][job['name']]
    assert core.sha(folder/'checkpoint.pkl')==record['checkpoint_sha256']
    assert not (folder/'result.json').exists()
    qa=core.read(core.ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/cuda_ordered_scatter_qa/full_network_qa.json')
    assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical']
    assert core.sha(qa['replacement'])==qa['replacement_sha256']
    for p,h in core.read(core.OUT/'protocol.json')['source_hashes'].items():assert core.sha(p)==h,p
    saved=core.load_pickle(folder/'checkpoint.pkl')
    assert saved['job']==job and saved['engine']['step']==record['resume_step']
    del saved
    install_short_tail(policy['post_second_confirmation_s'])
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    core.old.simulate_kick=wrap_simulator(core.old.simulate_kick,device_index=record['device'])
    core.write(folder/'computational_variant.json',dict(variant='CUDA_ordered_incoming',
        device=record['device'],actual_pid=os.getpid(),validation=str(qa['replacement']),
        replacement_sha256=qa['replacement_sha256'],observation_tail_amendment=str(POLICY),
        physics_and_random_history_unchanged=True,new_independent_samples=0))
    core.write(folder/'observation_tail_runtime.json',dict(pid=os.getpid(),policy=str(POLICY),
        post_second_confirmation_s=2,checkpoint_quantization_s=.5,
        native_Z_M_and_fast_dynamics_unchanged=True,full_state_and_noise_history_preserved=True,
        resume_step=record['resume_step'],device=record['device'],new_independent_samples=0,
        source=str(Path(__file__).resolve()),source_sha256=core.sha(__file__)))
    core.worker(job)
    result=core.read(folder/'result.json')
    result['observation_tail_amendment']=str(POLICY)
    result['post_second_confirmation_s']=2
    core.write(folder/'result.json',result)


if __name__=='__main__':main()
