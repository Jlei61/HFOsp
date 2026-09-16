#!/usr/bin/env python3
"""Run the unchanged M worker with a verified order-preserving scatter lookup."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import hashlib
import sys
import run_topic4_m_parameter_modes as core


def main():
    args=sys.argv[1:]
    assert len(args)==4 and args[1:3]==['worker','--job'],args
    assert Path(args[0]).resolve()==Path(core.__file__).resolve()
    path=Path(args[3]);assert path.parent.resolve()==(core.OUT/'jobs').resolve()
    qa_path=core.ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/scatter_lookup_qa/full_network_qa.json'
    qa=core.read(qa_path)
    assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical'] and qa['all_observations_bitwise_identical']
    replacement=Path(qa['replacement'])
    assert hashlib.sha256(replacement.read_bytes()).hexdigest()==qa['replacement_sha256']
    import src.topic4_serial_spike_scatter as original
    from src.topic4_serial_spike_scatter_lookup import scatter
    job=core.read(path);folder=core.OUT/'runs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'checkpoint.pkl').exists(),'This wrapper only dispatches never-started jobs'
    core.write(folder/'computational_variant.json',dict(
        variant='serial_delay_slot_lookup',validation=str(qa_path),
        actual_pid=os.getpid(),original_scatter=str(Path(original.__file__).resolve()),
        replacement=str(replacement),replacement_sha256=qa['replacement_sha256'],
        executor=str(Path(__file__).resolve()),executor_sha256=core.sha(__file__),
        original_worker=str(Path(core.__file__).resolve()),original_worker_sha256=core.sha(core.__file__),
        floating_addition_order_unchanged=True,physics_parameters_and_noise_unchanged=True,
        full_high_state_bitwise_QA_duration_s=.2,previous_running_workers_untouched=True))
    original.scatter=scatter
    core.worker(job)


if __name__=='__main__':main()
