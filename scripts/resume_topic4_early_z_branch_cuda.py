#!/usr/bin/env python3
"""Accelerate one existing early-refill branch without changing its protocol."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import run_topic4_early_z_refill_branches as branch


def main():
    args = sys.argv[1:]
    assert len(args) == 4 and args[1:3] == ['worker', '--name'], args
    assert Path(args[0]).resolve() == Path(branch.__file__).resolve()
    core = branch.core
    job = core.read(branch.OUT / 'jobs' / (args[3] + '.json'))
    folder = branch.OUT / 'runs' / job['name']
    plan = core.read(folder / 'cuda_resume_authorization.json')
    qa_path = branch.WINDOW / 'cuda_ordered_scatter_qa/full_network_qa.json'
    qa = core.read(qa_path)
    assert qa['status'] == 'PASS'
    assert qa['full_engine_state_bitwise_identical'] and qa['all_observations_bitwise_identical']
    assert core.sha(qa['replacement']) == qa['replacement_sha256']
    assert plan['job'] == job and plan['existing_branch_only']
    assert core.sha(branch.__file__) == plan['branch_producer_sha256']
    assert not (folder / 'result.json').exists()
    checkpoint = folder / 'checkpoint.pkl'
    assert core.sha(checkpoint) == plan['resume_checkpoint_sha256']
    saved = core.load_pickle(checkpoint)
    assert saved['job'] == job and saved['engine']['step'] == plan['resume_step']
    for source, digest in core.read(branch.OUT / 'protocol.json')['source_hashes'].items():
        assert core.sha(source) == digest, source
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    core.old.simulate_kick = wrap_simulator(core.old.simulate_kick, device_index=plan['device'])
    core.tracker_step = branch.early_tracker_step
    core.OUT = branch.OUT
    core.write(folder / 'computational_variant.json', dict(
        variant='CUDA_ordered_incoming', device=plan['device'], actual_pid=os.getpid(),
        validation=str(qa_path), replacement_sha256=qa['replacement_sha256'],
        executor=str(Path(__file__).resolve()), executor_sha256=core.sha(__file__),
        branch_producer=str(Path(branch.__file__).resolve()),
        branch_producer_sha256=plan['branch_producer_sha256'],
        resume_step=plan['resume_step'], checkpoint_sha256=plan['resume_checkpoint_sha256'],
        unchanged_parameters_intervention_endpoints_and_full_state=True,
        new_independent_samples=0))
    del saved
    core.worker(job)


if __name__ == '__main__':
    main()
