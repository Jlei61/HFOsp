#!/usr/bin/env python3
"""Resume one existing M-grid job with the bitwise-verified CUDA scatter."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import run_topic4_m_parameter_modes as core


def main():
    args = sys.argv[1:]
    assert len(args) == 4 and args[1:3] == ['worker', '--job'], args
    assert Path(args[0]).resolve() == Path(core.__file__).resolve()
    job_path = Path(args[3])
    assert job_path.parent.resolve() == (core.OUT / 'jobs').resolve()
    window = core.ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
    qa_path = window / 'cuda_ordered_scatter_qa/full_network_qa.json'
    qa = core.read(qa_path)
    assert qa['status'] == 'PASS'
    assert qa['full_engine_state_bitwise_identical'] and qa['all_observations_bitwise_identical']
    assert core.sha(qa['replacement']) == qa['replacement_sha256']
    job = core.read(job_path)
    folder = core.OUT / 'runs' / job['name']
    plan_path = folder / 'cuda_resume_authorization.json'
    plan = core.read(plan_path)
    assert plan['job'] == job and plan['existing_grid_job_only']
    assert not (folder / 'result.json').exists()
    cp = folder / 'checkpoint.pkl'
    assert core.sha(cp) == plan['resume_checkpoint_sha256']
    saved = core.load_pickle(cp)
    assert saved['job'] == job and saved['engine']['step'] == plan['resume_step']
    for source, digest in core.read(core.OUT / 'protocol.json')['source_hashes'].items():
        assert core.sha(source) == digest, source
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    core.old.simulate_kick = wrap_simulator(core.old.simulate_kick, device_index=plan['device'])
    core.write(folder / 'computational_variant.json', dict(
        variant='CUDA_ordered_incoming', device=plan['device'], actual_pid=os.getpid(),
        validation=str(qa_path), replacement_sha256=qa['replacement_sha256'],
        executor=str(Path(__file__).resolve()), executor_sha256=core.sha(__file__),
        original_worker=str(Path(core.__file__).resolve()), original_worker_sha256=core.sha(core.__file__),
        resume_step=plan['resume_step'], checkpoint_sha256=plan['resume_checkpoint_sha256'],
        unchanged_parameters_endpoints_and_full_state=True, new_independent_samples=0))
    del saved
    core.worker(job)


if __name__ == '__main__':
    main()
