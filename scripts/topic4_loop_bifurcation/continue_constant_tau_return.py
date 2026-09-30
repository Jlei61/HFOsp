#!/usr/bin/env python3
"""One bounded follow-up of the unexpected recovery without low-rate retention."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import shutil
import subprocess
import time
from campaign import ROOT, read, write, sha, PYTHON
import probe_natural_exit_mediators as runner
import run_topic4_loop_zk_conditional as native
from run_topic4_recovery_window import assert_same_state

OUT = ROOT/'constant_tau_return_followup'
NAME = runner.CONTROLS[1]
SOURCE = ROOT/'natural_exit_mediator_probes/runs'/NAME


def configure():
    runner.OUT = OUT
    native.OUT = OUT
    native.prepare = lambda: read(OUT/'protocol.json')


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    original = ROOT/'natural_exit_mediator_probes'
    result = read(original/'mediator_analysis/result.json')
    row = next(r for r in result['rows'] if r['name'] == NAME)
    assert row['first_both_core_Z_reference_s'] == 25.48
    assert len(row['complete_brief_after_Z_reference']) == 1
    prefix = read(original/'constant_tau_prefix_diagnostic.json')
    assert prefix['constant_tau_prefix_equivalence_supported']
    assert prefix['continuous_R_lower_bound_after_previous_sample_Hz'] > 5
    contract = copy.deepcopy(read(original/'contract.json'))
    contract.update(status='REGISTERED_ONE_CONSTANT_TAU_RETURN_FOLLOWUP', created_epoch=time.time(),
        question='Does the constant0.5s K-decay variant regain sustained native-like core-led interictal propagation after its interrupted but sufficient Z recovery?',
        motivation='The completed two-mediator experiment refuted necessity of5s low-rate retention for coreZ recovery: constanttau reachesbothreferences25.48s and onecompletebrief occurs26.47s. Ten seconds cannot establish sustainedreturn.',
        design='Exactly one30s continuation fromthe completed26.8s fullconstanttau state to56.8s. Same originalgraph, expectedandstochasticexternaldrive, nativeZ/M/G/K/voltage/ref/delay histories. Ktau0.5s bothaboveandbelowR5; noheldfields, reset,pulse,quiettimer ornewseed. Originalbaseline through56.8s reused.',
        prefix_equivalence='Original0-16.8s is invariant under this Ktau change: K is exactlyzero until9.874s; oncepositive, the continuousR lowerbound is13.864Hz>5. Thus the changed low-rate rule has no earlier physical effect. This is a paired parameter trajectory, not an independent seed.',
        readouts='Full16.8-56.8s activity andcoreZ; originalstrict eventobserver, >=10brief events spanning>=5s after bothcore references, brief fraction>=0.8 andsame-seedreference duration/IEI/peak ratios0.5-2. Spatialnative5ms core recruitment/propagation andraster reviewed separately. Reentries/censoring retained.',
        decisions='If core-led briefreturnpersists, 5Hz-dependent Kretention is not needed for this completed sequence under this pairedseed; it mainlychangesinterruption/waiting. Ifreturnfails, adequateZ alone remainsinsufficient. Do not promote a singlevariant to theformalFig5 orreplace acceptedsource.',
        stop='One30s continuation only; no automaticnewseed, furtherhorizon orparametergrid.',
        extension_producer_sha256=sha(__file__), continuation_source=str(SOURCE/'checkpoint.pkl'),
        continuation_source_sha256=sha(SOURCE/'checkpoint.pkl'), counts_as_independent_seed=False,
        counts_as_autonomous_loop=False, formal_bifurcation_allowed=False)
    write(OUT/'contract.json', contract)
    protocol = copy.deepcopy(read(original/'protocol.json'))
    protocol.update(stage='CONSTANT_TAU_RETURN_FOLLOWUP', deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json', protocol);shutil.copy2(original/'geometry.npz', OUT/'geometry.npz')
    configure();job = native.make_job(NAME, str(SOURCE/'checkpoint.pkl'), 30., clamp=False, common_input=False)
    saved = native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')
    before = native.read_pickle(SOURCE/'checkpoint.pkl')['engine']
    assert before['step'] == 268000 and before['slow']['kind'] == 'NoRetention'
    saved['engine']['slow']['kind'] = 'NoRetention'
    assert_same_state(saved['engine'], before)
    job.update(stage='CONSTANT_TAU_RETURN_FOLLOWUP', diagnostic_intervention='constant_tau_parameter_variant',
               counts_as_independent_seed=False, counts_as_autonomous_loop=False, off_tau_s=.5)
    saved['job'] = job;native.base.save_pickle(OUT/'runs'/NAME/'checkpoint.pkl', saved)
    write(OUT/'jobs'/f'{NAME}.json', job)
    write(OUT/'initial_state_qa.json', dict(status='PASS', complete_engine_bitwise=True,
        no_state_intervention=True, continued_equations_unchanged=True, source_step=268000))
    shutil.copy2(__file__, OUT/'producer.py')


def worker():
    assert sha(__file__) == read(OUT/'contract.json')['extension_producer_sha256']
    configure();runner.worker(NAME, 0)
    p = OUT/'runs'/NAME/'result.json';result = read(p)
    result.update(continuation_has_no_state_intervention=True, constant_K_tau_s=.5,
                  source_parameter_variant=NAME, counts_as_independent_seed=False)
    write(p, result);write(p.parent/'progress.json', result)


def supervise():
    assert not (OUT/'supervisor.json').exists()
    with (OUT/'worker.log').open('w') as log:
        p = subprocess.Popen([PYTHON, __file__, 'worker'], stdout=log, stderr=subprocess.STDOUT)
        write(OUT/'supervisor.json', dict(status='RUNNING_ONE_CONTINUATION', pid=os.getpid(), worker_pid=p.pid,
            updated_epoch=time.time(), native_start_s=26.8, native_end_s=56.8))
        code = p.wait()
    write(OUT/'supervisor.json', dict(status='COMPLETE_ANALYSIS_PENDING' if code == 0 else 'FAILED',
        pid=os.getpid(), worker_exit_code=code, updated_epoch=time.time()))
    assert code == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'supervise', 'worker']);a = p.parse_args()
    {'prepare': prepare, 'supervise': supervise, 'worker': worker}[a.command]()
