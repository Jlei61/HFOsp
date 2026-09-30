#!/usr/bin/env python3
"""Build one current-point proposal using repaired values and measured DC.

This deliberately keeps the earlier fixed-start proposal untouched. It does
not launch a response check or accept a root automatically.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
from campaign import ROOT,read,write,sha
import collect_held_exit_dc as collector

OUT=ROOT/'held_exit_phase_dc_operator_K9p35'
VALUES=ROOT/'held_exit_phase_stationarity_K9p35'
DC=ROOT/'held_exit_dc_K9p35'


def main(wait):
    OUT.mkdir(exist_ok=True);assert not (OUT/'analysis.json').exists()
    while not (VALUES/'result.json').exists():
        write(OUT/'preparation_progress.json',dict(status='WAITING_REPAIRED_ALLTARGET_VALUES',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    assert read(VALUES/'result.json')['status']=='COMPLETE_PHASE_ALLTARGET_STATIONARY_RESPONSE'
    assert read(DC/'analysis.json')['status']=='COMPLETE_DC_OPERATOR_AND_UNVALIDATED_STEP'
    assert sha(VALUES/'inputs.npz')==sha(ROOT/'held_exit_stationarity_K9p35/inputs.npz')
    for i in range(2):
        dest=OUT/f'part_{i}'
        if not dest.exists():dest.symlink_to(DC/f'part_{i}',target_is_directory=True)
        assert dest.resolve()==(DC/f'part_{i}').resolve()
    write(OUT/'value_repair_contract.json',dict(status='REGISTERED_BEFORE_REPAIRED_CORRECTION',created_epoch=time.time(),
        question='At the same K9.35 coordinates, does the phase-repaired value residual support one bounded current-point measured-DC correction?',
        values=str(VALUES),values_sha256=sha(VALUES/'response.npz'),DC=str(DC),DC_sha256=sha(DC/'measured_dc.npz'),
        derivative_scope='DC is measured at these same inputs, not transported fromK9. Selected observerprotocol51/51estimable comparisons passed; this does not promote alltarget amplitude failures or weak components. Preserve every batch and eight-block error.',
        solver='Existing implicitM/globalG exact moment construction and one bounded GMRES solve. A resulting proposal still needs current-point chain-rule QA and fresh nonlinear phase-aware response counts before acceptance.',
        stopping='No automatic continuation, new native job, eigenvalue interpretation or root promotion.',
        producer_sha256=sha(__file__),collector_sha256=sha(collector.__file__),formal_bifurcation_allowed=False))
    collector.OUT=OUT;collector.SOURCE=VALUES;collector.main(False)
    write(OUT/'preparation_progress.json',dict(status='COMPLETE_UNVALIDATED_CORRECTION_ONLY',updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
