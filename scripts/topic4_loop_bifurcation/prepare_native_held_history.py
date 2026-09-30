#!/usr/bin/env python3
"""One native K9.35 history control, reusing a completed K9 high state."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import copy,time
from pathlib import Path
import numpy as np
from campaign import ROOT,read,write,sha
import spatial_probes as probes
from run_topic4_recovery_window import assert_same_state

OUT=ROOT/'native_K9p35_held_history'
NAME='exit_z0.21_k9.35_fields16p7_held_K9_history'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'scientific_contract.json').exists()
    comparison=read(ROOT/'native_exit_K_bracket/density_correspondence/result.json')
    assert comparison['status']=='COMPLETE_NATIVE_DENSITY_K_BRACKET_COMPARISON'
    native_row=next(r for r in comparison['rows'] if r['K']==9.35)
    assert native_row['native_tail20to30s_rate_Hz'][2]<200
    den=ROOT/'density_exit_bracket_protocol/exit_z0.21_k9.35_fields16p7_high'
    assert read(den/'result.json')['status']=='COMPLETE'
    base=ROOT/'exit_return_probes';baseline='exit_z0.21_k9_fields16p7_high'
    job=read(base/'jobs'/f'{baseline}.json');source=base/'runs'/baseline/'checkpoint.pkl'
    assert read(base/'runs'/baseline/'result.json')['status']=='COMPLETE'
    old=probes.native.read_pickle(source);assert old['engine']['step']==420000
    original=read(base/'extended_analysis'/f'{baseline}.json')
    assert all(v>300 for v in original['five_second_windows'][-1]['mean_Hz_allE_A_B_other'][1:3])
    write(OUT/'scientific_contract.json',dict(status='REGISTERED_BEFORE_ONE_NATIVE_HISTORY_CONTROL',created_epoch=time.time(),
        question='Can the original network retain the high-recruitment K9.35 state when started from its completed heldK9 high history, instead of the original evolving12s history?',
        selection='After complete nativeK9.35 shows weakenedcoreB and immediate-history density reproduces that qualitativepattern, while carried-densityK9.35 staysbothcores high. This follow-up asks whether the carried high state is also reachable in the native network.',
        design='Exactly one30s native trajectory. Start from the complete42s endpoint of the existing heldK9 native high control. Keep V/synapses/pending/refractory/M/R/G and sameZfield; setK9.35 by the same16.7s field family; pair the complete future externalRNG/OU/drive to the original50s handoff. Only endogenous starting history differs from the existing K9.35 immediate-clamp condition. No new equation or Kschedule.',
        readouts='Complete30s rate/core/400cell fields/G/counterfactualZandKflow; exact300futureinput record pairing. Compare20-30s with the alreadycompleted original-historyK9.35 andcarried densityendpoint.',
        interpretation='This is a direct smallK step from a settled highhistory, not a slowKramp. Agreement supports conditional highstate relevance but doesnot certify stability/multistability/fold. Failure prevents promoting the densityhighbranch as nativeexit evidence. Finitehistories andnoise still remain distinct.',
        stopping='One30s run only. No automatic newK, extra noise seed, horizon extension, or branch certification.',
        source=str(source),source_sha256=sha(source),producer_sha256=sha(__file__),counts_as_autonomous_loop=False,formal_bifurcation_allowed=False))
    row=dict(name=NAME,Z=.21,K=9.35,duration_s=30.,cut='exit',history_label='held_K9_high',
        history_checkpoint=str(source),Z_template=job['Z_template'],K_template=job['K_template'],
        noise_checkpoint=job['external_noise_source'],reason='Test original-network highstate reachability atK9.35 from its own completed heldK9history.')
    spec=dict(stage='NATIVE_K9P35_HELD_HISTORY',question=read(OUT/'scientific_contract.json')['question'],rows=[row],max_workers=1)
    write(OUT/'probe_spec.json',spec);probes.prepare(OUT,spec)
    saved=probes.native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')
    fields=np.load(OUT/'runs'/NAME/'held_fields.npz')
    expected=copy.deepcopy(old['engine'])
    assert np.array_equal(expected['slow']['z'][:32000],fields['Z'])
    expected['termination_mechanism']['sahp_g'][:]=fields['K']
    common=probes.native.read_pickle(Path(job['external_noise_source']))['engine']
    for key in ['rng_state','xi','external_drive']:expected[key]=copy.deepcopy(common[key])
    offset=expected['step']-common['step']
    for key in ['next_step','last_step']:expected['external_drive'][key]+=offset
    assert_same_state(expected,saved['engine'])
    write(OUT/'initial_state_qa.json',dict(status='PASS',full_internal_engine_carried_bitwise_except_declaredK=True,
        complete_matched_exogenous_handoff=True,source_step=420000,K=9.35,
        native_equations_unchanged=True))
    print('ONE HELD-HISTORY NATIVE CONTROL PREPARED',flush=True)


if __name__=='__main__':main()
