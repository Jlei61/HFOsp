#!/usr/bin/env python3
"""One native upper-point check with the same held-K9 endogenous history."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,shutil,subprocess,time
from pathlib import Path
import numpy as np
from campaign import ROOT,REPO,PYTHON,read,write,sha
import spatial_probes as probes
from run_topic4_recovery_window import assert_same_state

OUT=ROOT/'native_K9p5_held_history'
NAME='exit_z0.21_k9.5_fields16p7_held_K9_history'


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'scientific_contract.json').exists()
    previous=ROOT/'native_K9p35_held_history';oldname='exit_z0.21_k9.35_fields16p7_held_K9_history'
    assert read(previous/'runs'/oldname/'result.json')['status']=='COMPLETE'
    job=read(previous/'jobs'/f'{oldname}.json');source=Path(job['source_checkpoint'])
    original=probes.native.read_pickle(source);assert original['engine']['step']==420000
    write(OUT/'scientific_contract.json',dict(status='REGISTERED_BEFORE_ONE_NATIVE_UPPER_POINT',created_epoch=time.time(),
        question='Does the completed heldK9 native high history also becomequiet atK9.5, or was the apparent upperboundary specific to the evolving12s history?',
        design='Exactlyone30s native conditionalrun from the same42s heldK9 fullendogenous checkpoint as the completedK9.35 heldhistory. Same actual16.7s Zfield, originalgraph/physics, pairedfuture50s externalRNG/OU/drive, originaldynamicG/M. Change only heldK from9.35 to9.5 relative to the existinghistory control.',
        reasoning='The previous9.5 quiet point has evolving12s history; it cannot byitself bound the disappearance of the heldK9 high state used by the newphase closure. This one missinghistory comparison directly tests that branch connection.',
        readouts='Complete30s, tail20-30s wholeE/coreA/coreB/400cellfield, actualG, counterfactualZ/Kflow, brief events andcensoring. Exact recordedfutureinput pairing with both completedK9.35 heldhistory andK9.5 evolvinghistory.',
        decision='Ifquiet, both testedhistories lack sustainedactivity atthispoint but stability/fold remainsuncertified. Ifactive, do not use9.5 as a universal upperboundary or certify an exitfold there.',
        stopping='Onepoint, one30s conditionaltrajectory only; noautomatic newK, seeds or longerhorizon. It doesnot count as an autonomous exit or recovery.',
        source=str(source),source_sha256=sha(source),producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    row=dict(name=NAME,Z=.21,K=9.5,duration_s=30.,cut='exit',history_label='held_K9_high',
        history_checkpoint=str(source),Z_template=job['Z_template'],K_template=job['K_template'],
        noise_checkpoint=job['external_noise_source'],reason=read(OUT/'scientific_contract.json')['question'])
    spec=dict(stage='NATIVE_K9P5_HELD_HISTORY_UPPER_POINT',question=row['reason'],rows=[row],max_workers=1)
    write(OUT/'probe_spec.json',spec);probes.prepare(OUT,spec)
    saved=probes.native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')
    fields=np.load(OUT/'runs'/NAME/'held_fields.npz');oldfields=np.load(job['held_fields_file'])
    assert np.array_equal(fields['Z'],oldfields['Z'])
    assert np.allclose(fields['K'],oldfields['K']*(9.5/9.35),rtol=1e-14,atol=1e-14)
    expected=copy.deepcopy(original['engine']);expected['termination_mechanism']['sahp_g'][:]=fields['K']
    common=probes.native.read_pickle(Path(job['external_noise_source']))['engine']
    for key in ['rng_state','xi','external_drive']:expected[key]=copy.deepcopy(common[key])
    offset=expected['step']-common['step']
    for key in ['next_step','last_step']:expected['external_drive'][key]+=offset
    assert_same_state(expected,saved['engine'])
    write(OUT/'initial_state_qa.json',dict(status='PASS',full_internal_engine_bitwise_except_declaredK=True,
        unchanged_Z_exact=True,K_field_shape_exact_to_roundoff=True,complete_exogenous_handoff=True))
    shutil.copy2(__file__,OUT/'producer.py')


def supervise():
    assert read(OUT/'initial_state_qa.json')['status']=='PASS';assert not (OUT/'supervisor.json').exists()
    # Existing resource controller dispatches one frozen native worker only.
    write(OUT/'supervisor.json',dict(status='RUNNING_ONE_POINT',pid=os.getpid(),updated_epoch=time.time()))
    with (OUT/'controller.log').open('w') as log:
        proc=subprocess.Popen([PYTHON,str(REPO/'scripts/topic4_loop_bifurcation/supervise_probes_v4.py'),
            '--root',str(OUT),'--max-workers','1'],stdout=log,stderr=subprocess.STDOUT)
        code=proc.wait()
    assert code==0 and read(OUT/'status.json')['stage']=='COMPLETE'
    import analyze_native
    from analyze_actual_G_history import load
    analyze_native.main(OUT)
    candidates=[(ROOT/'native_exit_K_bracket','exit_z0.21_k9.5_fields16p7_high','K9.5_evolving12s'),
        (ROOT/'native_K9p35_held_history','exit_z0.21_k9.35_fields16p7_held_K9_history','K9.35_heldK9'),
        (OUT,NAME,'K9.5_heldK9')]
    data=[load(root,name) for root,name,label in candidates];rows=[]
    for (root,name,label),d in zip(candidates,data):
        assert np.array_equal(d['inputs'],data[0]['inputs'])
        m=(d['time5']>=20)&(d['time5']<30);md=(d['drift_time']>20)&(d['drift_time']<=30);mg=(d['time1']>=20)&(d['time1']<30)
        rows.append(dict(label=label,name=name,root=str(root),tail_rate_Hz_allE_A_B_surround=d['rate'][m].mean(0).tolist(),
            tail_Graw=float(d['G'][mg].mean()),tail_counterfactual_Zdot_per_s=d['drift'][md,:,0].mean(0).tolist(),
            finite_window_state=d['generic']['finite_window_state'],tail_brief_events=d['generic']['tail_brief_events']))
    write(OUT/'comparison.json',dict(status='COMPLETE_ONE_NATIVE_UPPER_POINT_COMPARISON',rows=rows,
        recorded_future_inputs_exact=True,formal_bifurcation_allowed=False,
        scope='Threefiniteconditionalresponses frompairedendogenoushistories andcommoninput, not independentseeds, attractorcertification orautonomousexit.'))
    write(OUT/'supervisor.json',dict(status='COMPLETE_NATIVE_UPPER_POINT_AND_ANALYSIS',updated_epoch=time.time()))
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise']);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:supervise()
