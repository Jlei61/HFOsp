#!/usr/bin/env python3
"""Two native initial-G controls in the actual exit-field family, no new physics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import copy,time
import numpy as np
from campaign import ROOT,read,write,sha
import spatial_probes as probes
from run_topic4_recovery_window import assert_same_state

OUT=ROOT/'exit_actual_G_history_probes'
BASE=ROOT/'exit_return_probes'
BASE_NAME='exit_z0.21_k9_fields16p7_high'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'scientific_contract.json').exists()
    job=read(BASE/'jobs'/f'{BASE_NAME}.json')
    actual=probes.native.read_pickle(__import__('pathlib').Path(job['Z_template']))
    Gactual=float(actual['engine']['global_feedback_response']['global_state']*30.)
    source=probes.native.read_pickle(__import__('pathlib').Path(job['source_checkpoint']))
    Gsource=float(source['engine']['global_feedback_response']['global_state']*30.)
    assert actual['engine']['step']==167000 and source['engine']['step']==120000
    write(OUT/'scientific_contract.json',dict(status='REGISTERED_BEFORE_NEW_INITIAL_G_BRANCHES',created_epoch=time.time(),
        question='In the actual16.7s exit-field family at heldZmean.21/Kmean9, can changing only the carried initial global conductance alter persistence of the high state that already has a directly corresponding conditional branch?',
        reason='Earlier initialG interventions used the common20s field, whose K9 response differs markedly from the actualexit field. They cannot settle the carried-G explanation here. Natural exit carriesGraw about11.5 whereas the held high state is about4.4.',
        design=dict(new_branches=2,duration_s=30.,baseline_reused=str(BASE/'runs'/BASE_NAME),
            initial_G_raw=[0.,Gactual],baseline_initial_G_raw=Gsource,Z_mean=.21,K_mean=9.,
            history='Identical full12s high endogenous state including voltage, currents, queued spikes, M, refractory and causalR.',
            future_input='Same copied50s exogenous OU/Poisson/RNG state as existing actualK9 high baseline.',
            endogenous_G_and_M_remain_dynamic=True,source_seed=9108405),
        implementation='Reuse the already verified spatial-probe and ordered CUDA native engine. Before launch independently reconstruct the initial engine and compare every field; only the declared G scalar differs after the usual identical Z/K clamps and futureinput handoff.',
        readouts='First0-1/1-5/5-10s and last20-30s rates in allE/coreA/coreB/surround; full1msR/G and400-cell spatial fields; time spent below5Hz; counterfactualZdrift. Retain truncated events and use the frozen observer. Check recorded futureinputs bitwise against baseline.',
        interpretation='If carriedG changes the final conditional response while a high branch is still present, it supports history-dependent selection; it does not by itself certify a basin boundary or autonomous termination. If both controls reconverge high, initialG alone in this fixedZ/K family is insufficient; natural Z/K evolution and other histories remain distinct possibilities.',
        limits='One shared native noise/history, not new independent seeds. These are explicit conditional initial-state interventions with heldZ/K, do not count as autonomous exits or recovery, and do not require every seed to exit. No additionalGscan/extension is automatic.',
        producer_sha256=sha(__file__),counts_as_autonomous_loop=False,formal_bifurcation_allowed=False))
    rows=[]
    for tag,G in [('zero',0.),('carried16p7',Gactual)]:
        rows.append(dict(name=f'exit_z0.21_k9_actualfield_G_{tag}',Z=.21,K=9.,duration_s=30.,cut='exit',
            history_label='high',history_checkpoint=job['source_checkpoint'],Z_template=job['Z_template'],K_template=job['K_template'],
            G_raw_override=G,reason='Only initialG changes within actualexitfield family; distinguish carriedfeedback history from conditional equilibrium disappearance.'))
    spec=dict(stage='ACTUAL_EXIT_FIELD_INITIAL_G_CONTROLS',question=read(OUT/'scientific_contract.json')['question'],rows=rows,max_workers=2)
    write(OUT/'probe_spec.json',spec);probes.prepare(OUT,spec)
    with np.load(job['held_fields_file']) as z:Z=z['Z'];K=z['K']
    expected=copy.deepcopy(source['engine']);expected['slow']['kind']='ConditionalSlow';expected['slow']['z'][:32000]=Z;expected['termination_mechanism']['sahp_g'][:]=K
    common=probes.native.read_pickle(__import__('pathlib').Path(job['external_noise_source']))['engine']
    for key in ['rng_state','xi','external_drive']:expected[key]=copy.deepcopy(common[key])
    offset=expected['step']-common['step']
    for key in ['next_step','last_step']:expected['external_drive'][key]+=offset
    checks=[]
    for row in rows:
        newjob=read(OUT/'jobs'/f"{row['name']}.json")
        with np.load(newjob['held_fields_file']) as z:
            assert np.array_equal(z['Z'],Z) and np.array_equal(z['K'],K)
        saved=probes.native.read_pickle(OUT/'runs'/row['name']/'checkpoint.pkl')
        assert saved['identity']==source['identity']
        want=copy.deepcopy(expected);want['global_feedback_response']['global_state']=row['G_raw_override']/30.
        assert_same_state(want,saved['engine'])
        checks.append(dict(name=row['name'],entire_declared_initial_engine_bitwise=True,held_fields_match_baseline_bitwise=True,
            initial_R_Hz=float(saved['engine']['termination_mechanism']['r_global']),initial_G_raw=row['G_raw_override']))
    write(OUT/'initial_state_qa.json',dict(status='PASS',checks=checks,baseline_reused=BASE_NAME,
        simulator_physics_unchanged=True,counts_as_autonomous_loop=False))
    print('ACTUAL G HISTORY INITIAL STATE PASS',checks,flush=True)


if __name__=='__main__':main()
