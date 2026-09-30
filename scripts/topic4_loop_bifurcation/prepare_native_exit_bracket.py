#!/usr/bin/env python3
"""Native correspondence at the two sides of the held density transition."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import copy,time
from pathlib import Path
import numpy as np
from campaign import ROOT,read,write,sha
import spatial_probes as probes
from run_topic4_recovery_window import assert_same_state

OUT=ROOT/'native_exit_K_bracket'
BASE=ROOT/'exit_return_probes'
NAME='exit_z0.21_k9_fields16p7_high'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'scientific_contract.json').exists()
    lower=read(ROOT/'carried_exit_lower_holds/analysis/result.json')
    higher=read(ROOT/'carried_exit_fixed_holds/analysis/result.json')
    assert all(q['both_cores_active_in_last3s'] for q in lower['rows'])
    assert all(q['both_cores_low_in_last3s'] for q in higher['rows'])
    job=read(BASE/'jobs'/f'{NAME}.json')
    source=probes.native.read_pickle(Path(job['source_checkpoint']))
    write(OUT/'scientific_contract.json',dict(status='REGISTERED_BEFORE_NATIVE_COUNTS',created_epoch=time.time(),
        question='Does the original native network retain a high state or become quiet near the density held-state transition between K9.35 and9.5, with the same actual exit-field family?',
        selection='K9.35 remained active in the complete density8s hold; K9.5 became quiet. These are adaptive correspondence points, not independent discovery or a formal bifurcation interval.',
        design=dict(new_branches=2,duration_s=30.,K_values=[9.35,9.5],Z_mean=.21,
            source_history='Same full12s high state as existing K9 andK10.5 native controls.',
            future_input='Same50s exogenous OU/Poisson/RNG handoff as previous native controls.',
            field_family='Actual16.7s Z andK fields transformed by the same mapping.',
            numerical_density_protocol_difference='Native K is set immediately at the branch start; density carried its own high state through a0.15/s ramp. Compare finite conditional states and spatial fields, not identical histories or exit timing.'),
        readouts='Complete30s; tail20-30s core/allE/surround rates and400cell field, actualG, counterfactualZdrift, lowrate intervals and events using the existing observer.',
        interpretation='State correspondence is required before extending a density branch towards native exit. Disagreement triggers diagnosis of history/noise/closure; no automatic criticalK assignment, bifurcation type, broad sweep or tolerance change.',
        statistical_unit='One shared endogenous history and common future input; two parameter interventions, no new independent seeds.',
        counts_as_autonomous_loop=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__)))
    rows=[]
    for K in [9.35,9.5]:
        rows.append(dict(name=f'exit_z0.21_k{K:g}_fields16p7_high',Z=.21,K=K,duration_s=30.,cut='exit',
            history_label='high',history_checkpoint=job['source_checkpoint'],Z_template=job['Z_template'],K_template=job['K_template'],
            reason='Native correspondence at completed density held-state bracket, without changing network physics.'))
    spec=dict(stage='NATIVE_CORRESPONDENCE_AT_DENSITY_EXIT_BRACKET',question=read(OUT/'scientific_contract.json')['question'],rows=rows,max_workers=2)
    write(OUT/'probe_spec.json',spec);probes.prepare(OUT,spec)
    with np.load(job['held_fields_file']) as z:Z=z['Z'];K9=z['K']
    expected=copy.deepcopy(source['engine']);expected['slow']['kind']='ConditionalSlow';expected['slow']['z'][:32000]=Z
    common=probes.native.read_pickle(Path(job['external_noise_source']))['engine']
    for key in ['rng_state','xi','external_drive']:expected[key]=copy.deepcopy(common[key])
    offset=expected['step']-common['step']
    for key in ['next_step','last_step']:expected['external_drive'][key]+=offset
    checks=[]
    for row in rows:
        saved=probes.native.read_pickle(OUT/'runs'/row['name']/'checkpoint.pkl')
        with np.load(OUT/'runs'/row['name']/'held_fields.npz') as z:Znew=z['Z'];Knew=z['K']
        assert np.array_equal(Znew,Z)
        assert np.allclose(Knew,K9*(row['K']/9.),rtol=1e-14,atol=1e-14)
        want=copy.deepcopy(expected);want['termination_mechanism']['sahp_g'][:]=Knew
        assert_same_state(want,saved['engine'])
        checks.append(dict(name=row['name'],complete_initial_engine_bitwise=True,meanK=float(Knew.mean()),
            meanZ=float(Znew.mean()),only_new_declared_parameter='heldK'))
    write(OUT/'initial_state_qa.json',dict(status='PASS',checks=checks,simulator_physics_unchanged=True))
    print('NATIVE EXIT BRACKET PREPARED',flush=True)


if __name__=='__main__':main()
