"""Prepare deterministic design artifacts only; deliberately does not dispatch workers."""
from pathlib import Path
import json
import csv
import copy
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/core_connectivity_search_design_20260910'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    radius = 1.7525782905619196
    baseline = dict(EE_same_core_scale=1., EE_core_to_out_scale=1., EE_out_to_out_scale=1.,
        EI_same_core_scale=1., IE_same_core_scale=1., II_same_core_scale=1.,
        depth_A_scale=1., depth_B_scale=1., radius_A_mm=radius, radius_B_mm=radius,
        EE_core_to_out_degree_scale=1., EE_kernel_perp_scale=1., EE_kernel_parallel_scale=1.,
        EE_angle_offset_deg=0.)
    axes = {
        'EE_same_core_scale':([.75,1.25],[.6,1.5]),
        'EE_core_to_out_scale':([.75,1.25],[.6,1.6]),
        'EE_out_to_out_scale':([1.15],[.8,1.2]),
        'EI_same_core_scale':([.75,1.25],[.6,1.4]),
        'IE_same_core_scale':([.75,1.25],[.6,1.4]),
        'II_same_core_scale':([.75],[.6,1.4]),
        'depth_A_scale':([1.5],[.5,2.]), 'depth_B_scale':([1.5],[.5,2.]),
        'radius_A_mm':([2.5],[1.4,3.2]), 'radius_B_mm':([2.5],[1.4,3.2]),
        'EE_core_to_out_degree_scale':([1.5],[.7,1.6]),
        'EE_kernel_perp_scale':([1.5],[.75,2.]),
        'EE_kernel_parallel_scale':([1.5],[.75,1.7]),
        'EE_angle_offset_deg':([-15.,15.],[-20.,20.])}
    layouts = [
        dict(id='endpoint',old_id='base_off',A=[4.19921432,9.12890135]),
        dict(id='up4p5',old_id='A_up_4p5_off',A=[4.19921432,13.62890135]),
        dict(id='near_upper',old_id='A_near_upper_SCL_off',A=[5.712184916450571,13.574047194943589])]
    for layout in layouts:
        layout['B']=[16.47920304,3.96551153]
    candidates=[]
    for layout in layouts:
        variants=[('baseline',None,None)]+[(f'{key}_{value:g}',key,value) for key,(values,_) in axes.items() for value in values]
        for suffix,key,value in variants:
            params=copy.deepcopy(baseline)
            if key is not None:params[key]=value
            candidates.append(dict(id=f'{layout["id"]}__{suffix}',layout=layout['id'],
                centers_mm=[layout['A'],layout['B']],parameters=params,changed_parameter=key,changed_value=value,
                adjacency_changes=key in {'EE_core_to_out_degree_scale','EE_kernel_perp_scale','EE_kernel_parallel_scale','EE_angle_offset_deg'}))
    units=[dict(candidate=c['id'],layout=c['layout'],topology_seed=2511,dynamics_seed=s,
        duration_ms=20000,changed_parameter=c['changed_parameter'] or 'baseline',changed_value=c['changed_value'])
        for c in candidates for s in [847101,847102]]
    assert len(candidates)==60 and len(units)==120
    assert len({c['id'] for c in candidates})==60
    for c in candidates:
        for center,k in zip(c['centers_mm'],['A','B']):
            r=c['parameters'][f'radius_{k}_mm']
            assert all(r<=v<=20-r for v in center)
        assert math.dist(*c['centers_mm'])>c['parameters']['radius_A_mm']+c['parameters']['radius_B_mm']
        assert sum(c['parameters'][k]!=v for k,v in baseline.items())==(c['changed_parameter'] is not None)
    design=dict(schema='topic4.core_connectivity_search.DESIGN_ONLY.v1',status='DESIGN_PREPARED_NOT_DISPATCHED',
        legacy_worker_compatible=False,physics_implemented=False,
        physics=dict(version='core_connectivity_v2_PROPOSED',E_threshold_delta_max_mV=0,
            core_threshold_floor_mV=11,noise_support='E in the union of both geometric cores',
            outside_external_input='deterministic expected arrivals through unchanged synaptic filter',
            shared_core_OU=True,spatial_OU=False,slow_I_state=False,ZM=False,kick=False,
            GABA_decay_ms=18,reference_EE_angle_deg=-22.80538396505847,
            reference_EE_kernel_parallel_mm=.38*math.sqrt(2),reference_EE_kernel_perp_mm=.38/math.sqrt(2)),
        scope_definitions=dict(EI='E presynaptic -> I postsynaptic',IE='I presynaptic -> E postsynaptic',
            same_core='within A plus within B, excluding cross-core edges',out='outside both geometric cores',
            kernel_intervention='all E->E, same total indegree; actual edges and geometric delays change'),
        baseline_parameters=baseline,axes={k:dict(screen_values=v[0],adaptive_bounds=v[1]) for k,v in axes.items()},
        layouts=layouts,screen_candidates=candidates,screen_units=units,
        budget=dict(canary_max_units=6,canary_duration_ms=500,screen=120,adaptive=32,confirmation=16,total_formal=168),
        analysis=dict(burnin_ms=1500,minimum_events_per_training_replay=16,
            primary_score='0.5*frozen_L_off + 0.5*joint_mask_Doff/frozen_positive_patient_block_scale',
            mask_kernel_Hamming_bandwidths=[1,3,6],mask_target='full patient FIT natural frequencies',
            score_status='SPECIFIED_NOT_IMPLEMENTED_OR_CALIBRATED',
            forbidden=['negative loss clipping','per-event route stimuli','rank-reordering displayed contacts','time stretching','lineage-only training readout']),
        adaptive=dict(populations=2,parent_size=8,offspring_per_population_per_batch=4,batches=2,
            F=.6,CR=.7,rng_seed=934711,maximum_active_dimensions=6,update='deferred within batch; second batch uses replaced parents',
            center_offset_bounds_mm=[-.75,.75],polish=False),
        confirmation=dict(conditions=4,topologies=2,dynamics_per_topology=2,duration_ms=60000,
            seed_allocation='freeze unused explicit topology/dynamics identities before this stage'),
        resources=dict(initial_workers=2,max_workers=8,min_available_GiB=40,accounting='real process trees including graph builds and plotting'),
        historical_bridge=dict(root='/data/hfosp/topic4_sef_hfo/core_position_scl_response_20260910',
            candidate_ids=[x['old_id'] for x in layouts],reuse_old_outputs_only=True),
        stop='scientific review after bounded stages; no automatic extension, model freeze, or Fig5')
    (OUT/'search_design.json').write_text(json.dumps(design,indent=2,ensure_ascii=False))
    with (OUT/'screen_units.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(units[0]));w.writeheader();w.writerows(units)
    (OUT/'design_validation.json').write_text(json.dumps(dict(status='DESIGN_CHECKS_PASS',
        unique_conditions=len(candidates),paired_units=len(units),one_factor_screen=True,
        no_core_clipping_or_overlap_in_screen=True,physics_tested=False,simulation_dispatched=False),indent=2))
    print(json.dumps(dict(conditions=len(candidates),screen_runs=len(units),formal_budget=168,status=design['status'])))

if __name__=='__main__':main()
