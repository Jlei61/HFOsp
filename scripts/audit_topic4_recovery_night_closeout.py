"""Close the physical ledger without turning completion into scientific acceptance."""
import csv
import time
from pathlib import Path
from scripts import analyze_topic4_propagation_recovery_night as s


def main():
    out=s.night.OUT
    records=[];phases=[];sources=[];identities=[]
    for phase in ['wave1','long','final_A','final_B']:
        spec=s.rt.read(out/f'{phase}_units.json')
        if phase=='wave1':
            # The first phase predates the consolidated analysis driver.
            assert not s.rt.read(out/'analysis_wave1/review_state.json')['missing']
            assert s.rt.read(out/'main_review_wave1/manifest.json')['status']=='RENDERED_PENDING_SCIENTIFIC_REVIEW'
            assert s.rt.read(out/'distribution_review_wave1/manifest.json')['status']=='COMPLETE_PENDING_SCIENTIFIC_REVIEW'
            assert s.rt.read(out/'local_burst_audit_wave1/audit.json')['status']=='COMPLETE_READ_ONLY'
        else:
            assert s.rt.read(out/f'{phase}_analysis_complete.json')['status']=='COMPLETE_PENDING_SCIENTIFIC_REVIEW'
        actual={(c,int(t),int(d)) for c,t,d in spec['units']}
        with (out/f'analysis_{phase}'/'run_observations.csv').open() as f:observed=list(csv.DictReader(f))
        selected=[x for x in observed if (x['base_id'],int(x['topology_seed']),int(x['seed'])) in actual]
        total={layer:sum(int(x['n']) for x in selected if x['layer']==layer and x['mode']=='ALL') for layer in ['primary','all_detected']}
        for c,t,d in spec['units']:
            p=s.an.run.result_path(spec['stage'],c,t,d);r=s.rt.read(p);a=s.rt.read(p.parent.parent/'applied_physics.json')
            assert r['status']=='COMPLETE' and r['physical_status']=='COMPLETE_NO_RUNAWAY'
            assert r['actual_duration_ms']==spec['duration_ms']
            assert a['threshold']['n_raised']==0
            assert a['input']['n_stochastic']==a['threshold']['n_members']
            assert all(a['input'][key]=='off' for key in ['spatial_ou','slow_I_state','ZM','kick'])
            assert r.get('maximum_outside_rate_deviation',0)==0
            if phase=='final_B':
                q=r['core_ou_mixture_audit']
                assert q['intermediate_global_clip_steps']==q['final_negative_core_steps']==0
                # Algebraically identical sums may differ at float64 roundoff.
                assert max(q['maximum_core_rate_error'].values())<1e-12 and q['maximum_I_rate_error']==0
                cand=s.rt.read(s.an.run.OUT/'candidates'/f'{c}.json');parent=s.rt.read(s.an.run.OUT/'candidates'/f"{cand['parent_id']}.json")
                pr=s.rt.read(s.an.run.result_path(parent['stage'],parent['id'],t,d))
                assert r['static_array_identity']==pr['static_array_identity']
                identities.append(dict(candidate=c,seed=d,parent=parent['id'],same_static_arrays=True,
                    metadata_note='Inherited adjacency_changes/geometry_change fields describe an ancestor, not this rho intervention. Actual static identity equals the immediate parent.'))
            records.append(dict(phase=phase,candidate=c,topology=t,seed=d,duration_ms=r['actual_duration_ms'],
                status=r['status'],physical_status=r['physical_status'],n_lowered=a['threshold']['n_lowered'],
                n_raised=a['threshold']['n_raised'],n_core_E=a['threshold']['n_members'],
                path=str(p),json_sha256=s.rt.sha(p),arrays_sha256=r['arrays_sha256'],peak_rss_gib=r['peak_rss_gib']))
        phases.append(dict(phase=phase,runs=len(actual),duration_ms=spec['duration_ms'],**total))
    assert len(records)==48 and len({(r['phase'],r['candidate'],r['topology'],r['seed']) for r in records})==48
    for plan in ['plan.json','tonic_input_plan.json','core_ou_correlation_plan.json']:
        obj=s.rt.read(out/plan)
        for file,expected in obj.get('source_snapshot',{}).items():
            actual=s.rt.sha(file);assert actual==expected,(plan,file)
            sources.append(dict(plan=plan,path=file,sha256=actual))
    result=dict(status='PASS_PHYSICAL_AND_ANALYSIS_CLOSEOUT',completed_unix=time.time(),new_formal_runs=48,old_formal_runs=120,
        all_formal_runs=168,new_simulated_seconds=sum(r['duration_ms'] for r in records)/1000,
        engineering_canary_runs=6,engineering_simulated_seconds=3,
        new_topology_identities=sorted({r['topology'] for r in records}),dynamics_seed_identities=sorted({r['seed'] for r in records}),
        phases=phases,records=records,frozen_source_checks=sources,final_B_static_identity_checks=identities,
        scientific_acceptance='Not determined by this audit; see scientific_review.md.',user_visual_acceptance=False,
        event_count_boundary='Counts are trajectory records, not independent patient/network samples. Two 20s/60s prefix replays overlap; reference runs repeated in analysis folders are excluded from ledger totals.',
        parameter_inference='Only one topology identity and two noise identities were exercised in the new48. Changes of geometry/kernel/weights create distinct applied arrays; no new-topology confirmation.',
        producer=__file__,producer_sha256=s.rt.sha(__file__))
    s.rt.write(out/'physical_analysis_closeout.json',result)
    print({k:result[k] for k in ['status','new_formal_runs','all_formal_runs','new_simulated_seconds','phases']},flush=True)


if __name__=='__main__':main()
