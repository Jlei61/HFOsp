#!/usr/bin/env python3
"""Interim assessment only after all prespecified extra seeds finish per geometry."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_topic4_joint_xy_kernel_search as run


def main():
    out=run.OUT;plan=run.read(run.CONFIG);obj=run.KernelObjective(run.v1,out,run.KERNEL);cal=run.read(out/'patient_calibration.json')
    initial={r['candidate_id']:r for r in run.read(out/'baseline_scores.json')['candidates']}
    ids=run.read(out/'rounds/000/race_nomination.json')['candidate_ids'];directory=out/'execution/race_000/workers'
    folder=out/'completed_racer_review';folder.mkdir(exist_ok=True);report=[]
    for cid in ids:
        paths=[directory/f'{cid}_seed_{s}.json' for s in plan['search']['race_seeds']]
        if not all(p.exists() for p in paths):continue
        source={**initial[cid],'units':initial[cid]['units']+[
            {'worker_path':str(p),'worker_sha256':run.sha(p)} for p in paths]}
        combined=run.score_saved([source],obj,plan,folder/f'{cid}.json')[0]
        report.append({'candidate_id':cid,'n_networks':len(combined['units']),
            'n_events_per_seed':[u['metrics']['n_events'] for u in combined['units']],
            'initial_n_events':initial[cid]['n_events'],'expanded_n_events':combined['n_events'],
            'initial_joint_distance':initial[cid]['joint_distance'],'expanded_joint_distance':combined['joint_distance'],
            'assessment':run.assess(combined,cal,plan),'D_order':combined['D_order'],'D_lag':combined['D_lag'],
            'direction_distance':combined['direction_distance']})
    run.write(folder/'summary.json',{'status':'INTERIM_COMPLETE_GEOMETRIES_ONLY','updated_unix':time.time(),'rows':report,
        'n_nominated':len(ids),'nomination_changed':False,'live_search_modified':False,
        'source_hashes':{str(p):run.sha(p) for p in [Path(__file__),out/'rounds/000/race_nomination.json',out/'baseline_scores.json']}})
    print(report)


if __name__=='__main__':main()
