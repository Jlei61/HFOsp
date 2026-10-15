"""Offline diagnostics and genuine conditional XY response maps; no optimizer input."""
from pathlib import Path
import argparse, csv, json, pickle, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_topic4_xy_wide_response as run
from scripts import analyze_topic4_three_observable_bo as legacy
from scripts.report_topic4_label_free_dense import describe, metrics
rt, OUT=run.rt, run.OUT


def main(final_media=False):
    p=rt.read(OUT/'plan.json'); scores={f.stem:rt.read(f) for f in (OUT/'scores').glob('*.json')}
    ev,names,_=legacy.patient(); cache=OUT/'analysis/diagnostic_observables.json'
    data=rt.read(cache) if cache.exists() else dict(patient={},runs={})
    modes=[('ALL',None),('TA',1),('TB',0)]
    if not data['patient']:
        for mode,k in modes: data['patient'][mode]=describe(ev.fit if k is None else ev.fit[ev.fit_labels==k],names)
    joint_path=OUT/'analysis/offline_joint_scores.json'
    joint=rt.read(joint_path) if joint_path.exists() else {}
    audit_dir=OUT/'joint_feature_audit'
    joint_ready=(audit_dir/'summary.json').exists()
    if joint_ready:
        from scripts.audit_topic4_joint_observation import JointMap, statistic
        with Path(p['objective']['artifact']).open('rb') as f: objective=pickle.load(f)
        mapping=JointMap(objective)
        with np.load(audit_dir/'feature_map.npz') as z: mapping.W=z['weights'];mapping.phase=z['phases']
        with np.load(audit_dir/'target.npz') as z: target=z['target'];scale=float(z['scale'])
        device=run.dense.gpu_device()
    for cid,score in scores.items():
        if cid in data['runs'] and (not joint_ready or cid in joint): continue
        src=Path(score['source']); r,t,cn=run.dense.load_times(src); assert cn==names
        if joint_ready and cid not in joint:
            if len(t)>=16 and score['J'] is not None:
                value=statistic(mapping.features(objective.raw_features(t),device),target)
                joint[cid]=dict(**value,scaled=value['D_off']/scale,diagnostic_only=True)
            else: joint[cid]=dict(N=len(t),scaled=None,diagnostic_only=True)
        with np.load(src.with_suffix('.npz')) as z:
            ids=[int(i) for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']]
            lab=z['event_mode'][ids]
        data['runs'][cid]={mode:describe(t if k is None else t[lab==k],names) for mode,k in modes}
    rt.write(cache,data)
    if joint_ready:rt.write(joint_path,joint)
    rows=[]
    for cid,score in scores.items():
        c=rt.read(OUT/'candidates'/f'{cid}.json')
        for mode,_ in modes:
            q=data['runs'][cid][mode]; pat=data['patient'][mode]; v=metrics(q,pat,names)
            row=dict(candidate=cid,mode=mode,N=q['N'],J=score['J'],rank_error=v[0],within_rod_order_error=v[1],
                rod_lag_ms=v[2],participation_error=v[3],both_rods=v[4],
                left_x=c['centers_mm'][0][0],left_y=c['centers_mm'][0][1],right_x=c['centers_mm'][1][0],right_y=c['centers_mm'][1][1])
            row['offline_joint_score_ALL_only']=joint.get(cid,{}).get('scaled')
            for shaft in ['SCL','ICL']:
                vals=[abs(z['order_probability']-pat['pairs'][k]['order_probability']) for k,z in q['pairs'].items()
                      if z['shaft']==shaft and z['order_probability'] is not None and pat['pairs'][k]['order_probability'] is not None]
                row[shaft+'_order_error']=float(np.mean(vals)) if vals else None
            rows.append(row)
    with (OUT/'analysis/observables.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    good=[s for s in scores.values() if s['J'] is not None]
    best=min(good,key=lambda s:s['J'])
    rt.write(OUT/'analysis/current.json',dict(best=best,incumbent_J=scores[run.REF]['J'],complete=len(scores),
        scientific_status='PENDING_PROPAGATION_REVIEW',mode_labels='diagnostic only; not read by controller',
        best_diagnostics=[r for r in rows if r['candidate']==best['candidate']]))
    # Render the wider A from measured trajectories only. B/C/D keep their history.
    from scripts.paper_figures import plot_topic4_shared_plane_position_response as plate
    plate.OUT=OUT/'key_results/shared_plane_position_response'
    plate.FIG=plate.OUT/'figures'
    plate.main(wide_dir=OUT)
    rt.write(OUT/'analysis/figure_metadata.json',dict(producer=str(Path(__file__).resolve()),
        source_csv=str(OUT/'analysis/observables.csv'),
        composite=str(plate.FIG/'parameter_search_propagation_shared_plane.png'),
        statistical_unit='one 60s fixed-topology/fixed-noise trajectory per condition',
        interpolation=False,human_review='pending',formats=['png','editable_svg']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--final-media',action='store_true')
    main(ap.parse_args().final_media)
