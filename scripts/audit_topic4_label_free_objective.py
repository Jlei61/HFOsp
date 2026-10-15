"""Re-score an identical frozen pool; do not launch or alter the completed search."""
from pathlib import Path
import sys,json,pickle,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scipy.stats import spearmanr
from scripts import analyze_topic4_three_observable_bo as a
from src.topic4_three_observable_label_free import LabelFreeThreeObservableObjective
from src.topic4_three_observable_objective import GROUPS
D=Path('/data/hfosp/topic4_sef_hfo/label_free_objective_audit_20260915')

def main():
    D.mkdir(parents=True,exist_ok=True);(D/'per_unit').mkdir(exist_ok=True)
    old=a.load_objective();old_sha=a.rt.sha(a.A/'training_objective.pkl');ev,names,_=a.patient()
    obj=LabelFreeThreeObservableObjective(old)
    def forbidden(*args,**kwargs):raise RuntimeError('Classifier called during label-free scoring')
    old.km.predict=forbidden;old.labels=forbidden
    assert not any(hasattr(obj,k) for k in ['km','proportions','mode_means'])
    obj.fit_targets(ev.fit,device='cuda:0')
    for g in GROUPS:np.testing.assert_allclose(obj.targets[g],old.targets[g]['global_mean'],rtol=0,atol=1e-10)
    calibration=a.rt.read(a.A/'positive_scale_calibration.json');cal=ev.patient[np.asarray(calibration['parent_indices'],int)]
    assert len(cal)==4981
    cal_out=obj.calibrate(cal,calibration['draws'],device='cuda:0')
    a.rt.write(D/'label_free_scale_calibration.json',cal_out)
    with (D/'label_free_objective.pkl').open('wb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    rows=[];max_error=0.
    decomp_dir=a.OUT/'overnight_20260914/score_decomposition'
    # The computation below never opens event_mode or asks a classifier.
    for p in sorted((a.OUT/'scores').glob('*.json')):
        saved=a.rt.read(p)
        if saved['J'] is None:continue
        source=Path(saved['source']);r=a.rt.read(source)
        with np.load(source.with_suffix('.npz')) as z:
            times=z['centroid_ms'];ids=np.array([i for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
            assert z['contact_names'].tolist()==names
        sc=obj.score(times[ids],device='cuda:0');assert sc['N']==saved['N']
        cached=a.rt.read(decomp_dir/p.name)
        for g in GROUPS:
            equivalent=2*3*old.scales[g]*cached['per_group'][g]['global_features']
            error=abs(sc['groups'][g]['D_off']-equivalent);max_error=max(max_error,error)
            np.testing.assert_allclose(sc['groups'][g]['D_off'],equivalent,rtol=0,atol=1e-9)
        fixed=float(np.mean([sc['groups'][g]['D_off']/old.scales[g] for g in GROUPS]))
        record=dict(candidate=saved['candidate'],stage=saved['stage'],topology=saved['topology'],noise=saved['noise'],N=sc['N'],with_label_J=saved['J'],delete_labels_keep_old_scale_J=fixed,label_free_J=sc['J'],groups=sc['groups'],source=str(source),source_score_sha256=a.rt.sha(p))
        a.rt.write(D/'per_unit'/p.name,record);rows.append(record)
    assert len(rows)==72
    train=[r for r in rows if r['stage'] in ['initial','adaptive']];cond=[]
    for cid in sorted({r['candidate'] for r in train}):
        rr=sorted([r for r in train if r['candidate']==cid],key=lambda r:r['noise']);assert len(rr)==2
        cond.append(dict(candidate=cid,label=a.plot_label(cid),description=a.point_label(cid),**{k:float(np.mean([r[k] for r in rr])) for k in ['with_label_J','delete_labels_keep_old_scale_J','label_free_J']},per_noise=rr))
    ranks={}
    for key in ['with_label_J','delete_labels_keep_old_scale_J','label_free_J']:
        ranks[key]=[r['candidate'] for r in sorted(cond,key=lambda r:r[key])]
        for r in cond:r[key+'_rank']=ranks[key].index(r['candidate'])+1
    # Post-ranking validation may inspect old frozen labels; none fed into obj.
    raw={ (r['candidate'],r['topology'],r['noise']):r for r in a.records()}
    chosen=list(dict.fromkeys([a.rt.read(a.OUT/'plan.json')['reference_id'],ranks['with_label_J'][0],ranks['delete_labels_keep_old_scale_J'][0],ranks['label_free_J'][0]]))
    validation=[]
    for cid in chosen:
        for r in [r for r in train if r['candidate']==cid]:
            q=raw[(cid,r['topology'],r['noise'])]
            validation.append(dict(candidate=cid,topology=r['topology'],noise=r['noise'],N=r['N'],mode_counts=q['mode_counts'],raw=q['raw']))
    rho=float(spearmanr([r['with_label_J_rank'] for r in cond],[r['label_free_J_rank'] for r in cond]).statistic)
    summary=dict(time=datetime.datetime.now().astimezone().isoformat(),status='OFFLINE_SAME_POOL_AUDIT_COMPLETE_NOT_NEW_OPTIMIZATION',source_round=str(a.OUT),training_conditions=len(cond),scored_runs=len(rows),old_scales=old.scales,new_label_free_scales=obj.scales,conditions=cond,rankings=ranks,rank_spearman=rho,top5_overlap=len(set(ranks['with_label_J'][:5])&set(ranks['label_free_J'][:5])),post_rank_validation=validation,confirmation=[r for r in rows if r['stage']=='confirmation'],label_free_interface_fields=list(obj.__dict__),classifier_bomb_test_passed=True,raw_statistic_vs_cached_global_max_error=max_error,old_objective_sha256=old_sha,new_objective_sha256=a.rt.sha(D/'label_free_objective.pkl'),limitations=['same old-label-guided candidate pool, not an equal-budget new optimization','old scales retained only for isolated term deletion; fully label-free comparator recalibrates on same CAL draw IDs','newly selected candidates lack the completed winner new-topology confirmation; no new physical runs','coarse spatial initialization still uses patient-mode endpoint geometry, hence objective-label-free is not an entirely label-free scientific pipeline'])
    assert a.rt.sha(a.A/'training_objective.pkl')==old_sha
    a.rt.write(D/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','old_scales','new_label_free_scales','rank_spearman','top5_overlap','raw_statistic_vs_cached_global_max_error']},ensure_ascii=False),flush=True)
    print('SELECTED',[(r['candidate'],r['with_label_J_rank'],r['delete_labels_keep_old_scale_J_rank'],r['label_free_J_rank']) for r in cond if r['candidate'] in chosen],flush=True)
    for r in validation:print('VALIDATION',r['candidate'],r['noise'],r['mode_counts'],r['raw'],flush=True)

if __name__=='__main__':main()
